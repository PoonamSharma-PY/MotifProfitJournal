"""rl_motif_algorithm.py  (motif-aware)
====================================

RL seed selection for MOTIF-ORIENTED profit maximization, where BOTH the graph
structure AND the motif structure drive the learning.

The learned representation is a two-channel Structure2Vec over a node-motif
bipartite structure:

  * graph channel  : a node aggregates embeddings from its graph neighbours
                     (the diffusion structure).
  * motif channel  : a node aggregates embeddings of the motifs it belongs to,
                     and each motif aggregates embeddings of its member nodes.
                     Each motif also carries state features: its benefit, how
                     close it currently is to the activation threshold, and
                     whether it is already covered.

So Q(s, v) reflects "how many still-uncovered motifs would picking v help",
not just "how influential is v in the graph". The reward is unchanged and still
equals your motif profit (diffuse with ICM, then score against the motifs), so
the agent is trained to select the node that influences the most motifs.

MDP recap:
  state  (G, motifs, S_t)   ; action  add an affordable unselected node
  reward [Phi(S_{t+1}) - Phi(S_t)] - cost(v),  Phi(S) = E_ICM[influenced motif value]
  a motif is INFLUENCED when >= THRESHOLD of its nodes activate.
"""

import ast
import time
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                       # pragma: no cover
    HAVE_NUMBA = False
    def njit(f=None, **k):
        return (f if f else (lambda g: g))

import torch
import torch.nn as nn


# ----------------------------- config -------------------------------------
@dataclass
class RLConfig:
    threshold: int = 2
    objective: str = "profit"     # "profit" (benefit-weighted) or "count" (# motifs)
    reach_hops: int = 4           # how many hops the motif-influence feature looks ahead
    crn: bool = True              # common-random-numbers marginals (low-variance reward;
                                  #   biggest help on heterogeneous graphs like trivalency)
    emb_dim: int = 64
    t_layers: int = 3
    lr: float = 1e-3
    gamma: float = 0.99
    n_step: int = 2
    episodes: int = 200
    batch_size: int = 32
    buffer_cap: int = 5000
    reward_sims: int = 16
    eval_sims: int = 64
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 150
    target_sync: int = 20
    seed: int = 0
    verbose: bool = True
    # --- optional levers (default OFF; last turn's regression came from turning
    #     several of these on at once). Enable and test ONE at a time. ---
    budget_lo: float = None       # train samples budget in [lo,hi] each episode
    budget_hi: float = None       #   (budget-aware; helps low budgets, can flatten high)
    huber: bool = False           # smooth-L1 loss instead of MSE
    double_dqn: bool = False      # online selects, target evaluates
    grad_clip: float = 0.0        # gradient-norm clip (0 = off)
    scale_reward: bool = False    # divide reward by max motif value
    hybrid_topk: int = 0          # inference: Q proposes top-k, MC picks best (0=greedy)
    hybrid_sims: int = 64


# ----------------------------- ICM ----------------------------------------
@njit
def icm_mask(seed_arr, adj, prob, rng_seed):
    if rng_seed >= 0:
        np.random.seed(rng_seed)
    n = len(adj)
    active = np.zeros(n, dtype=np.bool_)
    newly = np.zeros(n, dtype=np.bool_)
    for s in seed_arr:
        active[s] = True
        newly[s] = True
    while np.any(newly):
        nxt = np.zeros(n, dtype=np.bool_)
        for u in range(n):
            if newly[u]:
                for j in range(adj.shape[1]):
                    v = adj[u, j]
                    if v == -1:
                        break
                    if not active[v] and np.random.rand() < prob[u, j]:
                        active[v] = True
                        nxt[v] = True
        newly = nxt
    return active


@njit
def sample_live(adj, prob, rng_seed):
    """Sample one live-edge graph: a coin per edge, independent of traversal.
    Reusing this same live graph for S and S+{v} gives common random numbers."""
    if rng_seed >= 0:
        np.random.seed(rng_seed)
    live = np.zeros(adj.shape, dtype=np.bool_)
    for u in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[u, j] == -1:
                break
            live[u, j] = np.random.rand() < prob[u, j]
    return live


@njit
def reach_live(seed_arr, adj, live):
    """Nodes reachable from the seeds along LIVE edges (deterministic BFS)."""
    n = len(adj)
    active = np.zeros(n, dtype=np.bool_)
    newly = np.zeros(n, dtype=np.bool_)
    for s in seed_arr:
        active[s] = True
        newly[s] = True
    while np.any(newly):
        nxt = np.zeros(n, dtype=np.bool_)
        for u in range(n):
            if newly[u]:
                for j in range(adj.shape[1]):
                    v = adj[u, j]
                    if v == -1:
                        break
                    if live[u, j] and not active[v]:
                        active[v] = True
                        nxt[v] = True
        newly = nxt
    return active


@njit
def motif_reach(adj, prob, target):
    """Weighted one-hop reach of each node into a target set: for node v,
    sum over out-edges of prob(v,u) * target[u]. Lets an OUTSIDE node be credited
    for its ability to influence motif members it doesn't belong to."""
    n = len(adj)
    reach = np.zeros(n)
    for v in range(n):
        s = 0.0
        for j in range(adj.shape[1]):
            u = adj[v, j]
            if u == -1:
                break
            s += prob[v, j] * target[u]
        reach[v] = s
    return reach


# ------------------------- motif-profit env -------------------------------
class MotifProfitEnv:
    """MDP over (graph, motifs, partial seed set). Tracks per-node activation
    probability under the current seed set so both node AND motif state feed the
    features. Node/motif incidence (Bn2m, Bm2n) is exposed for the GNN."""

    NODE_FEAT = 11
    MOTIF_FEAT = 3

    def __init__(self, adj, prob, costs, benefits, motifs, budget, cfg):
        self.adj, self.prob = adj, prob
        self.n = adj.shape[0]
        self.cfg = cfg
        self.threshold = cfg.threshold
        self.budget = float(budget)

        self.cost = np.array([costs.get(i, 1e9) for i in range(self.n)], float)
        self.benefit = np.array([benefits.get(i, 0.0) for i in range(self.n)], float)

        self.motifs = motifs
        self.num_m = len(motifs)
        self.M = (np.array([sorted(m) for m in motifs], dtype=np.int64)
                  if motifs else np.zeros((0, 1), dtype=np.int64))
        self.motif_benefit = np.array(
            [sum(self.benefit[i] for i in m) for m in motifs], float) \
            if motifs else np.zeros(0)

        # node<->motif incidence, row-normalised (mean aggregation)
        Bn2m = np.zeros((self.n, max(self.num_m, 1)), np.float32)
        Bm2n = np.zeros((max(self.num_m, 1), self.n), np.float32)
        mem = np.zeros(self.n)
        for j, m in enumerate(motifs):
            for i in m:
                Bn2m[i, j] = 1.0
                Bm2n[j, i] = 1.0
                mem[i] += 1
        self.motifs_per_node = mem
        self.Bn2m = Bn2m / np.clip(Bn2m.sum(1, keepdims=True), 1, None)
        self.Bm2n = Bm2n / np.clip(Bm2n.sum(1, keepdims=True), 1, None)

        outdeg = np.array([(adj[u] != -1).sum() for u in range(self.n)], float)
        self.outdeg = outdeg
        # probability-weighted out-degree: sum of outgoing edge probabilities.
        # On heterogeneous graphs (trivalency) this separates a node whose edges
        # are strong from one with the same degree but near-dead edges.
        wdeg = np.array([prob[u][adj[u] != -1].sum() for u in range(self.n)], float)
        self.wdeg = wdeg
        self._nz = {
            "ben": self.benefit.max() or 1.0,
            "deg": outdeg.max() or 1.0,
            "wdeg": wdeg.max() or 1.0,
            "mem": mem.max() or 1.0,
            "mben": (self.motif_benefit.max() or 1.0) if self.num_m else 1.0,
        }
        # max attainable motif value (all disjoint motifs covered) -> reward scale
        self.reward_scale = (max(self.motif_benefit.sum(), 1.0)
                             if cfg.scale_reward else 1.0)
        self.reset()

    def reset(self, budget=None):
        if budget is not None:
            self.budget = float(budget)
        self.S, self.S_set = [], set()
        self.remaining = self.budget
        self.cur_phi = 0.0
        self.act_prob = np.zeros(self.n)         # activation freq under current S
        return self.features()

    def affordable_mask(self):
        aff = self.cost <= self.remaining + 1e-9
        for i in self.S_set:
            aff[i] = False
        return aff

    # --- diffuse once and return (motif value, per-node activation prob) ---
    def _simulate(self, seed_list, sims):
        if not seed_list:
            return 0.0, np.zeros(self.n)
        arr = np.array(seed_list, dtype=np.int32)
        tot, actc = 0.0, np.zeros(self.n)
        for _ in range(sims):
            mask = icm_mask(arr, self.adj, self.prob, -1)
            tot += self._phi(mask)
            actc += mask
        return tot / sims, actc / sims

    def _phi(self, mask):
        if self.num_m == 0:
            return 0.0
        counts = mask[self.M].sum(1)
        infl = counts >= self.threshold
        if not infl.any():
            return 0.0
        if self.cfg.objective == "count":
            return float(infl.sum())
        nodes = np.unique(self.M[infl].ravel())   # union of influenced-motif nodes
        return float(self.benefit[nodes].sum())

    def phi_mc(self, seed_list, sims):
        return self._simulate(seed_list, sims)[0]

    # --- state features: node channel + motif channel ---
    def _k_hop_reach(self, target, hops):
        """Accumulated weighted reach into `target` over 1..hops steps:
        A.t + A^2.t + ... + A^hops.t. Each extra hop is naturally discounted by
        the product of edge probabilities, so no separate decay is needed."""
        total = np.zeros(self.n)
        cur = target.astype(np.float64)
        for _ in range(max(1, hops)):
            cur = motif_reach(self.adj, self.prob, cur)   # one more hop: A . cur
            total = total + cur
        return total

    def _motif_state(self):
        if self.num_m == 0:
            return np.zeros(0), np.zeros(0, bool)
        ec = self.act_prob[self.M].sum(1)          # expected activated nodes per motif
        influenced = ec >= self.threshold
        return ec, influenced

    def features(self):
        ec, influenced = self._motif_state()
        if self.num_m:
            uncovered_nodes = self.M[~influenced].ravel()
            uncov = np.bincount(uncovered_nodes, minlength=self.n).astype(float)
            # indicator of nodes that sit in a still-uncovered motif
            target = (uncov > 0).astype(np.float64)
            # weighted reach into uncovered-motif nodes over up to reach_hops steps
            reach = self._k_hop_reach(target, self.cfg.reach_hops)
        else:
            uncov = np.zeros(self.n)
            reach = np.zeros(self.n)

        in_seed = np.zeros(self.n)
        for i in self.S_set:
            in_seed[i] = 1.0
        aff = self.affordable_mask().astype(float)
        rem_frac = np.full(self.n, self.remaining / (self.budget or 1.0))  # horizon

        x_node = np.stack([
            self.benefit / self._nz["ben"],
            np.clip(self.cost, 0, self.budget) / (self.budget or 1.0),
            self.outdeg / self._nz["deg"],
            self.wdeg / self._nz["wdeg"],            # prob-weighted out-degree (influence)
            self.motifs_per_node / self._nz["mem"],
            in_seed,
            aff,
            self.act_prob,                          # current influence signal
            uncov / (uncov.max() or 1.0),           # uncovered motifs v BELONGS to
            reach / (reach.max() or 1.0),           # uncovered motifs v can INFLUENCE
            rem_frac,                               # fraction of budget left (horizon)
        ], axis=1).astype(np.float32)

        if self.num_m:
            x_motif = np.stack([
                self.motif_benefit / self._nz["mben"],
                np.clip(ec / max(self.threshold, 1), 0, 1),   # closeness to threshold
                influenced.astype(float),                     # already covered?
            ], axis=1).astype(np.float32)
        else:
            x_motif = np.zeros((0, self.MOTIF_FEAT), np.float32)
        return x_node, x_motif

    def _simulate_crn(self, seed_list, v, sims):
        """Common-random-numbers marginal of adding v: reuse ONE live-edge graph
        to evaluate both S and S+{v}, so the difference has far lower variance.
        Returns (marginal, phi_new, act_prob_new)."""
        Sarr = np.array(seed_list, dtype=np.int32)
        Sv = np.array(seed_list + [int(v)], dtype=np.int32)
        marg = 0.0
        phi_new = 0.0
        actc = np.zeros(self.n)
        for _ in range(sims):
            live = sample_live(self.adj, self.prob, -1)      # one shared live graph
            a_s = reach_live(Sarr, self.adj, live) if seed_list else np.zeros(self.n, np.bool_)
            a_sv = reach_live(Sv, self.adj, live)
            pv = self._phi(a_sv)
            marg += pv - (self._phi(a_s) if seed_list else 0.0)
            phi_new += pv
            actc += a_sv
        return marg / sims, phi_new / sims, actc / sims

    def step(self, v):
        if self.cfg.crn:
            marg, phi, actp = self._simulate_crn(self.S, v, self.cfg.reward_sims)
            self.S.append(int(v)); self.S_set.add(int(v))
            self.remaining -= self.cost[v]
            reward = (marg - self.cost[v]) / self.reward_scale
        else:
            self.S.append(int(v)); self.S_set.add(int(v))
            self.remaining -= self.cost[v]
            phi, actp = self._simulate(self.S, self.cfg.reward_sims)
            reward = ((phi - self.cur_phi) - self.cost[v]) / self.reward_scale
        self.cur_phi, self.act_prob = phi, actp
        done = not self.affordable_mask().any()
        return self.features(), float(reward), done


# ------------------- motif-aware Structure2Vec Q-net ----------------------
class MotifS2VQNet(nn.Module):
    def __init__(self, node_feat, motif_feat, emb=64, t_layers=3):
        super().__init__()
        self.T, self.emb = t_layers, emb
        self.theta1 = nn.Linear(node_feat, emb)     # node features
        self.theta2 = nn.Linear(emb, emb)           # graph-neighbour messages
        self.thetaC = nn.Linear(emb, emb)           # motif -> node messages
        self.thetaA = nn.Linear(emb, emb)           # node -> motif messages
        self.thetaB = nn.Linear(motif_feat, emb)    # motif features
        self.theta6 = nn.Linear(emb, emb)           # graph pool
        self.theta7 = nn.Linear(emb, emb)           # node
        self.theta8 = nn.Linear(emb, emb)           # motif pool
        self.theta5 = nn.Linear(3 * emb, 1)

    def forward(self, x_node, x_motif, A, Bn2m, Bm2n):
        n = x_node.size(0)
        m = x_motif.size(0)
        mu = torch.zeros(n, self.emb, device=x_node.device)
        nu = torch.zeros(m, self.emb, device=x_node.device)
        for _ in range(self.T):
            if m:
                nu = torch.relu(self.thetaA(Bm2n @ mu) + self.thetaB(x_motif))
            neigh = A @ mu
            motif_msg = (Bn2m @ nu) if m else torch.zeros_like(mu)
            mu = torch.relu(self.theta1(x_node) + self.theta2(neigh)
                            + self.thetaC(motif_msg))
        graph_pool = self.theta6(mu.sum(0, keepdim=True)).expand(n, -1)
        motif_pool = (self.theta8(nu.sum(0, keepdim=True)).expand(n, -1)
                      if m else torch.zeros(n, self.emb, device=x_node.device))
        q = self.theta5(torch.relu(torch.cat(
            [graph_pool, self.theta7(mu), motif_pool], dim=1)))
        return q.squeeze(-1)


# ----------------------------- DQN agent ----------------------------------
class DQNAgent:
    def __init__(self, node_feat, motif_feat, A, Bn2m, Bm2n, cfg):
        self.cfg = cfg
        self.A = torch.tensor(A, dtype=torch.float32)
        self.Bn2m = torch.tensor(Bn2m, dtype=torch.float32)
        self.Bm2n = torch.tensor(Bm2n, dtype=torch.float32)
        self.q = MotifS2VQNet(node_feat, motif_feat, cfg.emb_dim, cfg.t_layers)
        self.tgt = MotifS2VQNet(node_feat, motif_feat, cfg.emb_dim, cfg.t_layers)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = deque(maxlen=cfg.buffer_cap)
        self.updates = 0

    def _q(self, net, feat):
        xn, xm = feat
        return net(torch.tensor(xn), torch.tensor(xm), self.A, self.Bn2m, self.Bm2n)

    def act(self, feat, feasible, eps):
        idx = np.where(feasible)[0]
        if len(idx) == 0:
            return None
        if random.random() < eps:
            return int(random.choice(idx))
        with torch.no_grad():
            q = self._q(self.q, feat).numpy()
        return int(np.where(feasible, q, -1e9).argmax())

    def push(self, feat, a, R, feat_next, feas_next, done):
        self.buffer.append((feat, a, R, feat_next, feas_next, done))

    def train_step(self):
        if len(self.buffer) < self.cfg.batch_size:
            return None
        cfg = self.cfg
        batch = random.sample(self.buffer, cfg.batch_size)
        self.opt.zero_grad()
        qs, ys = [], []
        for feat, a, R, feat_next, feas_next, done in batch:
            qs.append(self._q(self.q, feat)[a])
            if done or feat_next is None or not feas_next.any():
                ys.append(torch.tensor(R, dtype=torch.float32))
            else:
                with torch.no_grad():
                    fmask = torch.tensor(feas_next)
                    if cfg.double_dqn:                      # online selects, target evaluates
                        q_on = self._q(self.q, feat_next).clone()
                        q_on[~fmask] = -1e9
                        a_star = int(q_on.argmax())
                        q_next = self._q(self.tgt, feat_next)[a_star]
                    else:
                        q_next = self._q(self.tgt, feat_next)[fmask].max()
                    ys.append(R + (cfg.gamma ** cfg.n_step) * q_next)
        q = torch.stack(qs)
        y = torch.stack(ys)
        if cfg.huber:
            loss = torch.nn.functional.smooth_l1_loss(q, y)
        else:
            loss = ((q - y) ** 2).mean()
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.opt.step()
        self.updates += 1
        if self.updates % cfg.target_sync == 0:
            self.tgt.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def train(self, env, budget_lo=None, budget_hi=None):
        cfg = self.cfg
        lo = budget_lo if budget_lo is not None else cfg.budget_lo
        hi = budget_hi if budget_hi is not None else cfg.budget_hi
        for ep in range(cfg.episodes):
            # budget-aware training: sample a budget so the policy learns to act
            # well at SHORT horizons too (fixes the low-budget collapse).
            b = random.uniform(lo, hi) if (lo is not None and hi is not None) else None
            env.reset(b)
            eps = max(cfg.eps_end, cfg.eps_start -
                      (cfg.eps_start - cfg.eps_end) * ep / cfg.eps_decay_episodes)
            traj, feat, done = [], env.features(), False
            while not done:
                feas = env.affordable_mask()
                a = self.act(feat, feas, eps)
                if a is None:
                    break
                feat_next, r, done = env.step(a)
                traj.append((feat, a, r, feat_next, env.affordable_mask(), done))
                feat = feat_next
            n = cfg.n_step
            for t in range(len(traj)):
                R, disc, last, boot = 0.0, 1.0, None, True
                for k in range(n):
                    idx = t + k
                    if idx >= len(traj):
                        boot = False
                        break
                    R += disc * traj[idx][2]
                    disc *= cfg.gamma
                    last = idx
                    if traj[idx][5]:
                        boot = False
                        break
                if boot:
                    fn, feasn, dn = traj[last][3], traj[last][4], False
                else:
                    fn, feasn, dn = None, None, True
                self.push(traj[t][0], traj[t][1], R, fn, feasn, dn)
            for _ in range(4):
                loss = self.train_step()
            if cfg.verbose and ep % max(1, cfg.episodes // 10) == 0:
                print(f"  ep {ep:4d} | eps {eps:.2f} | Phi {env.cur_phi:8.2f} "
                      f"| seeds {len(env.S)} | loss {loss}")
        return self

    def rollout(self, env):
        env.reset()
        feat, done = env.features(), False
        while not done:
            a = self.act(feat, env.affordable_mask(), eps=0.0)
            if a is None:
                break
            feat, _, done = env.step(a)
        return list(env.S), float(sum(env.cost[i] for i in env.S))

    def save(self, path, **meta):
        """Persist the trained network (+ optional metadata like train_time, cfg).
        Reload with build_agent(...) then agent.load(path) to infer at any budget."""
        payload = {"q": self.q.state_dict(), "tgt": self.tgt.state_dict()}
        payload.update(meta)
        torch.save(payload, path)

    def load(self, path):
        """Load network weights saved by .save(). Returns the full checkpoint dict
        (so callers can read metadata such as train_time and cfg)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.q.load_state_dict(ckpt["q"])
        self.tgt.load_state_dict(ckpt["tgt"])
        return ckpt

    def rollout_hybrid(self, env, topk=8, sims=64, stop_when_negative=False):
        """RL-guided greedy: Q (hedged with weighted-degree) proposes a shortlist;
        each candidate's TRUE marginal profit is scored by CRN Monte-Carlo and the
        best PER COST is added. Simulates ~2*topk candidates/step instead of all n.

        stop_when_negative is OFF by default: at tau>=2 the motif objective is
        non-submodular (the first seed in a motif has ~0 marginal, the second
        completes it), so a myopic stop would cut off complementary pairs.
        """
        env.reset()
        feat, done = env.features(), False
        while not done:
            feas = env.affordable_mask()
            fidx = np.where(feas)[0]
            if len(fidx) == 0:
                break
            with torch.no_grad():
                q = self._q(self.q, feat).numpy()
            q_rank = fidx[np.argsort(np.where(feas, q, -1e9)[fidx])[::-1][:topk]]
            w_rank = fidx[np.argsort(env.wdeg[fidx])[::-1][:topk]]
            cand = list(dict.fromkeys([int(x) for x in q_rank] +
                                      [int(x) for x in w_rank]))
            best_v, best_profit, best_ratio = None, -np.inf, -np.inf
            for v in cand:
                marg, _, _ = env._simulate_crn(env.S, v, sims)
                mp = marg - env.cost[v]
                ratio = mp / env.cost[v] if env.cost[v] > 0 else mp
                if ratio > best_ratio:
                    best_ratio, best_profit, best_v = ratio, mp, v
            if best_v is None or (stop_when_negative and best_profit <= 0):
                break
            feat, _, done = env.step(best_v)
        return list(env.S), float(sum(env.cost[i] for i in env.S))


# --------------------------- helpers --------------------------------------
def dense_row_normalised(adj, prob):
    n = adj.shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    for u in range(n):
        for j in range(adj.shape[1]):
            v = adj[u, j]
            if v == -1:
                break
            A[u, v] = prob[u, j]
    rs = A.sum(1, keepdims=True)
    rs[rs == 0] = 1.0
    return A / rs


def load_motifs(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(frozenset(ast.literal_eval(line)))
    return out


def build_agent(adj, prob, env, cfg):
    A = dense_row_normalised(adj, prob)
    return DQNAgent(env.NODE_FEAT, env.MOTIF_FEAT, A, env.Bn2m, env.Bm2n, cfg)


def train_and_select(adj, prob, costs, benefits, motifs, budget, cfg):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    env = MotifProfitEnv(adj, prob, costs, benefits, motifs, budget, cfg)
    agent = build_agent(adj, prob, env, cfg)
    agent.train(env)
    seeds, cost = agent.rollout(env)
    profit = env.phi_mc(seeds, cfg.eval_sims) - cost
    return seeds, cost, profit, agent


# ------------------- pipeline_common integration --------------------------
ALGO = "RLMotif"
EXTRA_COLUMNS = ("Motif_Profit_RL",)
MOTIF_FILE = "motifs_size3.txt"
_AGENT_CACHE = {}


def select_seeds(ctx):
    cfg = getattr(ctx.cfg, "rl", None) or RLConfig()
    motifs = load_motifs(MOTIF_FILE)
    env = MotifProfitEnv(ctx.adj, ctx.prob, ctx.costs, ctx.benefits, motifs,
                         ctx.budget, cfg)
    agent = _AGENT_CACHE.get(ctx.model_idx)
    if agent is None:
        agent = build_agent(ctx.adj, ctx.prob, env, cfg)
        agent.train(env)
        _AGENT_CACHE[ctx.model_idx] = agent
    seeds, cost = agent.rollout(env)
    profit = env.phi_mc(seeds, cfg.eval_sims) - cost
    return seeds, cost, {"Motif_Profit_RL": round(profit, 2)}


# ------------------------------ demo --------------------------------------
if __name__ == "__main__":
    import networkx as nx
    rng = random.Random(0)
    edges = set()
    while len(edges) < 30:
        u, v = rng.randint(0, 11), rng.randint(0, 11)
        if u != v:
            edges.add((u, v))
    n = 12
    md = max(1, max(sum(1 for a, b in edges if a == u) for u in range(n)))
    adj = -np.ones((n, md), dtype=np.int32)
    prob = np.zeros((n, md), dtype=np.float32)
    deg = np.zeros(n, dtype=np.int32)
    for u, v in edges:
        adj[u, deg[u]] = v
        prob[u, deg[u]] = 0.35
        deg[u] += 1
    costs = {i: 1.0 for i in range(n)}
    benefits = {i: float(rng.randint(1, 10)) for i in range(n)}
    motifs = [frozenset([0, 1, 2]), frozenset([3, 4, 5]),
              frozenset([6, 7, 8]), frozenset([9, 10, 11])]
    cfg = RLConfig(threshold=2, objective="profit", episodes=80,
                   reward_sims=10, eval_sims=64, emb_dim=32, verbose=True)
    t0 = time.time()
    seeds, cost, profit, _ = train_and_select(adj, prob, costs, benefits,
                                              motifs, budget=4, cfg=cfg)
    print(f"\nmotif-aware RL seed set: {seeds}  cost={cost}  "
          f"motif profit={profit:.2f}  ({time.time()-t0:.1f}s)")
