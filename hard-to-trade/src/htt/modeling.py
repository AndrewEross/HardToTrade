# Baselines, reward-matrix machinery, sklearn logistic regression, and a small torch net.
import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config as C

CLASSES = ["easy", "medium", "hard"]

# Reward matrix from the proposal, rows = PREDICTED, columns = TRUE (easy, medium, hard).
REWARD = np.array([[1, -1, -3],
                   [0, 1, -1],
                   [-1, 0, 1]], dtype=float)


def reward_decision(proba_true: np.ndarray, R: np.ndarray = REWARD) -> np.ndarray:
    # Given P(true class | x), pick the prediction with the highest expected reward.
    return np.argmax(proba_true @ R.T, axis=1)


# ------------------------------------------------------------ baselines
class HistoricalBaseline:
    # Predict the class a bucket key had most often in TRAIN (proposal: key = (ticker, slot)).
    # use_reward=True picks the reward-maximising class under the empirical distribution instead.
    def __init__(self, keys=("symbol", "slot"), use_reward=False):
        self.keys, self.use_reward = list(keys), use_reward

    def fit(self, meta: pd.DataFrame, y):
        d = meta[self.keys].copy()
        d["y"] = np.asarray(y)
        cols = [0, 1, 2]
        self.counts = d.groupby(self.keys + ["y"]).size().unstack(fill_value=0).reindex(columns=cols, fill_value=0)
        self.tick = d.groupby("symbol")["y"].value_counts().unstack(fill_value=0).reindex(columns=cols, fill_value=0)
        self.glob = np.bincount(d["y"], minlength=3).astype(float)
        return self

    def predict_proba(self, meta: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(meta[self.keys]) if len(self.keys) > 1 else pd.Index(meta[self.keys[0]])
        c = self.counts.reindex(idx).to_numpy(float)
        miss = np.isnan(c).any(axis=1)
        if miss.any():
            c[miss] = self.tick.reindex(meta["symbol"].to_numpy()[miss]).to_numpy(float)
        c = np.where(np.isnan(c), self.glob, c)
        return c / c.sum(axis=1, keepdims=True)

    def predict(self, meta):
        p = self.predict_proba(meta)
        return reward_decision(p) if self.use_reward else p.argmax(axis=1)


# ------------------------------------------------------------ logistic regression
class RewardDecisionLR:
    # L2 multinomial logistic regression; C tuned on validation reward; decisions via expected reward.
    def __init__(self, Cs=(0.003, 0.01, 0.03, 0.1, 0.3, 1, 3), use_reward_decision=True, seed=C.SEED):
        self.Cs, self.use_reward_decision, self.seed = Cs, use_reward_decision, seed

    def fit(self, Xtr, ytr, Xva, yva):
        best = (-np.inf, None, None)
        for c in self.Cs:
            m = LogisticRegression(C=c, max_iter=3000, random_state=self.seed).fit(Xtr, ytr)
            pred = self._decide(m.predict_proba(Xva))
            r = REWARD[pred, yva].mean()
            if r > best[0]:
                best = (r, c, m)
        self.val_reward, self.best_C, self.model = best
        return self

    def _decide(self, p):
        return reward_decision(p) if self.use_reward_decision else p.argmax(axis=1)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self._decide(self.predict_proba(X))


# ------------------------------------------------------------ torch models
def _build_net(d_in, hidden, dropout):
    import torch.nn as nn
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, 3))
    return nn.Sequential(*layers)


class TorchClassifier:
    # hidden=()        -> single-layer softmax model (paper's "single-layer regression")
    # hidden=(16, 8)   -> the proposal's feedforward net
    # loss="reward"    -> softmax outputs are a distribution over PREDICTED labels and the loss is the
    #                     negative expected reward (Hard to Park, Sec. 6.2).
    # loss="ce"        -> ordinary cross-entropy; decisions then use reward_decision on the outputs.
    def __init__(self, hidden=(16, 8), dropout=0.2, weight_decay=1e-3, lr=5e-3, epochs=300,
                 patience=25, batch_size=512, loss="reward", seed=C.SEED):
        self.hidden, self.dropout, self.weight_decay, self.lr = hidden, dropout, weight_decay, lr
        self.epochs, self.patience, self.batch_size, self.loss, self.seed = epochs, patience, batch_size, loss, seed

    def _loss(self, logits, y):
        import torch
        import torch.nn.functional as F
        if self.loss == "ce":
            return F.cross_entropy(logits, y)
        p = torch.softmax(logits, dim=1)
        R = torch.tensor(REWARD, dtype=torch.float32)  # [pred, true]
        return -(p * R[:, y].T).sum(dim=1).mean()

    def fit(self, Xtr, ytr, Xva, yva):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Xtr, Xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
        ytr, yva = torch.tensor(np.asarray(ytr), dtype=torch.long), torch.tensor(np.asarray(yva), dtype=torch.long)
        self.net = _build_net(Xtr.shape[1], self.hidden, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, bad, self.history = np.inf, None, 0, []
        for ep in range(self.epochs):
            self.net.train()
            perm = rng.permutation(len(Xtr))
            for i in range(0, len(perm), self.batch_size):
                b = torch.tensor(perm[i:i + self.batch_size])
                opt.zero_grad()
                self._loss(self.net(Xtr[b]), ytr[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                tr_l = self._loss(self.net(Xtr), ytr).item()
                va_l = self._loss(self.net(Xva), yva).item()
            self.history.append((tr_l, va_l))
            if va_l < best - 1e-5:
                best, best_state, bad = va_l, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()

    def predict(self, X):
        p = self.predict_proba(X)
        return p.argmax(axis=1) if self.loss == "reward" else reward_decision(p)
