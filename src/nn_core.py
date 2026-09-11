"""NN core shared by training (nn.py) and inference (script.py)."""
import numpy as np
import pandas as pd

CAT_SPECS = {'pitcher_id': 64, 'batter_id': 32, 'pitcher_team': 4, 'batter_team': 4, 'home_team': 4, 'count_state': 4, 'base_state': 3}


def build_net(n_num, cat_sizes, hidden=(512, 256, 128), drop=0.2):
    import torch, torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(n + 1, d) for n, d in cat_sizes])
            d_in = n_num + sum(d for _, d in cat_sizes)
            layers = []
            for h in hidden:
                layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop)]; d_in = h
            self.mlp = nn.Sequential(*layers)
            self.head4 = nn.Linear(d_in, 4); self.head_bs = nn.Linear(d_in, 3); self.head_pt = nn.Linear(d_in, 3)

        def forward(self, xn, xc):
            e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
            h = self.mlp(torch.cat([xn] + e, 1))
            return self.head4(h), self.head_bs(h), self.head_pt(h)
    return Net()


def transform_numeric(F, num_feats, qt, nan_cols):
    X = F[num_feats].astype(np.float32).values
    nanmask = np.isnan(X)
    Xq = qt.transform(np.nan_to_num(X, nan=0.0)).astype(np.float32)
    Xq[nanmask] = 0.0
    return np.concatenate([Xq, nanmask[:, nan_cols].astype(np.float32)], 1)


def transform_cats(F, cats, maps):
    codes = np.zeros((len(F), len(cats)), dtype=np.int64)
    for i, c in enumerate(cats):
        codes[:, i] = pd.Series(F[c].values).map(maps[c]).fillna(0).astype(int).values
    return codes


def nn_predict(models, Xn, Xc, bs=32768, init4=None):
    import torch
    out = np.zeros(len(Xn))
    Xn_t = torch.tensor(Xn); Xc_t = torch.tensor(Xc); I4 = None if init4 is None else torch.tensor(np.asarray(init4, dtype=np.float32))
    for net in models:
        net.eval(); ps = []
        with torch.no_grad():
            for i in range(0, len(Xn), bs):
                o4, _, _ = net(Xn_t[i:i + bs], Xc_t[i:i + bs])
                if I4 is not None: o4 = o4 + I4[i:i + bs]
                ps.append(torch.softmax(o4, 1)[:, 0].numpy())
        out += np.concatenate(ps)
    return out / len(models)
