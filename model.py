"""
モデルと評価

1着確率: レース内ソフトマックス（その場の 7〜9 人の中で誰が勝つか）を LightGBM で直接学習する。
         出走人数がレースごとに違ってもよい（BoatRace の all_venue_v2.SoftmaxGBM を可変人数にしたもの）。
2着・3着: Harville 式（1着確率を残りの選手で正規化して使い回す）。
評価: 時系列で分けたテスト期間で、対数損失・1着的中・2車単/3連単/3連複の上位 k 点的中、
      払戻があれば上位 k 点を均等買いしたときの回収率。
"""

import itertools
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from schema import KEY


def _segments(race_id):
    """連続して並んだ race_id の各レースの開始位置と人数"""
    starts = np.flatnonzero(np.r_[True, race_id[1:] != race_id[:-1]])
    return starts, np.diff(np.r_[starts, len(race_id)])


def group_softmax(score, starts):
    m = np.maximum.reduceat(score, starts)
    e = np.exp(score - np.repeat(m, np.diff(np.r_[starts, len(score)])))
    s = np.add.reduceat(e, starts)
    return e / np.repeat(s, np.diff(np.r_[starts, len(score)]))


class WinModel:
    """1着確率モデル。fit/predict に渡す d はレースキー・車番順に並べて使う"""

    def __init__(self, features, rounds=500, lr=0.03, leaves=31, min_leaf=100, ff=0.7, l2=1.0, seed=0,
                 categorical=('venue',)):
        self.features = list(features)
        self.rounds = rounds
        self.categorical = [c for c in categorical if c in self.features]
        self.params = dict(learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_leaf, feature_fraction=ff,
                           bagging_fraction=0.8, bagging_freq=1, lambda_l2=l2, verbose=-1, seed=seed,
                           num_threads=4)

    @staticmethod
    def _sorted(d):
        return d.sort_values(KEY + ['car'], kind='stable')

    def fit(self, d):
        d = self._sorted(trainable(d))
        rid = d.groupby(KEY, sort=False).ngroup().to_numpy()
        starts, _ = _segments(rid)
        y = (d['place'] == 1).to_numpy(float)

        def objective(preds, data):
            p = group_softmax(preds, starts)
            return p - y, np.maximum(p * (1 - p), 1e-6)

        params = dict(self.params, objective=objective)
        ds = lgb.Dataset(d[self.features], y, categorical_feature=self.categorical or 'auto', free_raw_data=False)
        self.model = lgb.train(params, ds, self.rounds)
        return self

    def predict(self, d):
        """d と同じ並びで 1着確率を返す"""
        ds = self._sorted(d)
        rid = ds.groupby(KEY, sort=False).ngroup().to_numpy()
        starts, _ = _segments(rid)
        p = group_softmax(self.model.predict(ds[self.features], raw_score=True), starts)
        return pd.Series(p, index=ds.index).reindex(d.index).to_numpy()

    def importance(self):
        return pd.Series(self.model.feature_importance('gain'), index=self.features).sort_values(ascending=False)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def trainable(d):
    """1着が1人だけ決まっているレースだけ残す（不成立・同着は学習に使わない）"""
    n_win = (d['place'] == 1).groupby([d[c] for c in KEY]).transform('sum')
    return d[n_win == 1]


# ---------------------------------------------------------------- 組み合わせの確率

def harville(p, k):
    """p: 1レースの1着確率（長さ n）。k 着までの順列と確率を、確率の高い順に返す"""
    n = len(p)
    perms = np.array(list(itertools.permutations(range(n), k)))
    prob = np.ones(len(perms))
    left = np.ones(len(perms))
    for j in range(k):
        pj = p[perms[:, j]]
        prob *= pj / np.maximum(left, 1e-12)
        left = left - pj
    o = np.argsort(-prob)
    return perms[o], prob[o]


def trio(p):
    perms, prob = harville(p, 3)
    key = np.sort(perms, 1)
    df = pd.DataFrame({'a': key[:, 0], 'b': key[:, 1], 'c': key[:, 2], 'p': prob})
    s = df.groupby(['a', 'b', 'c'])['p'].sum().sort_values(ascending=False)
    return np.array(s.index.tolist()), s.to_numpy()


def race_tickets(cars, p, top=(1, 3, 5, 10)):
    """1レースの買い目候補: 2車単・3連単・3連複を確率順に"""
    kmax = max(top)
    ex, pe = harville(p, 2)
    tr, pt = harville(p, 3)
    to, po = trio(p)
    fmt = lambda arr: ['-'.join(str(int(cars[i])) for i in row) for row in arr[:kmax]]
    fmt_o = lambda arr: ['='.join(str(int(c)) for c in sorted(cars[list(row)])) for row in arr[:kmax]]
    return {'2車単': (fmt(ex), pe[:kmax]), '3連単': (fmt(tr), pt[:kmax]), '3連複': (fmt_o(to), po[:kmax])}


# ---------------------------------------------------------------- 評価

PAY_COL = {'2車単': 'exacta_pay', '3連単': 'trifecta_pay', '3連複': 'trio_pay'}


def evaluate(d, p, label, top=(1, 3, 5, 10)):
    """d: テスト期間（結果あり）, p: d と同じ並びの1着確率"""
    d = d.assign(p=p)
    rows = dict(モデル=label)
    ll, hit1, races = [], [], 0
    hits = {(b, k): 0 for b in PAY_COL for k in top}
    pay = {(b, k): 0.0 for b in PAY_COL for k in top}
    n_pay = {b: 0 for b in PAY_COL}
    for _, r in trainable(d).sort_values(KEY + ['car']).groupby(KEY, sort=False):
        cars, pr, place = r['car'].to_numpy(), r['p'].to_numpy(), r['place'].to_numpy()
        pr = pr / pr.sum()
        races += 1
        w = np.flatnonzero(place == 1)[0]
        ll.append(-np.log(max(pr[w], 1e-12)))
        hit1.append(int(np.argmax(pr) == w))
        order = [np.flatnonzero(place == k) for k in (1, 2, 3)]
        if any(len(o) != 1 for o in order):
            continue
        actual = {'2車単': f'{cars[order[0][0]]}-{cars[order[1][0]]}',
                  '3連単': f'{cars[order[0][0]]}-{cars[order[1][0]]}-{cars[order[2][0]]}',
                  '3連複': '='.join(str(c) for c in sorted(cars[[o[0] for o in order]]))}
        tickets = race_tickets(cars, pr, top)
        for b, (names, _) in tickets.items():
            payout = r[PAY_COL[b]].iloc[0]
            has_pay = pd.notna(payout)
            n_pay[b] += has_pay
            for k in top:
                if actual[b] in names[:k]:
                    hits[(b, k)] += 1
                    if has_pay:
                        pay[(b, k)] += payout
    rows['レース数'] = races
    rows['対数損失'] = float(np.mean(ll))
    rows['1着的中'] = float(np.mean(hit1))
    for b in PAY_COL:
        for k in top:
            if b == '2車単' or k > 1:
                rows[f'{b}{k}点的中'] = hits[(b, k)] / races
    for b in PAY_COL:
        if n_pay[b]:
            for k in top:
                if k > 1 or b == '2車単':
                    rows[f'{b}{k}点回収'] = pay[(b, k)] / (100 * k * n_pay[b])
    return rows


def baseline_score_rank(d):
    """比較用: 競走得点だけで決める（得点のソフトマックス、温度は固定）"""
    z = d['score'].fillna(d['score'].mean()) / 3.0
    e = np.exp(z - z.groupby([d[c] for c in KEY]).transform('max'))
    return (e / e.groupby([d[c] for c in KEY]).transform('sum')).to_numpy()
