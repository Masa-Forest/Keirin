"""
特徴量の研究: グループを足したり外したりして、テスト期間の成績を比べる

使い方:
    python research.py data/races.csv --test-from 2026-09-01                # 基準＋各グループ、全部
    python research.py data/races.csv --test-from 2026-09-01 --drop         # 全部から1グループずつ外す
    python research.py data/races.csv --groups line history                 # 指定したグループだけ
    python research.py data/races.csv --extra line:leader_score,front_score  # 列単位で試す（グループ:列）

学習は test-from より前、評価は test-from 以降（時系列で分けるので未来の情報で学習しない）。
結果は画面に出し、research/log.csv に1行ずつ追記する（いつ・どのデータで・どの特徴量で・何点だったか）。
比較用に「競走得点だけ」の成績も出す。
"""

import argparse
import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd

import features as F
from model import WinModel, baseline_score_rank, evaluate
from schema import validate

LOG = Path('research/log.csv')
SHOW = ['レース数', '対数損失', '1着的中', '2車単1点的中', '2車単5点的中', '3連単10点的中', '3連複5点的中',
        '2車単5点回収', '3連単10点回収', '3連複5点回収']


def load(path, groups=None):
    d = validate(pd.read_csv(path, low_memory=False))
    return F.build(d, groups)


def split(d, test_from, train_from=None):
    tr = d[d['date'] < test_from]
    if train_from:
        tr = tr[tr['date'] >= train_from]
    return tr, d[d['date'] >= test_from]


def run(sets, train, test, params, data_name, test_from, log=True):
    rows, models = [], {}
    rows.append(evaluate(test, baseline_score_rank(test), '比較: 競走得点だけ'))
    for name, cols in sets.items():
        m = WinModel(cols, **params).fit(train)
        r = evaluate(test, m.predict(test), name)
        r['特徴量数'] = len(cols)
        rows.append(r)
        models[name] = m
        print(f'  {name}: 対数損失 {r["対数損失"]:.4f}  1着的中 {r["1着的中"]:.3f}', flush=True)
        if log:
            write_log(r, cols, data_name, test_from, len(train))
    return pd.DataFrame(rows).set_index('モデル'), models


def write_log(r, cols, data_name, test_from, n_train):
    LOG.parent.mkdir(exist_ok=True)
    row = {'日時': dt.datetime.now().strftime('%Y-%m-%d %H:%M'), 'データ': data_name, 'テスト開始': test_from,
           '学習行数': n_train, **r, '特徴量ID': hashlib.md5(','.join(sorted(cols)).encode()).hexdigest()[:8],
           '特徴量': ','.join(cols)}
    df = pd.DataFrame([row])
    if LOG.exists():
        old = pd.read_csv(LOG)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(LOG, index=False)


def main():
    ap = argparse.ArgumentParser(description='特徴量グループの比較')
    ap.add_argument('csv')
    ap.add_argument('--test-from', default='2026-09-01')
    ap.add_argument('--train-from', default=None)
    ap.add_argument('--groups', nargs='*', default=None, help='比べるグループ（省略時は全部）')
    ap.add_argument('--drop', action='store_true', help='全部から1グループずつ外して比べる')
    ap.add_argument('--extra', nargs='*', default=[], help='グループ:列1,列2 の形で、基準に足す列を指定')
    ap.add_argument('--rounds', type=int, default=500)
    ap.add_argument('--no-log', action='store_true')
    args = ap.parse_args()

    print('グループ:')
    for name, (_, doc) in F.GROUPS.items():
        print(f'  {name:10s} {doc}')
    d, by_group = load(args.csv)
    train, test = split(d, args.test_from, args.train_from)
    print(f'学習 {train["date"].min()}〜{train["date"].max()} / 評価 {test["date"].min()}〜{test["date"].max()}')

    others = [g for g in (args.groups or list(F.GROUPS)) if g not in F.BASE_GROUPS]
    base = F.columns(by_group, F.BASE_GROUPS)
    every = F.columns(by_group, F.BASE_GROUPS + others)
    sets = {}
    if args.extra:
        sets['基準'] = base
        for spec in args.extra:
            g, cols = spec.split(':')
            sets[f'基準+{spec}'] = base + [c for c in cols.split(',') if c in by_group[g]]
    elif args.drop:
        sets['全部'] = every
        sets.update({f'全部-{g}': [c for c in every if c not in by_group[g]] for g in others})
    else:
        sets['基準'] = base
        sets.update({f'基準+{g}': base + by_group[g] for g in others})
        if len(others) > 1:
            sets['全部'] = every

    out, models = run(sets, train, test, dict(rounds=args.rounds), Path(args.csv).name, args.test_from,
                      log=not args.no_log)
    ref = '全部' if args.drop else '基準'
    show = [c for c in SHOW if c in out.columns]     # 全部の列は research/log.csv に残る
    print('\n' + out[show].T.to_string(float_format=lambda x: f'{x:.4f}'))
    diff = out.loc[[i for i in out.index if i in sets and i != ref], ['対数損失', '1着的中']] - \
        out.loc[ref, ['対数損失', '1着的中']]
    print(f'\n{ref} との差（対数損失は小さいほど良い）:\n' + diff.to_string(float_format=lambda x: f'{x:+.4f}'))
    last = models[list(sets)[-1]]
    print(f'\n「{list(sets)[-1]}」の重要度（gain）上位20:\n' + last.importance().head(20).to_string(
        float_format=lambda x: f'{x:.0f}'))
    if not args.no_log:
        print(f'\n{LOG} に追記しました')


if __name__ == '__main__':
    main()
