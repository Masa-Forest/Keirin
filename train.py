"""
予想に使うモデルを学習して保存する

使い方:
    python train.py data/races.csv                         # 全グループ・全期間で学習 → models/win.pkl
    python train.py data/races.csv --groups line history   # 基準＋指定グループだけ
    python train.py data/races.csv --until 2026-11-22      # この日までのデータで学習

research.py で良かった組み合わせを --groups に指定する。
"""

import argparse
from pathlib import Path

import features as F
from model import WinModel
from research import load


def main():
    ap = argparse.ArgumentParser(description='モデルを学習して保存')
    ap.add_argument('csv')
    ap.add_argument('--groups', nargs='*', default=None)
    ap.add_argument('--until', default=None, help='この日まで（含む）で学習')
    ap.add_argument('--since', default=None)
    ap.add_argument('--rounds', type=int, default=500)
    ap.add_argument('--out', default='models/win.pkl')
    args = ap.parse_args()

    d, by_group = load(args.csv, args.groups)
    if args.until:
        d = d[d['date'] <= args.until]
    if args.since:
        d = d[d['date'] >= args.since]
    groups = list(by_group)
    m = WinModel(F.columns(by_group, groups), rounds=args.rounds).fit(d)
    m.groups = groups
    Path(args.out).parent.mkdir(exist_ok=True)
    m.save(args.out)
    print(f'{args.out}: {d["date"].min()}〜{d["date"].max()} で学習, グループ {groups}, 特徴量 {len(m.features)} 個')


if __name__ == '__main__':
    main()
