"""
当日の出走表から予想する

使い方:
    python predict.py 出走表.csv --history data/races.csv [--model models/win.pkl] [--out predictions/20261123.csv]

出走表.csv は schema.py の形式（結果の列は空欄でよい）。並び予想は line_id/line_pos の列か、
narabi 列（"17/243/586/9" のような文字列、レース内のどの行でもよい）で渡す。
過去の成績の特徴量を作るため、--history に過去データを渡す（出走表の日より前の分だけ使う）。

出力: 選手ごとの 1着・2着内・3着内の確率と、レースごとの 2車単・3連単・3連複の買い目候補（確率順）。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from model import WinModel, harville, race_tickets
from schema import KEY, parse_line, validate


def apply_narabi(card):
    if 'narabi' not in card.columns:
        return card
    card = card.copy()
    for key, r in card.groupby(KEY):
        s = r['narabi'].dropna()
        if s.empty:
            continue
        lines = parse_line(s.iloc[0])
        for i, car in zip(r.index, r['car']):
            if int(car) in lines:
                card.loc[i, ['line_id', 'line_pos']] = lines[int(car)]
    return card


def marginals(p):
    """Harville 式から、各選手の 2着内・3着内の確率"""
    n = len(p)
    perms, prob = harville(p, 3)
    top2, top3 = np.zeros(n), np.zeros(n)
    for j in range(3):
        np.add.at(top3, perms[:, j], prob)
        if j < 2:
            np.add.at(top2, perms[:, j], prob)
    return top2, top3


def predict(card, history_csv, model_path):
    m = WinModel.load(model_path)
    card = validate(apply_narabi(card))
    for c in ['place', 'agari', 'kimarite', 'exacta_pay', 'quinella_pay', 'trifecta_pay', 'trio_pay']:
        card[c] = np.nan                                      # 当日の結果は使わない
    first = card['date'].min()
    hist = validate(pd.read_csv(history_csv, low_memory=False))
    hist = hist[hist['date'] < first]
    both = pd.concat([hist.assign(_card=False), card.assign(_card=True)], ignore_index=True)
    d, _ = F.build(both, getattr(m, 'groups', None))
    d = d[d['_card']].copy()
    d['p_win'] = m.predict(d)
    rows, tickets = [], []
    for key, r in d.sort_values(KEY + ['car']).groupby(KEY, sort=False):
        p = r['p_win'].to_numpy() / r['p_win'].sum()
        t2, t3 = marginals(p)
        rows.append(r.assign(p_win=p, p_top2=t2, p_top3=t3))
        for bet, (names, probs) in race_tickets(r['car'].to_numpy(), p).items():
            for rank, (nm, pr) in enumerate(zip(names, probs), 1):
                tickets.append(dict(zip(KEY, key), bet=bet, rank=rank, ticket=nm, prob=pr))
    return pd.concat(rows), pd.DataFrame(tickets)


def main():
    ap = argparse.ArgumentParser(description='出走表から予想')
    ap.add_argument('card_csv')
    ap.add_argument('--history', required=True)
    ap.add_argument('--model', default='models/win.pkl')
    ap.add_argument('--out', default=None, help='選手ごとの確率の保存先（買い目は *_tickets.csv）')
    ap.add_argument('--show', type=int, default=5, help='画面に出す買い目の点数')
    args = ap.parse_args()

    riders, tickets = predict(pd.read_csv(args.card_csv), args.history, args.model)
    for key, r in riders.groupby(KEY, sort=False):
        name = r['venue_name'].iloc[0] if pd.notna(r['venue_name'].iloc[0]) else f'場{key[1]}'
        print(f'\n■ {key[0]} {name} {key[2]}R  {r["stage"].iloc[0] if pd.notna(r["stage"].iloc[0]) else ""}')
        show = r.sort_values('p_win', ascending=False)
        for _, x in show.iterrows():
            ln = f'L{int(x["line_id"])}-{int(x["line_pos"])}' if pd.notna(x['line_id']) else ''
            print(f'  {int(x["car"])}番 {str(x["name"]):10s} {str(x["style"]):2s} 得点{x["score"]:6.2f} {ln:6s}'
                  f'  1着 {x["p_win"]:.1%}  2着内 {x["p_top2"]:.1%}  3着内 {x["p_top3"]:.1%}')
        t = tickets[(tickets['date'] == key[0]) & (tickets['venue'] == key[1]) & (tickets['race'] == key[2])]
        for bet, tb in t.groupby('bet', sort=False):
            tb = tb.head(args.show)
            print(f'  {bet}: ' + '  '.join(f'{a}({b:.1%})' for a, b in zip(tb['ticket'], tb['prob'])))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cols = KEY + ['venue_name', 'car', 'name', 'style', 'score', 'line_id', 'line_pos', 'p_win', 'p_top2',
                      'p_top3']
        riders[cols].to_csv(args.out, index=False)
        tickets.to_csv(str(Path(args.out).with_suffix('')) + '_tickets.csv', index=False)
        print(f'\n{args.out} と買い目を保存しました')


if __name__ == '__main__':
    main()
