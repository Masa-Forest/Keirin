"""
予想（predict.py の *_tickets.csv）と結果を照合する

使い方:
    python check_results.py predictions/20261123_tickets.csv data/races.csv [--top 5]

買い目ごとに、確率上位 top 点を100円ずつ買ったときの的中レース数・回収率を出す。
"""

import argparse

import pandas as pd

from model import PAY_COL
from schema import KEY, validate


def actual_tickets(res):
    out = {}
    for key, r in res.groupby(KEY):
        r = r.dropna(subset=['place']).sort_values('place')
        if len(r) < 3 or list(r['place'].iloc[:3]) != [1, 2, 3]:
            continue
        c = r['car'].astype(int).tolist()
        out[key] = {'2車単': f'{c[0]}-{c[1]}', '3連単': f'{c[0]}-{c[1]}-{c[2]}',
                    '3連複': '='.join(map(str, sorted(c[:3])))}
        out[key]['pay'] = {b: r[col].iloc[0] for b, col in PAY_COL.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description='予想と結果の照合')
    ap.add_argument('tickets_csv')
    ap.add_argument('result_csv')
    ap.add_argument('--top', type=int, default=5)
    args = ap.parse_args()

    t = pd.read_csv(args.tickets_csv)
    res = validate(pd.read_csv(args.result_csv, low_memory=False))
    res = res[res['date'].isin(t['date'].unique())]
    act = actual_tickets(res)
    t = t[t['rank'] <= args.top]
    rows = []
    for bet, tb in t.groupby('bet'):
        races = hit = 0
        pay = 0.0
        for key, r in tb.groupby(KEY):
            if key not in act:
                continue
            races += 1
            if act[key][bet] in set(r['ticket']):
                hit += 1
                pay += act[key]['pay'][bet] if pd.notna(act[key]['pay'][bet]) else 0
        if races:
            rows.append(dict(買い目=bet, レース数=races, 的中=hit, 的中率=hit / races,
                             回収率=pay / (100 * args.top * races)))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
