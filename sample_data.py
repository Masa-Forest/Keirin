"""
動作確認用の合成データを作る（本物の競輪データではない）

使い方:
    python sample_data.py --out data/sample.csv --start 2025-06-01 --end 2026-11-20

schema.py の形式どおりの CSV を出す。本物のデータが手に入るまで、コードが最後まで動くかを
確かめるためだけに使う。ここでの的中率・回収率には意味がない（作り方の仮定を当てているだけ）。

作り方の仮定: 選手ごとの隠れた強さ＋調子、ラインの先頭の強さが番手に効く、同じレースに
逃げ選手が多いと先頭どうしがつぶし合う、333 バンクは先行有利・500 バンクは追込有利。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from schema import validate

REGIONS = {
    '北日本': ['青森', '岩手', '宮城', '福島', '北海道', '山形'],
    '関東': ['茨城', '栃木', '群馬', '埼玉', '千葉', '東京', '神奈川'],
    '南関東': ['静岡', '山梨'],
    '中部': ['愛知', '岐阜', '三重', '富山', '石川', '福井', '長野'],
    '近畿': ['大阪', '京都', '奈良', '和歌山', '兵庫', '滋賀'],
    '中四国': ['岡山', '広島', '山口', '香川', '徳島', '愛媛', '高知', '鳥取', '島根'],
    '九州': ['福岡', '佐賀', '長崎', '熊本', '大分', '宮崎', '鹿児島', '沖縄'],
}
PREF_REGION = {p: r for r, ps in REGIONS.items() for p in ps}


def make_riders(rng, n=2400):
    prefs = list(PREF_REGION)
    strength = rng.normal(0, 1, n)
    style = rng.choice(['逃', '両', '追'], n, p=[0.35, 0.25, 0.40])
    age = np.clip(rng.normal(34, 8, n), 19, 60).round()
    strength -= np.clip(age - 38, 0, None) * 0.03
    q = np.quantile(strength, [0.03, 0.15, 0.40, 0.65, 0.85])
    rank = np.select([strength > q[4], strength > q[3], strength > q[2], strength > q[1], strength > q[0]],
                     ['SS', 'S1', 'S2', 'A1', 'A2'], 'A3')
    return pd.DataFrame({
        'reg_no': 10000 + np.arange(n), 'name': [f'選手{i:04d}' for i in range(n)],
        'pref': rng.choice(prefs, n), 'term': (130 - (age - 20) * 1.1).round().clip(60, 128),
        'rank': rank, 'age': age, 'style': style, 'strength': strength,
        'form': np.zeros(n), 'gear': rng.choice([3.92, 3.93, 4.00, 4.08], n, p=[0.5, 0.2, 0.2, 0.1]),
    })


def waku_of(car, n):
    """車番 → 枠番（9車: 1,2,3,4,4,5,5,6,6 / 7車: 1,2,3,4,5,6,6）"""
    table = {9: [1, 2, 3, 4, 4, 5, 5, 6, 6], 7: [1, 2, 3, 4, 5, 6, 6]}
    return table.get(n, list(range(1, n + 1)))[car - 1]


def make_lines(rng, field):
    """同じ地区でラインを組み、逃げ（なければ両）の選手を先頭にする"""
    field = field.assign(region=field['pref'].map(PREF_REGION))
    lines = []
    for _, g in field.groupby('region', sort=False):
        idx = list(g.sample(frac=1, random_state=int(rng.integers(1e9))).index)
        while idx:
            k = min(len(idx), rng.choice([1, 2, 3], p=[0.15, 0.45, 0.40]))
            lines.append(idx[:k])
            idx = idx[k:]
    order = {'逃': 0, '両': 1, '追': 2}
    lines = [sorted(l, key=lambda i: (order[field.at[i, 'style']], -field.at[i, 'strength'])) for l in lines]
    rng.shuffle(lines)
    line_id, line_pos = {}, {}
    for li, l in enumerate(lines, 1):
        for pos, i in enumerate(l, 1):
            line_id[i], line_pos[i] = li, pos
    return pd.Series(line_id), pd.Series(line_pos)


def simulate_race(rng, field, bank):
    s = field['strength'] + field['form']
    pos, lid = field['line_pos'], field['line_id']
    leader_s = pd.Series({i: s[(lid == lid[i]) & (pos == 1)].iloc[0] for i in field.index})
    n_nige = ((field['style'] == '逃') & (pos == 1)).sum()
    perf = s.copy()
    perf += np.where(pos == 2, 0.45 * leader_s, 0) + np.where(pos == 3, 0.2 * leader_s, 0)
    perf -= np.where((pos == 1) & (field['style'] == '逃'), 0.15 * max(n_nige - 2, 0), 0)
    bank_adj = {333: 0.25, 400: 0.0, 500: -0.25}.get(bank, 0)
    perf += np.where(pos == 1, bank_adj, -bank_adj / 2)
    perf += rng.gumbel(0, 1.1, len(field))
    place = (-perf).rank(method='first')
    fall = rng.random(len(field)) < 0.012
    place = place.where(~fall).rank(method='first')
    return place, perf


def main():
    ap = argparse.ArgumentParser(description='動作確認用の合成データ（本物ではない）')
    ap.add_argument('--out', default='data/sample.csv')
    ap.add_argument('--start', default='2025-06-01')
    ap.add_argument('--end', default='2026-11-20')
    ap.add_argument('--venues-per-day', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    riders = make_riders(rng).set_index('reg_no', drop=False)
    venues = pd.DataFrame({'venue': range(11, 54), 'venue_name': [f'場{i}' for i in range(11, 54)],
                           'bank': rng.choice([333, 400, 500], 43, p=[0.2, 0.7, 0.1])})
    rows = []
    for day in pd.date_range(args.start, args.end):
        riders['form'] = riders['form'] * 0.98 + rng.normal(0, 0.05, len(riders))
        for _, v in venues.sample(args.venues_per_day, random_state=int(rng.integers(1e9))).iterrows():
            grade = rng.choice(['G1', 'G3', 'F1', 'F2'], p=[0.05, 0.15, 0.45, 0.35])
            day_number = int(rng.integers(1, 5 if grade in ('G1', 'G3') else 4))
            for race in range(1, 12):
                cls = 'S' if (grade in ('G1', 'G3') or race >= 7) else 'A'
                pool = riders[riders['rank'].isin(['SS', 'S1', 'S2'] if cls == 'S' else ['A1', 'A2', 'A3'])]
                n = 9 if (cls == 'S' or rng.random() < 0.5) else 7
                field = pool.sample(n, weights=np.exp(pool['strength'] * (race / 11)),
                                    random_state=int(rng.integers(1e9))).reset_index(drop=True)
                field['line_id'], field['line_pos'] = make_lines(rng, field)
                place, perf = simulate_race(rng, field, v['bank'])
                cars = rng.permutation(np.arange(1, n + 1))
                s = field['strength'] + field['form']
                noisy = lambda x, sd: x + rng.normal(0, sd, n)
                stage = '決勝' if (race == 11 and day_number >= 3) else ('準決勝' if race >= 9 and day_number == 2
                                                                     else ('特選' if race >= 10 else '予選'))
                winner = place == 1
                pos = field['line_pos']
                kim = np.where(~winner, '', np.where(pos == 1, np.where(field['style'] == '逃', '逃げ', '捲り'),
                                                     np.where(pos == 2, 'マーク', '差し')))
                for i in range(n):
                    rows.append(dict(
                        date=day.strftime('%Y-%m-%d'), venue=v['venue'], venue_name=v['venue_name'], race=race,
                        start_time=f'{10 + race // 2:02d}:{(race % 2) * 30:02d}', grade=grade, race_class=cls,
                        stage=stage, day_number=day_number, bank=v['bank'], distance=2025 if v['bank'] != 500 else 2000,
                        weather=rng.choice(['晴', '曇', '雨'], p=[0.6, 0.3, 0.1]),
                        wind_speed=round(float(rng.gamma(2, 1.2)), 1),
                        car=int(cars[i]), waku=waku_of(int(cars[i]), n),
                        reg_no=int(field.at[i, 'reg_no']), name=field.at[i, 'name'], pref=field.at[i, 'pref'],
                        term=int(field.at[i, 'term']), rank=field.at[i, 'rank'], age=int(field.at[i, 'age']),
                        style=field.at[i, 'style'], gear=field.at[i, 'gear'],
                        score=round(float(noisy(95 + 6 * s, 1.5)[i] - (8 if cls == 'A' else 0)), 2),
                        win_rate=round(float(np.clip(noisy(15 + 12 * s, 5)[i], 0, 80)), 1),
                        top2_rate=round(float(np.clip(noisy(30 + 15 * s, 6)[i], 0, 95)), 1),
                        top3_rate=round(float(np.clip(noisy(45 + 15 * s, 7)[i], 0, 100)), 1),
                        nige=int(rng.poisson(3 if field.at[i, 'style'] == '逃' else 0.3)),
                        makuri=int(rng.poisson(1.5 if field.at[i, 'style'] != '追' else 0.3)),
                        sashi=int(rng.poisson(2 if field.at[i, 'style'] != '逃' else 0.3)),
                        mark=int(rng.poisson(1.5 if field.at[i, 'style'] == '追' else 0.2)),
                        s_count=int(rng.poisson(1)), b_count=int(rng.poisson(4 if field.at[i, 'style'] == '逃' else 1)),
                        line_id=int(field.at[i, 'line_id']), line_pos=int(field.at[i, 'line_pos']),
                        place=place[i], kimarite=kim[i],
                        agari=round(float(11.6 - 0.25 * perf[i] / 2 + rng.normal(0, 0.2)), 1),
                        s_flag=0, b_flag=int(pos[i] == 1 and rng.random() < 0.3),
                    ))
    d = pd.DataFrame(rows)
    # 払戻: 競走得点から作った「人気」で決める（本物の払戻ではない）
    k = ['date', 'venue', 'race']
    pop = np.exp((d['score'] - d.groupby(k)['score'].transform('mean')) / 4)
    pop = pop / pop.groupby([d[c] for c in k]).transform('sum')
    by = d.assign(pop=pop).sort_values(k + ['place'])
    top = by[by['place'] <= 3].groupby(k)['pop'].apply(list)
    pays = {}
    for key, ps in top.items():
        if len(ps) < 3:
            continue
        p1, p2, p3 = ps
        ex = p1 * p2 / (1 - p1)
        tri = ex * p3 / (1 - p1 - p2)
        pays[key] = dict(exacta_pay=round(75 / ex, -1), quinella_pay=round(75 / (2 * ex), -1),
                         trifecta_pay=round(75 / tri, -1), trio_pay=round(75 / (6 * tri), -1))
    pay = pd.DataFrame.from_dict(pays, orient='index')
    pay.index = pd.MultiIndex.from_tuples(pay.index, names=k)
    d = d.join(pay, on=k)
    d = validate(d)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    print(f'{args.out}: {d.groupby(k).ngroups} レース, {len(d)} 行（合成データ）')


if __name__ == '__main__':
    main()
