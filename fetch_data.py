"""
データ集め: 集めたデータを schema.py の形式にそろえて 1つの CSV にまとめる

使い方:
    python fetch_data.py merge data/raw/*.csv --out data/races.csv      # CSV を検証してまとめる（重複は新しい方）
    python fetch_data.py check data/races.csv                            # 中身の確認（期間・レース数・欠けている列）
    python fetch_data.py fetch 20260101 20261122 --out data/races.csv    # サイトから取得（取得元は未実装）

取得元（fetch）について:
    BoatRace の BOATRACE Open API のような、競輪の公開 JSON は見つかっていない。
    出走表（競走得点・脚質・ギヤ・並び予想）と結果（着順・上がり・決まり手・払戻）が取れるサイトの
    ページを読む Source を SOURCES に登録すれば、fetch で動くようにしてある。
    各サイトの利用規約を確認し、アクセスの間隔をあけること（REQUEST_INTERVAL 秒）。
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from schema import ALL_COLS, KEY, validate

REQUEST_INTERVAL = 2.0


class Source:
    """取得元の型。1日分の (出走表＋結果) を schema.py の形式の DataFrame で返す"""
    name = 'base'

    def fetch_day(self, day: dt.date) -> pd.DataFrame:
        raise NotImplementedError


SOURCES = {}        # 名前 → Source のサブクラス


def merge(paths, out):
    frames = []
    for p in paths:
        d = pd.read_csv(p, low_memory=False)
        frames.append(validate(d).assign(_order=len(frames)))
        print(f'{p}: {len(d)} 行')
    d = pd.concat(frames, ignore_index=True).sort_values('_order')
    d = d.drop_duplicates(KEY + ['car'], keep='last').drop(columns='_order')
    d = validate(d)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f'{out}: {d.groupby(KEY).ngroups} レース, {len(d)} 行')


def check(path):
    d = validate(pd.read_csv(path, low_memory=False))
    print(f'期間 {d["date"].min()}〜{d["date"].max()}, {d.groupby(KEY).ngroups} レース, {len(d)} 行')
    fill = d[list(ALL_COLS)].notna().mean().sort_values()
    print('列ごとの埋まり具合（1.0 = 全行あり）:')
    print(fill.to_string(float_format=lambda x: f'{x:.2f}'))
    n = d.groupby(KEY)['car'].count()
    print('出走人数:', n.value_counts().sort_index().to_dict())
    has_line = d.groupby(KEY)['line_id'].count().gt(0).mean()
    print(f'並び予想のあるレース: {has_line:.1%}')


def fetch(start, end, out, source):
    if source not in SOURCES:
        sys.exit('取得元がまだありません。fetch_data.py の SOURCES に Source を登録してください '
                 f'（登録済み: {list(SOURCES) or "なし"}）。手元の CSV があれば merge を使ってください。')
    import time
    src = SOURCES[source]()
    frames = []
    day = dt.datetime.strptime(start, '%Y%m%d').date()
    last = dt.datetime.strptime(end, '%Y%m%d').date()
    while day <= last:
        try:
            frames.append(src.fetch_day(day))
        except Exception as e:
            print(f'{day}: 取得失敗 ({e})', file=sys.stderr)
        time.sleep(REQUEST_INTERVAL)
        day += dt.timedelta(days=1)
    tmp = Path(out).with_suffix('.new.csv')
    pd.concat(frames, ignore_index=True).to_csv(tmp, index=False)
    merge([p for p in [out] if Path(p).exists()] + [str(tmp)], out)
    tmp.unlink()


def main():
    ap = argparse.ArgumentParser(description='データ集め')
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('merge')
    m.add_argument('paths', nargs='+')
    m.add_argument('--out', default='data/races.csv')
    c = sub.add_parser('check')
    c.add_argument('path')
    f = sub.add_parser('fetch')
    f.add_argument('start')
    f.add_argument('end')
    f.add_argument('--out', default='data/races.csv')
    f.add_argument('--source', default=next(iter(SOURCES), ''))
    args = ap.parse_args()
    if args.cmd == 'merge':
        merge(args.paths, args.out)
    elif args.cmd == 'check':
        check(args.path)
    else:
        fetch(args.start, args.end, args.out, args.source)


if __name__ == '__main__':
    main()
