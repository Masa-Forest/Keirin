"""
データ集め: 集めたデータを schema.py の形式にそろえて 1つの CSV にまとめる

使い方:
    python fetch_data.py merge data/raw/*.csv --out data/races.csv      # CSV を検証してまとめる（重複は新しい方）
    python fetch_data.py check data/races.csv                            # 中身の確認（期間・レース数・欠けている列）
    python fetch_data.py fetch 20261101 20261122 --out data/races.csv    # KEIRIN.JP から取得して races.csv に足す
    python fetch_data.py card 20261124 --out cards/20261124.csv          # 当日・翌日の出走表（並び予想つき）だけ

取得元（fetch / card）について:
    keirinjp.py（KEIRIN.JP の画面が読んでいる JSON）を使う。個人の研究用に限り、2秒以上の間隔で読む。
    過去日の JSON は data/raw/keirinjp/ に保存し、取り直さない。
    並び予想は当日のうちしか取れないので、並びを貯めるなら毎日 fetch（または card）を回す。
    merge は同じ選手の行が重なったとき「新しい方の値、空欄なら古い方の値」を残すので、
    当日朝に取った並び予想が夜の取り直しで消えることはない。
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from schema import ALL_COLS, KEY, validate

import keirinjp

REQUEST_INTERVAL = 0     # 日ごとの追加の待ち時間（リクエストごとの間隔は keirinjp.MIN_INTERVAL）


class Source:
    """取得元の型。1日分の (出走表＋結果) を schema.py の形式の DataFrame で返す"""
    name = 'base'

    def fetch_day(self, day: dt.date) -> pd.DataFrame:
        raise NotImplementedError


SOURCES = {'keirinjp': keirinjp.KeirinJP}        # 名前 → Source のクラス


def merge(paths, out):
    frames = []
    for p in paths:
        d = pd.read_csv(p, low_memory=False)
        frames.append(validate(d).assign(_order=len(frames)))
        print(f'{p}: {len(d)} 行')
    d = pd.concat(frames, ignore_index=True).sort_values('_order', kind='stable')
    # 同じ選手の行が重なったら、列ごとに新しい方の値（空欄なら古い方）を残す
    d = d.groupby(KEY + ['car'], as_index=False, sort=False).last().drop(columns='_order')
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


def _source(source, **kw):
    if source not in SOURCES:
        sys.exit(f'取得元 {source!r} がありません（登録済み: {list(SOURCES)}）')
    return SOURCES[source](**kw)


def _days(start, end):
    day = dt.datetime.strptime(start, '%Y%m%d').date()
    last = dt.datetime.strptime(end or start, '%Y%m%d').date()
    while day <= last:
        yield day
        day += dt.timedelta(days=1)


def fetch(start, end, out, source, gear=True, venues=None):
    """start〜end を1日ずつ読み、out に足していく（1日ごとに保存するので、途中で止めても続きから回せる）"""
    import time
    src = _source(source, gear=gear, venues=venues)
    for day in _days(start, end):
        try:
            d = src.fetch_day(day)
        except Exception as e:
            print(f'{day}: 取得失敗 ({e})', file=sys.stderr)
            continue
        if d is None or d.empty:
            print(f'{day}: 開催なし')
            continue
        tmp = Path(out).with_suffix('.new.csv')
        tmp.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(tmp, index=False)
        merge([p for p in [out] if Path(p).exists()] + [str(tmp)], out)
        tmp.unlink()
        time.sleep(REQUEST_INTERVAL)
    print(f'読み込み {getattr(getattr(src, "c", None), "n_requests", "?")} 回')


def card(day, out, source, gear=True, venues=None):
    """1日分の出走表だけを out に書く（predict.py にそのまま渡せる。結果の列は空欄）"""
    src = _source(source, gear=gear, venues=venues)
    d = src.fetch_day(dt.datetime.strptime(day, '%Y%m%d').date())
    if d is None or d.empty:
        sys.exit(f'{day}: 開催なし')
    d = validate(d)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    has_line = d.groupby(KEY)['line_id'].count().gt(0).mean()
    print(f'{out}: {d.groupby(KEY).ngroups} レース, {len(d)} 行, 並び予想あり {has_line:.0%}')


def main():
    ap = argparse.ArgumentParser(description='データ集め')
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('merge')
    m.add_argument('paths', nargs='+')
    m.add_argument('--out', default='data/races.csv')
    c = sub.add_parser('check')
    c.add_argument('path')
    f = sub.add_parser('fetch', help='期間を取得して --out に足す')
    f.add_argument('start', help='YYYYMMDD')
    f.add_argument('end', nargs='?', help='YYYYMMDD（省略で start の1日だけ）')
    f.add_argument('--out', default='data/races.csv')
    k = sub.add_parser('card', help='1日分の出走表だけを書く')
    k.add_argument('day', help='YYYYMMDD')
    k.add_argument('--out', required=True)
    for p in (f, k):
        p.add_argument('--source', default='keirinjp')
        p.add_argument('--no-gear', action='store_true', help='ギヤを読まない（読み込み回数がほぼ半分になる）')
        p.add_argument('--venue', type=int, action='append', help='場コードで絞る（何度でも指定可）')
    args = ap.parse_args()
    if args.cmd == 'merge':
        merge(args.paths, args.out)
    elif args.cmd == 'check':
        check(args.path)
    elif args.cmd == 'fetch':
        fetch(args.start, args.end, args.out, args.source, not args.no_gear, args.venue)
    else:
        card(args.day, args.out, args.source, not args.no_gear, args.venue)


if __name__ == '__main__':
    main()
