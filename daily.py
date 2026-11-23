"""
毎日の取得: 今日の出走表（並び予想つき）と、昨日の結果を data/daily/YYYYMMDD.csv に保存する

使い方:
    python daily.py                      # 日本時間の今日・昨日
    python daily.py --today 20261124     # 日付を指定

並び予想はレース当日しか取れないので、朝（全レースの前）に回す。
昨日の分は朝に取った並び予想を残したまま、結果の列だけを埋める（fetch_data.merge は空欄で上書きしない）。
1日ごとのファイルにしているのは、毎日コミットしても差分が小さくて済むように。全部をまとめるときは:
    python fetch_data.py merge data/daily/*.csv --out data/races.csv
"""

import argparse
import datetime as dt
from pathlib import Path

import fetch_data

JST = dt.timezone(dt.timedelta(hours=9))


def main():
    ap = argparse.ArgumentParser(description='毎日の取得')
    ap.add_argument('--today', help='YYYYMMDD（省略で日本時間の今日）')
    ap.add_argument('--dir', default='data/daily')
    args = ap.parse_args()
    today = (dt.datetime.strptime(args.today, '%Y%m%d').date() if args.today
             else dt.datetime.now(JST).date())
    yesterday = today - dt.timedelta(days=1)
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    # 昨日: 結果を埋める（ギヤは前日の朝に取ってあるので読まない）
    y = yesterday.strftime('%Y%m%d')
    fetch_data.fetch(y, y, f'{args.dir}/{y}.csv', 'keirinjp', gear=not Path(f'{args.dir}/{y}.csv').exists())
    # 今日: 出走表と並び予想
    t = today.strftime('%Y%m%d')
    fetch_data.fetch(t, t, f'{args.dir}/{t}.csv', 'keirinjp', gear=True)


if __name__ == '__main__':
    main()
