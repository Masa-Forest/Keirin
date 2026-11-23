"""
KEIRIN.JP（競輪オフィシャルサイト）から 1日分の出走表＋結果を読み、schema.py の形にそろえる

サイトの画面は https://keirin.jp/pc/json?type=JSJxxx&... の JSON を読んで描いているので、同じ JSON を読む。
ログインは要らない。

    JSJ057 kday=YYYYMMDD    その日の開催場の一覧（場コード・グレード・開催の鍵 encPrm）
    JSJ001 encp=開催の鍵     開催の日程（何日目か）と、各レースの鍵 encParaR
    JSJ017 encp=開催の鍵     出走表一覧（種別・発走時刻・並び予想）※並び予想はその日のレースが終わると消える
    JSJ002 encp=レースの鍵   その日その場の全レースの出走表（距離・周回・競走得点・脚質・決まり手回数・S/B 回数・勝率）
    JSJ011 encp=レースの鍵   ギヤ倍数（1レースずつ）
    JSJ012 encp=レースの鍵   競走結果（着順・上がり・決まり手・H/B・天候・風速・払戻）

注意（2026-11-23 時点）:
    - robots.txt はこれらのページを機械的に読むことを認めていない。サイトポリシーは「私的使用のための複製」は
      認め、第三者から金銭を得ることを禁じている。個人の研究用にだけ使い、間隔をあけて読むこと（MIN_INTERVAL）。
    - 一度読んだ過去日の JSON は cache_dir に保存し、二度は読まない。
    - 並び予想は当日のうちしか取れない。過去日を取り直しても line_id/line_pos は空欄になる。
      並びを貯めるには、毎日レース前（または当日中）に fetch を回す（merge は空欄で上書きしない）。
"""

import datetime as dt
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

BASE = 'https://keirin.jp/pc/json'
MIN_INTERVAL = 2.0      # 秒。リクエストの間隔（これより短くしないこと）

# バンク周長（m）。場コードは KEIRIN.JP の場コード（jyolist の jocd）
BANK = {
    11: 400, 12: 400, 13: 400, 21: 400, 22: 335, 23: 400, 24: 500, 25: 500, 26: 400, 27: 400, 28: 400,
    31: 333, 32: 250, 34: 400, 35: 400, 36: 333, 37: 333, 38: 400, 42: 400, 43: 400, 44: 400, 45: 400,
    46: 333, 47: 400, 48: 400, 51: 400, 53: 333, 54: 400, 55: 400, 56: 400, 61: 400, 62: 400, 63: 333,
    71: 400, 73: 400, 74: 500, 75: 400, 81: 400, 83: 400, 84: 400, 85: 400, 86: 400, 87: 400,
}

KIMARITE = {'逃げ': '逃げ', '捲り': '捲り', '差し': '差し', 'マーク': 'マーク'}
PAY = {'exacta_pay': 'ST2', 'quinella_pay': 'SH2', 'trifecta_pay': 'RT3', 'trio_pay': 'RH3'}


def _z(s):
    """全角→半角・空白除去（'熊　本' → '熊本', 'Ｓ級一般' → 'S級一般'）"""
    if s is None:
        return None
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(s)))


def _num(s):
    if s is None:
        return None
    s = str(s).replace(',', '').replace('円', '').strip()
    try:
        return float(s)
    except ValueError:
        return None


def race_class(syumoku):
    s = _z(syumoku) or ''
    if 'ガールズ' in s or s.startswith('L級'):
        return 'L'
    if 'チャレンジ' in s:
        return 'A-chal'
    if s.startswith('S級') or s.startswith('SS'):
        return 'S'
    if s.startswith('A級'):
        return 'A'
    return None


def stage(syumoku):
    """'S級一般' → '一般'、'A級チ予選' → '予選'、'L級ガ予2' → '予2'（級とチャレンジ/ガールズの印を取る）

    表記はサイトのまま（一予2・二予A・特予選・初特選・DMD など）。features.stage_code で段階に直す。
    """
    s = _z(syumoku) or ''
    s = re.sub(r'^(SS|S級|A級|L級)', '', s)
    return re.sub(r'^[チガ](?=.)', '', s) or None


def narabi_lines(n_info):
    """並び予想（JSJ017 の nInfo: 車番と画面上の位置 X, Y）→ {車番: (ライン番号, ライン内の位置)}

    X は左から並べた位置で、ラインの切れ目は X が1つ空く。Y が 2 以上は競り（同じ位置を争う）。
    """
    if not n_info:
        return {}
    cols = {}
    for x in n_info:
        cols.setdefault(int(x['narabiX']), []).append(int(x['syaban']))
    out, line, pos, prev = {}, 0, 0, None
    for xpos in sorted(cols):
        if prev is None or xpos > prev + 1:
            line, pos = line + 1, 0
        pos += 1
        for car in cols[xpos]:
            out[car] = (line, pos)
        prev = xpos
    return out


class Client:
    """間隔をあけて JSON を読む。過去日の分はファイルに保存して使い回す"""

    def __init__(self, cache_dir='data/raw/keirinjp', interval=MIN_INTERVAL):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.interval = max(interval, MIN_INTERVAL)
        self.last = 0.0
        self.n_requests = 0
        self.s = requests.Session()
        self.s.headers.update({'User-Agent': 'Mozilla/5.0 (personal research)',
                               'Referer': 'https://keirin.jp/pc/top'})

    def get(self, typ, cache_key=None, **params):
        path = None
        if self.cache_dir and cache_key:
            path = self.cache_dir / f'{cache_key}_{typ}.json'
            if path.exists():
                return json.loads(path.read_text(encoding='utf-8'))
        wait = self.last + self.interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                r = self.s.get(BASE, params={'type': typ, **params}, timeout=20)
                r.raise_for_status()
                data = r.json()
                break
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(self.interval * (attempt + 2))
        self.last = time.monotonic()
        self.n_requests += 1
        if path is not None and data.get('resultCd', 0) == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        return data


class KeirinJP:
    """1日分の出走表＋結果（fetch_data.Source と同じ形: fetch_day(day) → DataFrame）"""
    name = 'keirinjp'

    def __init__(self, cache_dir='data/raw/keirinjp', gear=True, venues=None, log=print):
        self.c = Client(cache_dir)
        self.gear = gear                 # False ならギヤ（1レース1回の読み込み）を飛ばす
        self.venues = set(venues) if venues else None
        self.log = log

    def fetch_day(self, day: dt.date) -> pd.DataFrame:
        kday = day.strftime('%Y%m%d')
        past = day < dt.date.today()     # 過去日だけ保存する（当日以降は中身が変わる）
        ck = (lambda s: f'{kday}/{s}') if past else (lambda s: None)
        top = self.c.get('JSJ057', ck('list'), kday=kday)
        frames = []
        for v in top.get('kInfo') or []:
            bkcd = int(v['KeirinCd'])
            if self.venues and bkcd not in self.venues:
                continue
            try:
                d = self.fetch_meet(day, v, ck)
            except Exception as e:     # 1場の失敗で1日全部を落とさない
                self.log(f'{kday} {v.get("jyoName")}: 取得失敗 ({e})')
                continue
            if d is not None and len(d):
                frames.append(d)
                self.log(f'{kday} {_z(v["jyoName"])}: {d["race"].nunique()} R, {len(d)} 行')
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fetch_meet(self, day, v, ck):
        bkcd = int(v['KeirinCd'])
        enc = v['encPrm']
        info = self.c.get('JSJ001', ck(f'{bkcd}'), encp=enc).get('C0201data') or {}
        races = info.get('C0201race') or []
        if not races:
            return None
        days = info.get('C0201kaisai') or []
        md = f'{day.month:02d}/{day.day:02d}'
        day_number = next((i + 1 for i, x in enumerate(days) if x.get('txtEventDate') == md), None)
        if info.get('selKaisai') and info['selKaisai'] != day.strftime('%Y%m%d'):
            return None                  # 前日発売などで別の日を指しているとき
        lst = self.c.get('JSJ017', ck(f'{bkcd}'), encp=enc)
        by_race = {int(r['raceNo']): r for r in lst.get('rInfo') or []}

        card = self.c.get('JSJ002', ck(f'{bkcd}'), encp=races[0]['encParaR'])
        rows = []
        for ri in card.get('raceInfo') or []:
            rno = int(ri['raceNo'])
            summary = by_race.get(rno, {})
            syumoku = ri.get('shumokuName') or summary.get('syumoku')
            lines = narabi_lines(summary.get('nInfo')) if summary.get('narabiFlg') else {}
            race = {
                'date': day.isoformat(), 'venue': bkcd, 'venue_name': _z(v.get('jyoName')), 'race': rno,
                'start_time': ri.get('hassouYotei') or summary.get('stTime'),
                'grade': _z(v.get('gradeIconChar')), 'race_class': race_class(syumoku), 'stage': stage(syumoku),
                'day_number': day_number, 'bank': BANK.get(bkcd), 'distance': _num(ri.get('kyori')),
                'race_name': info.get('raceName'), 'laps': _num(ri.get('shukai')),
            }
            for s in ri.get('sensyuTypeInfo') or []:
                car = int(s['syaban'])
                ln = lines.get(car, (None, None))
                rows.append({**race,
                             'car': car, 'waku': _num(s.get('wakuban')) or car, 'reg_no': int(s['sensyuRegistNo']),
                             'name': _z(s.get('sensyuName')), 'pref': _z(s.get('huKen')),
                             'term': _num(s.get('sotugyouki')), 'rank': _z(s.get('kyuhan')),
                             'age': _num(s.get('age')), 'style': _z(s.get('kyakusitu')),
                             'score': _num(s.get('heikinTokuten')), 'win_rate': _num(s.get('syouritu')),
                             'top2_rate': _num(s.get('rentairitu2')), 'top3_rate': _num(s.get('rentairitu3')),
                             'nige': _num(s.get('nigeCnt')), 'makuri': _num(s.get('makuriCnt')),
                             'sashi': _num(s.get('sasiCnt')), 'mark': _num(s.get('markCnt')),
                             's_count': _num(s.get('stTori')), 'b_count': _num(s.get('backCnt')),
                             'h_count': _num(s.get('homeTori')),
                             'line_id': ln[0], 'line_pos': ln[1],
                             'note': _z(s.get('ketujyouTuikaHojyu')) or None})
        if not rows:
            return None
        d = pd.DataFrame(rows)

        for i, r in enumerate(races):
            rno = i + 1
            key = r['encParaR']
            if self.gear:
                g = self.c.get('JSJ011', ck(f'{bkcd}_{rno}'), encp=key)
                for s in g.get('sensyuTypeInfo') or []:
                    m = (d['race'] == rno) & (d['car'] == int(s['syaban']))
                    sub = s.get('commentOrderCntSubData') or {}
                    d.loc[m, 'gear'] = _num(sub.get('gearAfter'))
            if r.get('flgRaceEnd'):
                self.apply_result(d, rno, self.c.get('JSJ012', ck(f'{bkcd}_{rno}'), encp=key))
        return d

    @staticmethod
    def apply_result(d, rno, res):
        items = res.get('tyakujyunItemSubData') or []
        m_race = d['race'] == rno
        if not items:
            pay = json.dumps(res.get('haraiGakuSubData') or {}, ensure_ascii=False)
            if '返還' in pay:
                d.loc[m_race, 'result_note'] = '中止'
            return
        d.loc[m_race, 'weather'] = _z(res.get('tenki'))
        d.loc[m_race, 'wind_speed'] = _num(res.get('husoku'))
        pay = res.get('haraiGakuSubData') or {}
        for col, k in PAY.items():
            lst = pay.get(f'{k}HaraiGakuDispItemSubData') or []
            d.loc[m_race, col] = _num(lst[0].get('haraiGaku')) if lst else None
        # s_flag（スタート取り）は結果に出ないので空欄のまま
        for x in items:
            m = m_race & (d['car'] == int(x['syaban']))
            place = _num(x.get('tyaku'))
            bh = _z(x.get('BH')) or ''
            d.loc[m, 'place'] = place
            d.loc[m, 'agari'] = _num(x.get('agari'))
            d.loc[m, 'kimarite'] = KIMARITE.get(_z(x.get('kimarite'))) if place == 1 else None
            d.loc[m, 'b_flag'] = int('B' in bh)
            d.loc[m, 'h_flag'] = int('H' in bh)
            states = [_z(k.get('kojinState')) for k in x.get('kojinStateItemSubData') or [] if k.get('kojinState')]
            d.loc[m, 'result_note'] = ' '.join(states) or (None if place else _z(x.get('tyaku')))
