"""
データの形式（1行=1選手）

レースは (date, venue, race) で一意に決まる。過去データ（結果あり）も当日の出走表（結果なし）も
同じ形式にそろえておけば、特徴量づくり・学習・予想がすべて同じコードで動く。

必須でない列は空欄でよい（特徴量の側で欠損として扱う）。
"""

import pandas as pd

KEY = ['date', 'venue', 'race']

# ---- レースの情報（同じレースの全選手で同じ値）
RACE_COLS = {
    'date': '開催日 YYYY-MM-DD',
    'venue': '場コード（数値）',
    'venue_name': '場名',
    'race': 'レース番号',
    'start_time': '発走時刻 HH:MM（なければ空欄）',
    'grade': 'グレード GP/G1/G2/G3/F1/F2',
    'race_class': '級 S/A/A-chal（チャレンジ）/L（ガールズ）',
    'stage': '種別（一次予選・二次予選・準決勝・決勝・特選・選抜・一般 など、出走表の表記そのまま）',
    'day_number': '開催何日目か',
    'bank': 'バンク周長 m（333/400/500 など）',
    'distance': '距離 m',
    'weather': '天候（晴/曇/雨/雪）',
    'wind_speed': '風速 m/s',
}

# ---- 選手の情報（出走表・レース前に分かるもの）
RIDER_COLS = {
    'car': '車番 1〜9',
    'waku': '枠番',
    'reg_no': '選手登録番号',
    'name': '選手名',
    'pref': '府県',
    'term': '期別',
    'rank': '級班 SS/S1/S2/A1/A2/A3/L1',
    'age': '年齢',
    'style': '脚質 逃/両/追',
    'gear': 'ギヤ倍率',
    'score': '競走得点',
    'win_rate': '勝率 %',
    'top2_rate': '2連対率 %',
    'top3_rate': '3連対率 %',
    'nige': '決まり手 逃げ 回数（直近4か月）',
    'makuri': '決まり手 捲り 回数',
    'sashi': '決まり手 差し 回数',
    'mark': '決まり手 マーク 回数',
    's_count': 'S（スタート取り）回数',
    'b_count': 'B（バック先頭）回数',
    'line_id': '並び予想のライン番号（レース内で 1,2,3…、単騎も1ライン）',
    'line_pos': 'ライン内の位置 1=先頭 2=番手 3=3番手 …',
}

# ---- 結果（過去データのみ。当日の出走表では空欄）
RESULT_COLS = {
    'place': '着順（失格・落車・棄権・欠場は空欄）',
    'kimarite': 'この選手の決まり手（1着のみ: 逃げ/捲り/差し/マーク）',
    'agari': '上がりタイム 秒（最終半周）',
    's_flag': 'このレースでSを取ったか 0/1',
    'b_flag': 'このレースでBを取ったか 0/1',
    'exacta_pay': '2車単の払戻（100円あたり、レース共通）',
    'quinella_pay': '2車複の払戻',
    'trifecta_pay': '3連単の払戻',
    'trio_pay': '3連複の払戻',
}

ALL_COLS = {**RACE_COLS, **RIDER_COLS, **RESULT_COLS}
REQUIRED = KEY + ['car', 'reg_no']

RANK_CODES = {'SS': 1, 'S1': 2, 'S2': 3, 'A1': 4, 'A2': 5, 'A3': 6, 'L1': 7}
GRADE_CODES = {'GP': 1, 'G1': 2, 'G2': 3, 'G3': 4, 'F1': 5, 'F2': 6}
CLASS_CODES = {'S': 1, 'A': 2, 'A-chal': 3, 'L': 4}
STYLE_CODES = {'逃': 1, '両': 2, '追': 3}
WEATHER_CODES = {'晴': 1, '曇': 2, '雨': 3, '雪': 4}


def validate(df):
    """必須列の有無と、1レース内で車番が重複していないかを確認して、型をそろえて返す"""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'必須の列がありません: {missing}')
    d = df.copy()
    for c in ALL_COLS:
        if c not in d.columns:
            d[c] = pd.NA
    d['date'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d')
    num = ['venue', 'race', 'day_number', 'bank', 'distance', 'wind_speed', 'car', 'waku', 'reg_no', 'term',
           'age', 'gear', 'score', 'win_rate', 'top2_rate', 'top3_rate', 'nige', 'makuri', 'sashi', 'mark',
           's_count', 'b_count', 'line_id', 'line_pos', 'place', 'agari', 's_flag', 'b_flag',
           'exacta_pay', 'quinella_pay', 'trifecta_pay', 'trio_pay']
    for c in num:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    dup = d.duplicated(KEY + ['car'])
    if dup.any():
        raise ValueError(f'同じレースで車番が重複しています: {d.loc[dup, KEY + ["car"]].head().to_dict("records")}')
    return d.sort_values(KEY + ['car']).reset_index(drop=True)


def parse_line(narabi):
    """並び予想の文字列をライン番号・位置に変換する

    例: "17/243/586/9" または "1-7 2-4-3 5-8-6 9" → {1: (1, 1), 7: (1, 2), 2: (2, 1), ...}
    （車番 → (ライン番号, ライン内の位置)）
    """
    s = str(narabi).strip().replace('／', '/').replace('　', ' ')
    if '/' in s:
        groups = [g.replace('-', '').replace(' ', '') for g in s.split('/')]
        groups = [[int(ch) for ch in g] for g in groups if g]
    else:
        groups = [[int(x) for x in g.split('-') if x] for g in s.split()]
    out = {}
    for li, g in enumerate(groups, 1):
        for pos, car in enumerate(g, 1):
            out[car] = (li, pos)
    return out
