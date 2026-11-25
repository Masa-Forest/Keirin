"""
特徴量（グループ単位で登録する）

新しい特徴量の足し方:
    1. このファイルに関数を書き、@feature_group('名前') を付ける
    2. 関数は 1行=1選手の DataFrame d を受け取り、新しい列だけの DataFrame を返す
       （d には schema.py の列と、先に登録されたグループの列が入っている）
    3. python research.py data/races.csv --groups 名前   で基準との差を測る

注意: 過去の成績から作る特徴量は、そのレースより前のレースの結果だけを使うこと
（_prior_rolling を使えば自動でそうなる）。当日の結果（place, agari, kimarite …）を
そのまま特徴量に入れると未来の情報が漏れる。
"""

import numpy as np
import pandas as pd

from schema import CLASS_CODES, GRADE_CODES, KEY, RANK_CODES, STYLE_CODES, WEATHER_CODES

GROUPS = {}          # 名前 → (関数, 説明)
BASE_GROUPS = ['base']


def feature_group(name):
    def deco(fn):
        GROUPS[name] = (fn, (fn.__doc__ or '').strip().splitlines()[0] if fn.__doc__ else '')
        return fn
    return deco


def _g(d, col):
    return d.groupby(KEY, sort=False)[col]


def _prior_rolling(d, by, col, window, how='mean', min_periods=3):
    """by ごとに、時刻順で直前 window 走の集計（その行自身は含めない）。d は時刻順に並んでいること"""
    return d.groupby(by, sort=False)[col].transform(
        lambda s: getattr(s.shift().rolling(window, min_periods=min_periods), how)())


# ---------------------------------------------------------------- 基準

@feature_group('base')
def base(d):
    """出走表の値そのもの（競走得点・勝率・脚質・級班・ギヤ・バンク）とレース内の比較"""
    out = pd.DataFrame(index=d.index)
    out['car'] = d['car']
    out['n_riders'] = _g(d, 'car').transform('count')
    out['bank'] = d['bank']
    out['grade_code'] = d['grade'].map(GRADE_CODES)
    out['class_code'] = d['race_class'].map(CLASS_CODES)
    out['rank_code'] = d['rank'].map(RANK_CODES)
    out['style_code'] = d['style'].map(STYLE_CODES)
    for c in ['age', 'term', 'gear', 'score', 'win_rate', 'top2_rate', 'top3_rate', 's_count', 'b_count']:
        out[c] = d[c]
    out['score_dev'] = d['score'] - _g(d, 'score').transform('mean')
    out['score_rank'] = _g(d, 'score').rank(ascending=False)
    out['score_gap_top'] = _g(d, 'score').transform('max') - d['score']
    k = d[['nige', 'makuri', 'sashi', 'mark']].fillna(0)
    tot = k.sum(1).replace(0, np.nan)
    for c in k:
        out[f'{c}_share'] = k[c] / tot
    out['kimarite_total'] = k.sum(1)
    return out


# ---------------------------------------------------------------- ライン（並び予想）

@feature_group('line')
def line(d):
    """並び予想から: ラインの人数・位置・ライン全体の強さ・先頭の脚質、先行争いの激しさ"""
    out = pd.DataFrame(index=d.index)
    lk = KEY + ['line_id']
    gl = d.groupby(lk, sort=False)
    out['line_size'] = gl['car'].transform('count')
    out['line_pos'] = d['line_pos']
    out['is_tanki'] = (out['line_size'] == 1).astype(int)
    out['line_score_sum'] = gl['score'].transform('sum')
    out['line_score_mean'] = gl['score'].transform('mean')
    # 並びのない行は除き、競り（同じ位置に2人）は前の行を使う（重複した索引で reindex できないため）
    lead = d[d['line_pos'] == 1].drop_duplicates(lk).set_index(lk)
    li = pd.MultiIndex.from_frame(d[lk])
    out['leader_score'] = lead['score'].reindex(li).to_numpy()
    out['leader_is_nige'] = (lead['style'].reindex(li).to_numpy() == '逃').astype(int)
    out['leader_b_count'] = lead['b_count'].reindex(li).to_numpy()
    out['leader_score_diff'] = d['score'].to_numpy() - out['leader_score']
    # レース内のライン順位（ライン合計得点ではなく平均で: 人数の多いラインが有利になりすぎないように）
    lines = d.groupby(lk, sort=False).agg(m=('score', 'mean'), lead=('score', 'first')).reset_index()
    lines['line_rank'] = lines.groupby(KEY)['m'].rank(ascending=False)
    lines['n_lines'] = lines.groupby(KEY)['m'].transform('count')
    lines['line_mean_gap'] = lines.groupby(KEY)['m'].transform('max') - lines['m']
    m = d[lk].merge(lines[lk + ['line_rank', 'n_lines', 'line_mean_gap']], on=lk, how='left')
    for c in ['line_rank', 'n_lines', 'line_mean_gap']:
        out[c] = m[c].to_numpy()
    # 先行争い: 先頭に立つ「逃」の選手の数、その中での自分のラインの先頭の得点順位
    lead_nige = (d['line_pos'] == 1) & (d['style'] == '逃')
    out['n_nige_leaders'] = lead_nige.groupby([d[c] for c in KEY]).transform('sum')
    ln = (d.loc[lead_nige].assign(r=lambda x: x.groupby(KEY)['score'].rank(ascending=False))
          .drop_duplicates(lk).set_index(lk)['r'])
    out['leader_nige_rank'] = ln.reindex(li).to_numpy()
    # 番手: 自分の前の選手（ラインで一つ前）の得点と脚質
    prev = (d[d['line_pos'].notna()].assign(line_pos=lambda x: x['line_pos'] + 1)
            .drop_duplicates(lk + ['line_pos']).set_index(lk + ['line_pos']))
    pi = pd.MultiIndex.from_frame(d[lk + ['line_pos']])
    out['front_score'] = prev['score'].reindex(pi).to_numpy()
    out['front_is_nige'] = (prev['style'].reindex(pi).to_numpy() == '逃').astype(float)
    out.loc[d['line_pos'] == 1, 'front_is_nige'] = np.nan
    out['same_pref_front'] = (prev['pref'].reindex(pi).to_numpy() == d['pref'].to_numpy()).astype(float)
    return out


# ---------------------------------------------------------------- レース条件

def stage_code(s):
    s = str(s)
    if '決勝' in s and '準' not in s:
        return 4
    if '準決' in s:
        return 3
    if any(k in s for k in ['特選', '選抜', 'ドリーム', 'シャイニング', '特秀', 'DMD', 'ダイヤモンド']):
        return 2
    if '予' in s:        # 予選・一予2・二予A・特予選・予2（KEIRIN.JP の略記）
        return 1
    return 0


@feature_group('race')
def race_conditions(d):
    """種別（予選/準決/決勝）・開催日目・場・天候・風、出走メンバーのレベルとばらつき"""
    out = pd.DataFrame(index=d.index)
    out['stage_code'] = d['stage'].map(stage_code)
    out['day_number'] = d['day_number']
    out['venue'] = d['venue']
    out['race_no'] = d['race']
    out['weather_code'] = d['weather'].map(WEATHER_CODES)
    out['wind_speed'] = d['wind_speed']
    out['field_score_mean'] = _g(d, 'score').transform('mean')
    out['field_score_std'] = _g(d, 'score').transform('std')
    out['n_ss_s1'] = d['rank'].isin(['SS', 'S1']).groupby([d[c] for c in KEY]).transform('sum')
    out['n_nige_style'] = (d['style'] == '逃').groupby([d[c] for c in KEY]).transform('sum')
    return out


# ---------------------------------------------------------------- 過去の成績（未来の情報は使わない）

@feature_group('history')
def history(d):
    """直近の着順・3着内率・上がりタイム、このバンク周長・この場での成績、前走からの間隔、今節の成績"""
    t = pd.to_datetime(d['date']) + pd.to_timedelta(d['race'].fillna(0) * 30, unit='min')
    h = d.assign(_t=t, _i=np.arange(len(d))).sort_values(['_t', 'venue', 'car'], kind='stable')
    done = h['place'].notna() | h.groupby(KEY)['place'].transform('count').gt(0)   # 結果が出ているレース
    h['place9'] = h['place'].fillna(9).where(done)            # 落車・失格は9着扱い
    h['top3'] = (h['place'] <= 3).astype(float).where(done)
    h['win'] = (h['place'] == 1).astype(float).where(done)
    h['agari_'] = h['agari'].where(done)
    out = pd.DataFrame(index=h.index)
    out['recent5_place'] = _prior_rolling(h, 'reg_no', 'place9', 5, min_periods=2)
    out['recent10_top3'] = _prior_rolling(h, 'reg_no', 'top3', 10, min_periods=3)
    out['recent20_win'] = _prior_rolling(h, 'reg_no', 'win', 20, min_periods=5)
    out['recent5_agari'] = _prior_rolling(h, 'reg_no', 'agari_', 5, min_periods=2)
    out['bank_top3'] = _prior_rolling(h, ['reg_no', 'bank'], 'top3', 20, min_periods=3)
    out['venue_top3'] = _prior_rolling(h, ['reg_no', 'venue'], 'top3', 15, min_periods=3)
    pos1 = h['top3'].where(h['line_pos'] == 1)
    out['as_leader_top3'] = _prior_rolling(h.assign(x=pos1), 'reg_no', 'x', 20, min_periods=3) \
        if pos1.notna().any() else np.nan
    days = pd.to_datetime(h['date'])
    out['days_since_last'] = (days - days.groupby(h['reg_no']).shift()).dt.days
    # 今節（同じ場で2日を超えて空いたら別の節）の平均着順
    gap = (days - days.groupby([h['reg_no'], h['venue']]).shift()).dt.days.fillna(99) > 2
    meet = gap.groupby([h['reg_no'], h['venue']]).cumsum()
    gm = h.assign(meet=meet, p=h['place9'].fillna(0), n=h['place9'].notna().astype(int)) \
        .groupby(['reg_no', 'venue', 'meet'], sort=False)
    n_before = gm['n'].cumsum() - h['place9'].notna().astype(int)
    out['meet_n'] = n_before
    out['meet_place'] = (gm['p'].cumsum() - h['place9'].fillna(0)) / n_before.replace(0, np.nan)
    out['recent_place_dev'] = out['recent5_place'] - out.groupby([h[c] for c in KEY])['recent5_place'] \
        .transform('mean')
    return out.reindex(d.index)


# ---------------------------------------------------------------- 組み立て

def build(d, groups=None):
    """groups（省略時は全部）の特徴量を付けた DataFrame と {グループ名: 列名リスト} を返す"""
    groups = list(groups or GROUPS)
    for gname in reversed(BASE_GROUPS):
        if gname not in groups:
            groups.insert(0, gname)
    d = d.copy()
    by_group, seen = {}, set()
    for gname in groups:
        new = GROUPS[gname][0](d)
        new = new[[c for c in new.columns if c not in seen]]
        d[list(new.columns)] = new
        by_group[gname] = list(new.columns)
        seen |= set(new.columns)
    return d, by_group


def columns(by_group, groups):
    return [c for g in groups for c in by_group[g]]
