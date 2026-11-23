# 競輪予想システム

BoatRace リポジトリ（`real_data_model.py` / `feature_ablation.py` / `all_venue_v2.py` / `predict_today.py`）と
同じ流れを競輪向けに作り直したもの。目的は **特徴量の研究を続けること** なので、特徴量はグループ単位で
足し外しでき、比べた結果が毎回 `research/log.csv` に残るようにしてある。

```
データ集め            特徴量                  モデル                    研究・予想
fetch_data.py  →  features.py         →  model.py              →  research.py（比べる）
（schema.py の形）   @feature_group で登録    レース内ソフトマックス      train.py → predict.py → check_results.py
```

## 必要なもの

```bash
pip install -r requirements.txt
```

## すぐ試す（合成データ）

本物のデータが入るまでは、`sample_data.py` の合成データで最後まで動くことを確認できる。
**合成データの的中率・回収率には意味がない**（作るときの仮定を当てているだけ）。

```bash
python sample_data.py --out data/sample.csv                       # 約3分、1.8万レース
python research.py data/sample.csv --test-from 2026-09-01 --no-log  # 特徴量グループの比較（約1分）
python train.py data/sample.csv --until 2026-11-19
python predict.py 出走表.csv --history data/sample.csv --out predictions/20261120.csv
python check_results.py predictions/20261120_tickets.csv data/sample.csv
```

## データの形（schema.py）

1行 = 1選手。レースは `date, venue, race` で決まる。過去データも当日の出走表も同じ形で、
当日の出走表では結果の列（`place` など）を空欄にする。

| 区分 | 主な列 |
|---|---|
| レース | date, venue, race, grade(GP〜F2), race_class(S/A/A-chal/L), stage(予選・準決勝・決勝…), day_number, bank(333/400/500), weather, wind_speed |
| 選手 | car(車番), reg_no, pref(府県), term(期別), rank(SS〜A3/L1), age, style(逃/両/追), gear, score(競走得点), win_rate, top2_rate, top3_rate, nige/makuri/sashi/mark(決まり手回数), s_count, b_count |
| 並び予想 | line_id（ライン番号）, line_pos（1=先頭, 2=番手…）。出走表では `narabi` 列に `17/243/586/9` と書いてもよい |
| 結果 | place, kimarite, agari(上がり), s_flag, b_flag, exacta_pay(2車単), quinella_pay, trifecta_pay, trio_pay |

## データ集め（KEIRIN.JP から）

```bash
python fetch_data.py fetch 20261101 20261122 --out data/races.csv   # 過去の出走表＋結果を足していく
python fetch_data.py card 20261124 --out cards/20261124.csv         # 当日・翌日の出走表（並び予想つき）
python fetch_data.py check data/races.csv                           # 期間・列の埋まり具合・並びの有無
python fetch_data.py merge data/raw/*.csv --out data/races.csv      # 手元の CSV をまとめる
```

`keirinjp.py` が KEIRIN.JP の画面の裏で使われている JSON（`/pc/json?type=JSJxxx`）を読む。

- **個人の研究用に限る。** robots.txt はこのページの機械的な読み込みを認めていない。サイトポリシーは私的使用のための
  複製を認め、第三者から金銭を得ることを禁じている。リクエストは 2 秒以上あける（`keirinjp.MIN_INTERVAL`）。
- 1 場 1 日でおよそ 28 回（`--no-gear` なら 16 回）、1 分ほど。1 日 8〜10 場なので **1 日分で約 10 分、1 か月で約 5 時間**。
  1 日ごとに `--out` に保存するので、止めても同じコマンドで続きから回せる。過去日の JSON は `data/raw/keirinjp/` に
  残すので、二度は読まない。
- **並び予想は当日のうちしか取れない**（レースが終わった日の分は消える）。過去日の `line_id/line_pos` は空欄になる。
  並びを貯めるには毎日 `fetch 今日` を回す。`merge` は空欄で上書きしないので、朝に取った並びは夜の取り直しで消えない。
- 毎日の取得は `python daily.py`（日本時間の今日の出走表＋並び予想、昨日の結果を `data/daily/YYYYMMDD.csv` に保存）。
  毎朝 6:47（日本時間）に自動で回し、ブランチ `claude/keirin-data` にコミットしている。まとめるときは
  `python fetch_data.py merge data/daily/*.csv --out data/races.csv`。
- 取れない列: `s_flag`（そのレースでスタートを取ったか）は結果に出ないので空欄。
- 追加の列: `race_name`（開催名）、`laps`（周回数）、`h_count`（ホーム取り回数）、`h_flag`（このレースでHを取ったか）、
  `note`（補充・追加など）、`result_note`（落車棄権・失格など、着順が空欄の理由）。

## 特徴量グループ（features.py）

| グループ | 中身 |
|---|---|
| base（基準） | 出走表の値（競走得点・勝率・脚質・級班・ギヤ・S/B回数・決まり手の割合）、レース内の得点偏差・順位・トップとの差 |
| line | ラインの人数・位置・単騎か、ラインの平均/合計得点・順位、先頭の得点と脚質、番手から見た前の選手、同県か、逃げ先頭の数（先行争い） |
| race | 種別・開催日目・場・天候・風、出走メンバーの得点の平均とばらつき、SS/S1 の人数、逃げ脚質の人数 |
| history | 直近の着順・3着内率・勝率・上がり、このバンク周長／この場での3着内率、先頭を任されたときの3着内率、前走からの日数、今節の成績 |

過去の成績から作るもの（history）は、そのレースより前の結果だけを使う（`_prior_rolling`）。

### 特徴量を足す

```python
# features.py
@feature_group('my_idea')
def my_idea(d):
    """1行目が research.py の一覧に出る説明"""
    out = pd.DataFrame(index=d.index)
    out['gear_dev'] = d['gear'] - _g(d, 'gear').transform('mean')
    return out
```

```bash
python research.py data/races.csv --groups my_idea             # 基準 vs 基準+my_idea
python research.py data/races.csv --drop                       # 全部から1つずつ外す（効いているかの確認）
python research.py data/races.csv --extra line:front_score     # 列1本だけ試す
```

## モデル（model.py）

- 1着確率: その場の 7〜9 人の中で誰が勝つかを LightGBM のレース内ソフトマックスで直接学習（人数が違ってもよい）
- 2車単・3連単・3連複: Harville 式（1着確率を残りの選手で割り直して使う）
- 評価: 対数損失、1着的中、2車単/3連単/3連複 の上位 k 点的中、払戻があれば上位 k 点を均等買いした回収率
- 比較用に「競走得点だけ」の成績も毎回出す

## 次にやること

1. 毎日 `fetch` を回して並び予想を貯める（過去分の並びは KEIRIN.JP からは取れない。line グループは貯まってから効く）
2. 2着・3着を Harville ではなく条件付きモデルにする（BoatRace の all_venue_v2 と同じ考え方: 1着がライン先頭なら番手が2着に来やすい、など）
3. オッズを取って期待値で買い目を選ぶ（BoatRace の all_venue_v4 と同じ）
