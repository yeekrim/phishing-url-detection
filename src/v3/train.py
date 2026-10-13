"""v3 학습: n-gram 선형 모델(스태킹) + LightGBM. (v3 3단계)

구성
    1) n-gram 선형 모델 (SGD, 로지스틱 손실) — 파일 단위로 순서대로 학습(out-of-core)
         url 모델: URL 블록만 / all 모델: URL + HTML + 본문 블록
       학습 데이터 안에서 도메인 묶음 5-겹 교차 예측(OOF) 점수를 만들고, 평가 데이터에는 전체로 학습한 모델 점수를 쓴다.
    2) LightGBM: 손으로 만든 피처 + 위 두 점수.

분할
    PhreshPhish 의 train/test 를 그대로 쓴다 (test 는 train 보다 나중 시점 — 시간 분할).
    train 안의 교차 검증은 등록 도메인(사설 접미사 반영) 묶음으로 나눠 같은 사이트가 양쪽에 들어가지 않게 한다.

설정 고르기 (평가 데이터는 보지 않는다)
    - 기준 점수: 피싱 비율 0.05~5% 에서의 평균 정밀도(AP) 평균. 벤치마크가 이 비율들로 채점하므로,
      검증 데이터의 피싱 행에 가중치를 줘서 같은 비율을 흉내 낸다 (low_rate_ap).
    - n-gram 모델: 규제 강도(alpha)와 호스팅 플랫폼 정상 가중치를 교차 예측 점수로 고른다.
    - LightGBM: train 의 가장 최근 15% 를 검증용으로 떼고(test 가 train 보다 나중이므로 같은 방향),
      그 검증 데이터와 도메인이 겹치는 행은 학습에서 뺀다. 여기서 설정·가중치·반복 횟수를 고른다.
    - 호스팅 플랫폼 정상 가중치: train 에서 호스팅 플랫폼 페이지는 정상 697개 / 피싱 68,007개라
      "플랫폼이면 피싱" 으로 배우기 쉽다. test 오탐(오탐률 0.1%)의 16.5% 가 플랫폼 정상 페이지였다(정상 전체 중 0.2%).
      그래서 플랫폼 정상 행의 학습 가중치를 키우는 후보를 같이 비교한다.

실행 (본문 전체 벡터 사용, extract_fulltext.py 결과):
    V3_VECTORS=data/v3/interim/vectors_full python src/v3/train.py
    본문 앞 2,000자 벡터(기본값)보다 교차 예측 기준 점수 0.8393 → 0.8455, 시간 검증 0.8622 → 0.8644 로 나아서 이쪽을 쓴다.

출력: data/v3/interim/scores.parquet (sha256, split, label, 모델별 점수), models/v3/*.txt,
      models/v3/train_config.json, reports/v3/tuning.md
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score

import features as F
from text_vectors import FEATURE_DIR, N_FEATURES, VECTOR_DIR

N_FOLDS = 5
EPOCHS = 3
ALPHAS = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
HOSTED_WEIGHTS = (1, 5, 20, 50)
RATES = (0.0005, 0.001, 0.005, 0.01, 0.05)   # 벤치마크의 피싱 비율
HOLDOUT_SHARE = 0.15
# 조기 종료도 기준 점수(low_rate_ap)로 한다. 기본값(log loss)으로 멈추면 최근 데이터에서 50~70회 만에
# 멈춰 덜 학습된 모델이 골라졌다 (첫 시도: 피처만 모델의 벤치마크 AP 0.392 → 0.268).
LGBM_BASE = dict(objective="binary", metric="None", learning_rate=0.05, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
LGBM_GRID = [dict(num_leaves=nl, min_data_in_leaf=md, feature_fraction=ff)
             for nl in (31, 127, 255) for md in (50, 300) for ff in (0.5, 0.8)]
SITE_LABEL = ["site_label_len", "site_label_digits", "site_label_hyphens", "site_label_entropy",
              "site_label_suspicious"]
TEXT_MODELS = {"url": slice(0, N_FEATURES), "all": slice(None)}
TEXT_COLUMNS = ["url_clean", "title", "text", "html_prefix"]
DROP = ["sha256", "url", "label", "target", "date", "lang", "parse_error", "tld"] + TEXT_COLUMNS
MODEL_DIR = os.path.join("models", "v3")
SCORES_PATH = os.path.join("data", "v3", "interim", "scores.parquet")
TUNING_REPORT = os.path.join("reports", "v3", "tuning.md")


def fold_of(group: str) -> int:
    return int(hashlib.sha256(group.encode("utf-8")).hexdigest(), 16) % N_FOLDS


def load_table() -> pd.DataFrame:
    """숫자 피처 + 메타. 파일 이름과 파일 안 순번을 같이 둬서 벡터 파일과 맞춘다."""
    frames = []
    for path in sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet"))):
        name = os.path.basename(path).removesuffix(".parquet")
        cols = [c for c in pq.read_schema(path).names if c not in ("html_prefix", "title", "text")]
        df = pd.read_parquet(path, columns=cols)
        df["shard"], df["row"] = name, np.arange(len(df))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["split"] = np.where(df["shard"].str.startswith("train"), "train", "test")
    df["group"] = df["url_clean"].map(lambda u: F.registered_domain(F.safe_split(u).hostname or ""))
    df["fold"] = np.where(df["split"] == "train", df["group"].map(fold_of), -1)
    return df


def low_rate_ap(y, s) -> float:
    """피싱 비율이 RATES 일 때의 AP 평균. 피싱 행에 가중치를 줘서 그 비율을 흉내 낸다."""
    y = np.asarray(y)
    n1, n0 = y.sum(), len(y) - y.sum()
    aps = [average_precision_score(y, s, sample_weight=np.where(y == 1, r * n0 / ((1 - r) * n1), 1.0))
           for r in RATES]
    return float(np.mean(aps))


def hosted_weight(df: pd.DataFrame, w: float) -> np.ndarray:
    return np.where((df["label"] == 0) & (df["is_hosted_platform"] == 1), w, 1.0)


# ---------------------------------------------------------------------------
# n-gram 선형 모델
# ---------------------------------------------------------------------------

def fit_text(df: pd.DataFrame, configs: list[tuple[str, float, float]]) -> dict:
    """(모델, alpha, 가중치) 조합마다 5-겹 + 전체 모델을 파일 한 번 읽을 때 같이 갱신하고 점수를 돌려준다."""
    models = {(c, k): SGDClassifier(loss="log_loss", alpha=c[1], random_state=0)
              for c in configs for k in list(range(N_FOLDS)) + ["full"]}
    train_shards = sorted(df.loc[df.split == "train", "shard"].unique())
    rng = np.random.default_rng(0)
    for epoch in range(EPOCHS):
        start = time.time()
        for shard in rng.permutation(train_shards):
            X = sp.load_npz(os.path.join(VECTOR_DIR, f"{shard}.npz"))
            meta = df[df.shard == shard].sort_values("row")
            order = rng.permutation(len(meta))
            X, meta = X[order], meta.iloc[order]
            y, folds = meta["label"].to_numpy(), meta["fold"].to_numpy()
            for k in list(range(N_FOLDS)) + ["full"]:
                mask = np.ones(len(y), bool) if k == "full" else folds != k
                parts = {m: X[mask][:, cols] for m, cols in TEXT_MODELS.items()}
                for c in configs:
                    m, _, w = c
                    models[(c, k)].partial_fit(parts[m], y[mask], classes=[0, 1],
                                               sample_weight=hosted_weight(meta[mask], w))
        print(f"  n-gram 학습 {epoch + 1}/{EPOCHS} 회차 {time.time() - start:.0f}초 ({len(configs)}개 조합)", flush=True)

    scores = {c: np.full(len(df), np.nan) for c in configs}
    for shard in df["shard"].unique():
        X = sp.load_npz(os.path.join(VECTOR_DIR, f"{shard}.npz"))
        idx = df.index[df.shard == shard].to_numpy()[np.argsort(df.loc[df.shard == shard, "row"].to_numpy())]
        folds = df.loc[idx, "fold"].to_numpy()
        for c in configs:
            Xm = X[:, TEXT_MODELS[c[0]]]
            if shard.startswith("test"):
                scores[c][idx] = models[(c, "full")].predict_proba(Xm)[:, 1]
            else:
                for k in range(N_FOLDS):
                    sel = folds == k
                    if sel.any():
                        scores[c][idx[sel]] = models[(c, k)].predict_proba(Xm[sel])[:, 1]
    return scores


def train_text_models(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """1단계: 가중치 1 에서 alpha 고르기 → 2단계: 고른 alpha 에서 호스팅 정상 가중치 고르기."""
    train = (df.split == "train").to_numpy()
    y = df.label[train]

    def oof(s):
        return low_rate_ap(y, s[train])

    stage1 = fit_text(df, [(m, a, 1) for m in TEXT_MODELS for a in ALPHAS])
    log += ["## n-gram 모델 (train 교차 예측, 기준 점수 = 피싱 비율 0.05~5% AP 평균)", "",
            "| 모델 | alpha | 가중치 | 기준 점수 |", "|---|---|---|---|"]
    log += [f"| {c[0]} | {c[1]:g} | {c[2]} | {oof(s):.4f} |" for c, s in stage1.items()]
    best_alpha = {m: max(ALPHAS, key=lambda a: oof(stage1[(m, a, 1)])) for m in TEXT_MODELS}
    stage2 = fit_text(df, [(m, best_alpha[m], w) for m in TEXT_MODELS for w in HOSTED_WEIGHTS if w != 1])
    log += [f"| {c[0]} | {c[1]:g} | {c[2]} | {oof(s):.4f} |" for c, s in stage2.items()]
    candidates = {**stage1, **stage2}
    chosen = {}
    for m in TEXT_MODELS:
        c = max([(m, best_alpha[m], w) for w in HOSTED_WEIGHTS], key=lambda c: oof(candidates[c]))
        chosen[m] = {"alpha": c[1], "hosted_weight": c[2], "oof_score": oof(candidates[c])}
        df[f"score_{m}"] = candidates[c]
    log += ["", f"선택: {chosen}", ""]
    print(f"  n-gram 선택: {chosen}", flush=True)
    return df, chosen


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def numeric_columns(df: pd.DataFrame) -> list[str]:
    skip = set(DROP) | {"shard", "row", "split", "group", "fold", "score_url", "score_all"}
    return [c for c in df.columns if c not in skip and df[c].dtype.kind in "ifb"]


def holdout(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train 의 가장 최근 15% = 검증. 검증과 같은 도메인 묶음은 학습에서 뺀다."""
    date = pd.to_datetime(train["date"], errors="coerce")
    cutoff = date.quantile(1 - HOLDOUT_SHARE)
    val = train[date >= cutoff]
    fit = train[(date < cutoff) & ~train["group"].isin(set(val["group"]))]
    return fit, val


def _low_rate_metric(preds, data):
    return "low_rate_ap", low_rate_ap(data.get_label(), preds), True


def lgbm_try(fit, val, columns, params, w) -> tuple[float, int]:
    booster = lgb.train({**LGBM_BASE, **params},
                        lgb.Dataset(fit[columns], fit.label, weight=hosted_weight(fit, w)),
                        num_boost_round=3000, valid_sets=[lgb.Dataset(val[columns], val.label)],
                        feval=_low_rate_metric, callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster.best_score["valid_0"]["low_rate_ap"], booster.best_iteration


def train_lgbm(df: pd.DataFrame, columns: list[str], name: str, log: list[str]) -> tuple[np.ndarray, dict]:
    """시간 검증으로 설정 → 호스팅 정상 가중치 순서로 고른 뒤 train 전체로 다시 학습해 test 를 예측한다."""
    train = df[df.split == "train"]
    fit, val = holdout(train)
    log += [f"## LightGBM [{name}] (시간 검증: 학습 {len(fit):,}행 / 검증 {len(val):,}행, "
            f"검증의 호스팅 정상 {int(((val.label == 0) & (val.is_hosted_platform == 1)).sum())}행)", "",
            "| 설정 | 가중치 | 반복 | 기준 점수 |", "|---|---|---|---|"]
    results = {}
    for params in LGBM_GRID:
        results[(json.dumps(params), 1)] = lgbm_try(fit, val, columns, params, 1)
    best_params = max(LGBM_GRID, key=lambda p: results[(json.dumps(p), 1)][0])
    for w in HOSTED_WEIGHTS[1:]:
        results[(json.dumps(best_params), w)] = lgbm_try(fit, val, columns, best_params, w)
    for (p, w), (score, rounds) in results.items():
        log.append(f"| {p} | {w} | {rounds} | {score:.4f} |")
    (p, w), (score, rounds) = max(results.items(), key=lambda kv: kv[1][0])
    params = json.loads(p)
    no_site = [c for c in columns if c not in SITE_LABEL]
    ablation, _ = lgbm_try(fit, val, no_site, params, w)
    log += ["", f"선택: {params}, 가중치 {w}, 반복 {rounds}, 기준 점수 {score:.4f} "
            f"(사이트 이름 피처를 빼면 {ablation:.4f})", ""]

    final = lgb.train({**LGBM_BASE, **params}, lgb.Dataset(train[columns], train.label, weight=hosted_weight(train, w)),
                      num_boost_round=rounds)
    os.makedirs(MODEL_DIR, exist_ok=True)
    final.save_model(os.path.join(MODEL_DIR, f"lgbm_{name}.txt"))
    print(f"  LightGBM[{name}] {len(columns)}개 피처, {params}, 가중치 {w}, {rounds}회, 검증 {score:.4f}", flush=True)
    out = np.full(len(df), np.nan)
    out[(df.split == "test").to_numpy()] = final.predict(df.loc[df.split == "test", columns])
    return out, {"params": params, "hosted_weight": w, "rounds": rounds, "val_score": score,
                 "val_score_without_site_label": ablation}


def main() -> None:
    start = time.time()
    df = load_table()
    print(f"불러옴: {len(df):,}행 (train {int((df.split == 'train').sum()):,} / test {int((df.split == 'test').sum()):,})",
          flush=True)
    log = ["# v3 설정 고르기", "", "평가(test) 데이터는 쓰지 않았다. 기준 점수 = 피싱 비율 0.05/0.1/0.5/1/5% 에서의 AP 평균.", ""]
    df, text_cfg = train_text_models(df, log)
    num = numeric_columns(df)
    df["score_lgbm_features"], feat_cfg = train_lgbm(df, num, "features", log)
    df["score_lgbm_stacked"], stack_cfg = train_lgbm(df, num + ["score_url", "score_all"], "stacked", log)

    keep = ["sha256", "split", "label", "target", "date", "shard", "group",
            "score_url", "score_all", "score_lgbm_features", "score_lgbm_stacked"]
    df[keep].to_parquet(SCORES_PATH, index=False)
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as fh:
        json.dump(num, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(MODEL_DIR, "train_config.json"), "w") as fh:
        json.dump({"text": text_cfg, "lgbm_features": feat_cfg, "lgbm_stacked": stack_cfg}, fh, indent=1)
    os.makedirs(os.path.dirname(TUNING_REPORT), exist_ok=True)
    with open(TUNING_REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))
    print(f"저장: {SCORES_PATH} ({time.time() - start:.0f}초)")


if __name__ == "__main__":
    main()
