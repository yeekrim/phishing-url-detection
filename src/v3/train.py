"""v3 학습: n-gram 선형 모델(스태킹) + LightGBM. (v3 3단계)

구성
    1) n-gram 선형 모델 (SGD, 로지스틱 손실) — 파일 단위로 순서대로 학습(out-of-core)
         url 모델: URL 블록만 / all 모델: URL + HTML + 본문 블록
       학습 데이터 안에서 도메인 묶음 5-겹 교차 예측(OOF) 점수를 만들고, 평가 데이터에는 전체로 학습한 모델 점수를 쓴다.
    2) LightGBM: 손으로 만든 피처 + 위 두 점수.

분할
    PhreshPhish 의 train/test 를 그대로 쓴다 (test 는 train 보다 나중 시점 — 시간 분할).
    train 안의 교차 검증은 등록 도메인(사설 접미사 반영) 묶음으로 나눠 같은 사이트가 양쪽에 들어가지 않게 한다.

출력: data/v3/interim/scores.parquet (sha256, split, label, 모델별 점수), models/v3/*.txt
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
ALPHAS = (3e-7, 3e-6)
TEXT_MODELS = {"url": slice(0, N_FEATURES), "all": slice(None)}
TEXT_COLUMNS = ["url_clean", "title", "text", "html_prefix"]
DROP = ["sha256", "url", "label", "target", "date", "lang", "parse_error", "tld"] + TEXT_COLUMNS
MODEL_DIR = os.path.join("models", "v3")
SCORES_PATH = os.path.join("data", "v3", "interim", "scores.parquet")


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


def train_text_models(df: pd.DataFrame) -> pd.DataFrame:
    """모든 (모델, alpha, 겹) 조합을 파일 한 번 읽을 때 같이 갱신한다."""
    keys = [(m, a, k) for m in TEXT_MODELS for a in ALPHAS for k in list(range(N_FOLDS)) + ["full"]]
    models = {key: SGDClassifier(loss="log_loss", alpha=key[1], random_state=0) for key in keys}
    train_shards = sorted(df.loc[df.split == "train", "shard"].unique())
    rng = np.random.default_rng(0)
    for epoch in range(EPOCHS):
        start = time.time()
        for shard in rng.permutation(train_shards):
            X = sp.load_npz(os.path.join(VECTOR_DIR, f"{shard}.npz"))
            meta = df[df.shard == shard].sort_values("row")
            y, folds = meta["label"].to_numpy(), meta["fold"].to_numpy()
            order = rng.permutation(len(y))
            X, y, folds = X[order], y[order], folds[order]
            for (m, a, k), model in models.items():
                mask = np.ones(len(y), bool) if k == "full" else folds != k
                model.partial_fit(X[mask][:, TEXT_MODELS[m]], y[mask], classes=[0, 1])
        print(f"  n-gram 학습 {epoch + 1}/{EPOCHS} 회차 {time.time() - start:.0f}초", flush=True)

    scores = {f"{m}_{a}": np.full(len(df), np.nan) for m in TEXT_MODELS for a in ALPHAS}
    for shard in df["shard"].unique():
        X = sp.load_npz(os.path.join(VECTOR_DIR, f"{shard}.npz"))
        idx = df.index[df.shard == shard].to_numpy()[np.argsort(df.loc[df.shard == shard, "row"].to_numpy())]
        folds = df.loc[idx, "fold"].to_numpy()
        for m, cols in TEXT_MODELS.items():
            for a in ALPHAS:
                out = scores[f"{m}_{a}"]
                if shard.startswith("test"):
                    out[idx] = models[(m, a, "full")].predict_proba(X[:, cols])[:, 1]
                else:
                    for k in range(N_FOLDS):
                        sel = folds == k
                        if sel.any():
                            out[idx[sel]] = models[(m, a, k)].predict_proba(X[sel][:, cols])[:, 1]

    # 교차 예측 점수의 평균 정밀도로 alpha 를 고른다 (평가 데이터는 보지 않음)
    train = (df.split == "train").to_numpy()
    chosen = {}
    for m in TEXT_MODELS:
        best = max(ALPHAS, key=lambda a: average_precision_score(df.label[train], scores[f"{m}_{a}"][train]))
        chosen[m] = best
        df[f"score_{m}"] = scores[f"{m}_{best}"]
    print(f"  선택한 alpha: {chosen}", flush=True)
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    skip = set(DROP) | {"shard", "row", "split", "group", "fold", "score_url", "score_all"}
    return [c for c in df.columns if c not in skip and df[c].dtype.kind in "ifb"]


def train_lgbm(df: pd.DataFrame, columns: list[str], name: str) -> np.ndarray:
    """fold 0 으로 조기 종료 횟수를 정한 뒤 train 전체로 다시 학습해 test 를 예측한다."""
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
    train = df[df.split == "train"]
    fit, val = train[train.fold != 0], train[train.fold == 0]
    booster = lgb.train(params, lgb.Dataset(fit[columns], fit.label), num_boost_round=3000,
                        valid_sets=[lgb.Dataset(val[columns], val.label)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = booster.best_iteration
    final = lgb.train(params, lgb.Dataset(train[columns], train.label), num_boost_round=rounds)
    os.makedirs(MODEL_DIR, exist_ok=True)
    final.save_model(os.path.join(MODEL_DIR, f"lgbm_{name}.txt"))
    print(f"  LightGBM[{name}] {len(columns)}개 피처, {rounds}회", flush=True)
    out = np.full(len(df), np.nan)
    out[(df.split == "test").to_numpy()] = final.predict(df.loc[df.split == "test", columns])
    return out


def main() -> None:
    start = time.time()
    df = load_table()
    print(f"불러옴: {len(df):,}행 (train {int((df.split == 'train').sum()):,} / test {int((df.split == 'test').sum()):,})",
          flush=True)
    df = train_text_models(df)
    num = numeric_columns(df)
    df["score_lgbm_features"] = train_lgbm(df, num, "features")
    df["score_lgbm_stacked"] = train_lgbm(df, num + ["score_url", "score_all"], "stacked")

    keep = ["sha256", "split", "label", "target", "date", "shard", "group",
            "score_url", "score_all", "score_lgbm_features", "score_lgbm_stacked"]
    df[keep].to_parquet(SCORES_PATH, index=False)
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as fh:
        json.dump(num, fh, ensure_ascii=False, indent=1)
    print(f"저장: {SCORES_PATH} ({time.time() - start:.0f}초)")


if __name__ == "__main__":
    main()
