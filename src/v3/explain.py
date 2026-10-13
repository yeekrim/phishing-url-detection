"""v3 설명: SHAP 과 n-gram 가중치. (v3 5단계)

1) LightGBM SHAP — LightGBM 내장 TreeSHAP(pred_contrib)으로 test 표본의 피처별 기여도를 구한다.
   스태킹 모델(피처 + n-gram 점수)과 피처만 쓴 모델 둘 다. 피처만 쓴 모델이 "어떤 성질이 피싱다운가" 를 보기 쉽다.
2) n-gram 가중치 — 판정의 대부분이 n-gram 점수에서 나오므로, 선형 모델의 가중치를 원래 글자·단어 조각으로
   되돌려 본다. 해시 번호 → 조각 대응표는 학습 데이터 일부에서 조각을 모아 다시 해시해서 만든다.
   목적: 데이터셋 제작 과정의 흔적(가림 표시, 수집기 오류 페이지 등)이 지름길로 쓰이는지 확인.

출력: reports/v3/explain.md
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.utils import murmurhash3_32

from text_vectors import BLOCKS, FEATURE_DIR, N_FEATURES, VECTOR_DIR, VECTORIZERS
from train import MODEL_DIR, SCORES_PATH, hosted_weight, load_table

REPORT = os.path.join("reports", "v3", "explain.md")
SAMPLE = 30_000
TOKEN_SHARDS = 4   # 해시 번호 → 조각 대응표를 만들 학습 파일 수
TOP = 25


def lgbm_shap(df: pd.DataFrame, name: str) -> pd.DataFrame:
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lgbm_{name}.txt"))
    cols = booster.feature_name()
    test = df[df.split == "test"]
    sample = pd.concat([test[test.label == k].sample(n=SAMPLE // 2, random_state=0) for k in (0, 1)])
    contrib = booster.predict(sample[cols], pred_contrib=True)[:, :-1]  # 마지막 열은 기준값
    out = pd.DataFrame({
        "평균 |SHAP|": np.abs(contrib).mean(axis=0),
        # 피처 값이 클수록 피싱 쪽(+) 인지 정상 쪽(-) 인지: 값과 SHAP 의 순위 상관
        "방향(값↑→)": [pd.Series(sample[c].to_numpy()).corr(pd.Series(contrib[:, i]), method="spearman")
                    for i, c in enumerate(cols)],
        "정상 평균값": sample[sample.label == 0][cols].mean().to_numpy(),
        "피싱 평균값": sample[sample.label == 1][cols].mean().to_numpy(),
    }, index=cols)
    out["비중"] = out["평균 |SHAP|"] / out["평균 |SHAP|"].sum()
    return out.sort_values("평균 |SHAP|", ascending=False)


def fit_full_sgd(df: pd.DataFrame) -> SGDClassifier:
    """train 전체로 URL+HTML+본문 n-gram 모델을 다시 학습 (train.py 는 점수만 저장하므로).
    alpha 와 호스팅 정상 가중치는 train.py 가 고른 값(train_config.json)을 쓴다."""
    with open(os.path.join(MODEL_DIR, "train_config.json")) as fh:
        cfg = json.load(fh)["text"]["all"]
    model = SGDClassifier(loss="log_loss", alpha=cfg["alpha"], random_state=0)
    shards = sorted(df.loc[df.split == "train", "shard"].unique())
    rng = np.random.default_rng(0)
    for _ in range(3):
        for shard in rng.permutation(shards):
            X = sp.load_npz(os.path.join(VECTOR_DIR, f"{shard}.npz"))
            meta = df[df.shard == shard].sort_values("row")
            order = rng.permutation(len(meta))
            model.partial_fit(X[order], meta["label"].to_numpy()[order], classes=[0, 1],
                              sample_weight=hosted_weight(meta, cfg["hosted_weight"])[order])
    return model


def hashed_column(token: str) -> int:
    """HashingVectorizer 와 같은 열 번호 (부호 있는 murmurhash3 의 절댓값 % 열 수)."""
    return abs(murmurhash3_32(token, seed=0)) % N_FEATURES


def token_index() -> dict[int, set[str]]:
    """전체 열 번호 → 그 번호로 해시되는 조각들 (학습 파일 몇 개에서 수집)."""
    index: dict[int, set[str]] = defaultdict(set)
    paths = sorted(glob.glob(os.path.join(FEATURE_DIR, "train-*.parquet")))
    for path in paths[:: max(1, len(paths) // TOKEN_SHARDS)][:TOKEN_SHARDS]:
        df = pd.read_parquet(path, columns=["url_clean", "html_prefix", "title", "text"])
        docs = {"url": df["url_clean"], "html": df["html_prefix"], "text": df["title"] + " " + df["text"]}
        for b, block in enumerate(BLOCKS):
            analyzer = VECTORIZERS[block].build_analyzer()
            for doc in docs[block].fillna(""):
                for t in analyzer(doc):
                    index[b * N_FEATURES + hashed_column(t)].add(t)
    return index


def top_tokens(model: SGDClassifier, index: dict[int, set[str]]) -> list[str]:
    coef = model.coef_.ravel()
    lines = []
    for b, block in enumerate(BLOCKS):
        sl = slice(b * N_FEATURES, (b + 1) * N_FEATURES)
        w = coef[sl]
        rows = []
        for direction, order in (("피싱 쪽", np.argsort(-w)), ("정상 쪽", np.argsort(w))):
            shown = 0
            for j in order:
                toks = index.get(b * N_FEATURES + j)
                if not toks:
                    continue
                rows.append({"방향": direction, "가중치": round(float(w[j]), 3),
                             "조각": " / ".join(sorted(toks, key=len)[:3]).replace("|", "¦")})
                shown += 1
                if shown == TOP:
                    break
        lines += [f"### n-gram 블록: {block}", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    return lines


def main() -> None:
    df = load_table()
    scores = pd.read_parquet(SCORES_PATH)[["sha256", "score_url", "score_all"]]
    df = df.merge(scores, on="sha256", how="left")
    lines = ["# v3 설명 (SHAP · n-gram 가중치)", "", f"test 표본 {SAMPLE:,}개 (정상·피싱 반반)", ""]
    for name, title in (("stacked", "LightGBM (피처 + n-gram 점수)"), ("features", "LightGBM (손으로 만든 피처만)")):
        table = lgbm_shap(df, name)
        lines += [f"## SHAP — {title}", "", "방향: 피처 값이 클수록 피싱 쪽이면 +, 정상 쪽이면 -", "",
                  table.head(30).round(3).to_markdown(), ""]
    print("SHAP 완료, n-gram 모델 재학습 중", flush=True)
    model = fit_full_sgd(df)
    lines += ["## n-gram 가중치 상위 조각 (URL+HTML+본문 선형 모델)", "",
              f"해시 충돌로 한 번호에 여러 조각이 들어갈 수 있다 (대응표: 학습 파일 {TOKEN_SHARDS}개에서 수집)", ""]
    lines += top_tokens(model, token_index())
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"저장: {REPORT}")


if __name__ == "__main__":
    main()
