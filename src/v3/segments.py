"""페이지 종류별 모델 시험. (v3)

생각: 피싱의 "성격" 은 예측할 때 알 수 없으므로, 누구나 볼 수 있는 페이지 겉모습으로 묶음을 나누고
묶음마다 n-gram 선형 모델 또는 LightGBM 을 따로 둔다. 선형 모델은 모든 페이지에 같은 가중치를 쓰므로
("login" 이 로그인 화면에 나올 때와 블로그에 나올 때를 구분 못 함) 묶음별 모델이 도움이 될 수 있다.

묶음 기준 (근거)
    crp   : 비밀번호 입력란이 있거나, 폼 + 이메일 입력란이 있음 — PhishIntention(Liu 외, USENIX Security 2022)의
            "자격증명을 요구하는 페이지(CRP)". train 에서 n-gram 교차 예측 기준 점수가 CRP 안 0.926 / 밖 0.816 으로 달랐다.
    blank : 본문 50자 미만이거나 태그 20개 미만 — CrawlPhish(Zhang 외, IEEE S&P 2021)의 클라이언트 측 가림(cloaking)처럼
            내용이 비어 보이는 페이지. train 정상 7,248 / 피싱 30,166.
    빼 둔 기준: 호스팅 플랫폼(train 정상 2,369 개뿐), 루트 페이지·언어(안팎 점수 차이 작음),
            "깊은 경로·본문 적음·태그 많음"(오차 나무로 찾았지만 train 정상 285 개뿐).

비교 (모두 train 의 시간 검증, test 는 보지 않음)
    base      : 지금 모델 (피처 + 전체 n-gram 점수 2개)
    +seg      : base + 묶음별 n-gram 점수 2개
    per-seg   : 묶음마다 LightGBM 을 따로 학습 (입력은 base 와 같음)
나누는 방식은 두 가지: crp 2묶음 (crp / 그 외), blank_crp 3묶음 (blank / crp / 그 외, blank 를 먼저 판정).

출력: data/v3/interim/segment_scores.parquet, reports/v3/segments.md
"""

from __future__ import annotations

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier

import train as T
from text_vectors import VECTOR_DIR

OUT = os.path.join("data", "v3", "interim", "segment_scores.parquet")
REPORT = os.path.join("reports", "v3", "segments.md")


def crp(df: pd.DataFrame) -> np.ndarray:
    return ((df.n_password_inputs > 0) | ((df.n_forms > 0) & (df.n_email_inputs > 0))).to_numpy()


def blank(df: pd.DataFrame) -> np.ndarray:
    return ((df.text_len < 50) | (df.n_tags < 20)).to_numpy()


SCHEMES = {
    "crp": lambda df: np.where(crp(df), "crp", "other"),
    "blank_crp": lambda df: np.select([blank(df), crp(df)], ["blank", "crp"], "other"),
}


def segment_text_scores(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """묶음별 n-gram 모델 (alpha·호스팅 가중치는 train.py 가 고른 값). train 은 교차 예측, test 는 전체 모델."""
    segs = {name: fn(df) for name, fn in SCHEMES.items()}
    keys = [(sc, s, m, k) for sc in SCHEMES for s in np.unique(segs[sc]) for m in T.TEXT_MODELS
            for k in list(range(T.N_FOLDS)) + ["full"]]
    models = {key: SGDClassifier(loss="log_loss", alpha=cfg[key[2]]["alpha"], random_state=0) for key in keys}
    shards = sorted(df.loc[df.split == "train", "shard"].unique())
    rng = np.random.default_rng(0)
    for epoch in range(T.EPOCHS):
        start = time.time()
        for shard in rng.permutation(shards):
            X = sp.load_npz(os.path.join(VECTOR_DIR, f"{shard}.npz"))
            idx = df.index[df.shard == shard].to_numpy()[np.argsort(df.loc[df.shard == shard, "row"].to_numpy())]
            order = rng.permutation(len(idx))
            X, idx = X[order], idx[order]
            meta = df.loc[idx]
            y, folds = meta.label.to_numpy(), meta.fold.to_numpy()
            for (sc, s, m, k), model in models.items():
                mask = (segs[sc][idx] == s) & (np.ones(len(y), bool) if k == "full" else folds != k)
                if mask.any():
                    model.partial_fit(X[mask][:, T.TEXT_MODELS[m]], y[mask], classes=[0, 1],
                                      sample_weight=T.hosted_weight(meta[mask], cfg[m]["hosted_weight"]))
        print(f"  묶음별 n-gram {epoch + 1}/{T.EPOCHS} 회차 {time.time() - start:.0f}초", flush=True)

    out = pd.DataFrame({"sha256": df.sha256})
    for sc in SCHEMES:
        out[f"seg_{sc}"] = segs[sc]
        for m in T.TEXT_MODELS:
            out[f"seg_{sc}_{m}"] = np.nan
    for shard in df.shard.unique():
        X = sp.load_npz(os.path.join(VECTOR_DIR, f"{shard}.npz"))
        idx = df.index[df.shard == shard].to_numpy()[np.argsort(df.loc[df.shard == shard, "row"].to_numpy())]
        folds = df.loc[idx, "fold"].to_numpy()
        for sc in SCHEMES:
            seg = segs[sc][idx]
            for m, cols in T.TEXT_MODELS.items():
                Xm = X[:, cols]
                for s in np.unique(seg):
                    for k in (["full"] if shard.startswith("test") else range(T.N_FOLDS)):
                        sel = (seg == s) if k == "full" else (seg == s) & (folds == k)
                        if sel.any():
                            out.loc[idx[sel], f"seg_{sc}_{m}"] = models[(sc, s, m, k)].predict_proba(Xm[sel])[:, 1]
    return out


def per_segment_lgbm(fit, val, columns, params, w, seg_fit, seg_val) -> tuple[float, dict]:
    """묶음마다 LightGBM 을 따로 학습해 검증 예측을 합친 뒤 기준 점수를 잰다."""
    pred = np.zeros(len(val))
    rounds = {}
    for s in np.unique(seg_fit):
        f, v = fit[seg_fit == s], val[seg_val == s]
        booster = lgb.train({**T.LGBM_BASE, **params}, lgb.Dataset(f[columns], f.label, weight=T.hosted_weight(f, w)),
                            num_boost_round=3000, valid_sets=[lgb.Dataset(v[columns], v.label)],
                            feval=T._low_rate_metric, callbacks=[lgb.early_stopping(100, verbose=False)])
        pred[seg_val == s] = booster.predict(v[columns], num_iteration=booster.best_iteration)
        rounds[s] = booster.best_iteration
    return T.low_rate_ap(val.label, pred), rounds


def main() -> None:
    df = T.load_table()
    with open(os.path.join(T.MODEL_DIR, "train_config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(T.MODEL_DIR, "feature_columns.json")) as fh:
        num = json.load(fh)
    df = df.merge(pd.read_parquet(T.SCORES_PATH)[["sha256", "score_url", "score_all"]], on="sha256", how="left")
    if os.path.exists(OUT):
        seg = pd.read_parquet(OUT)
    else:
        seg = segment_text_scores(df, cfg["text"])
        seg.to_parquet(OUT, index=False)
    df = df.merge(seg, on="sha256", how="left")

    fit, val = T.holdout(df[df.split == "train"])
    stack = cfg["lgbm_stacked"]
    base = num + ["score_url", "score_all"]
    lines = ["# 페이지 종류별 모델 시험 (train 시간 검증, 기준 점수 = 피싱 비율 0.05~5% AP 평균)", "",
             f"학습 {len(fit):,}행 / 검증 {len(val):,}행. LightGBM 설정은 train_config.json 의 스태킹 설정 그대로.", ""]
    for sc in SCHEMES:
        counts = val.groupby([f"seg_{sc}", "label"]).size().unstack(fill_value=0)
        lines += [f"검증 데이터 묶음 구성 ({sc}):", "", counts.to_markdown(), ""]
    rows = [{"방식": "base (지금)", "기준 점수": T.lgbm_try(fit, val, base, stack["params"], stack["hosted_weight"])[0]}]
    for sc in SCHEMES:
        extra = [f"seg_{sc}_url", f"seg_{sc}_all"]
        rows.append({"방식": f"+seg ({sc})",
                     "기준 점수": T.lgbm_try(fit, val, base + extra, stack["params"], stack["hosted_weight"])[0]})
        score, rounds = per_segment_lgbm(fit, val, base, stack["params"], stack["hosted_weight"],
                                         fit[f"seg_{sc}"].to_numpy(), val[f"seg_{sc}"].to_numpy())
        rows.append({"방식": f"per-seg ({sc})", "기준 점수": score, "반복": json.dumps(rounds)})
        print(rows[-2], rows[-1], flush=True)
    # n-gram 점수만 비교 (교차 예측, train 전체)
    tr = df[df.split == "train"]
    text_rows = [{"점수": "score_all (전체 모델)", "기준 점수": T.low_rate_ap(tr.label, tr.score_all)}]
    text_rows += [{"점수": f"seg_{sc}_all", "기준 점수": T.low_rate_ap(tr.label, tr[f"seg_{sc}_all"])} for sc in SCHEMES]
    lines += ["## LightGBM (시간 검증)", "", pd.DataFrame(rows).round(4).to_markdown(index=False), "",
              "## n-gram URL+HTML+본문 점수만 (train 교차 예측)", "", pd.DataFrame(text_rows).round(4).to_markdown(index=False), ""]
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
