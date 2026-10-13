"""v3 평가. (v3 4단계)

1) test 전체: ROC-AUC, 평균 정밀도(AP), 오탐률 1% / 0.1% 일 때의 탐지율
2) PhreshPhish 벤치마크: 피싱 비율 0.05% / 0.1% / 0.5% / 1% / 5% 의 평가 세트 975개.
   세트마다 AP 와 "탐지율 90% 일 때의 정밀도(P@R=0.9)" 를 구해 비율별로 평균한다.
   논문 표 4 와 같은 방식이라 기준 모델(Linear, FFN 등)과 바로 비교할 수 있다.
   (벤치마크 파일은 sha256 목록. benign.txt 는 모든 세트가 같이 쓰는 정상 목록)

출력: reports/v3/evaluation.md
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from extract_phreshphish import USER_AGENT
from train import SCORES_PATH

API = "https://huggingface.co/api/datasets/phreshphish/phreshphish/tree/main/benchmark"
RESOLVE = "https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/benchmark"
BENCH_DIR = os.path.join("data", "v3", "interim", "benchmark")
REPORT = os.path.join("reports", "v3", "evaluation.md")
MODELS = {"score_url": "n-gram 선형 (URL만)", "score_all": "n-gram 선형 (URL+HTML+본문)",
          "score_lgbm_features": "LightGBM (손으로 만든 피처만)", "score_lgbm_stacked": "LightGBM (피처 + n-gram 점수)"}
# 논문 표 4 (Dalton 외 2025) — 비교용 기준선
PAPER = {"Linear": {"0.05%": 0.2230, "0.1%": 0.3191, "0.5%": 0.5825, "1%": 0.6841, "5%": 0.8610},
         "FFN": {"0.05%": 0.2511, "0.1%": 0.3368, "0.5%": 0.5798, "1%": 0.6813, "5%": 0.8640},
         "GTE(BERT)": {"0.05%": 0.6050, "0.1%": 0.6879, "0.5%": 0.8354, "1%": 0.8762, "5%": 0.9382}}
RATES = {"benchmark-005": "0.05%", "benchmark-010": "0.1%", "benchmark-050": "0.5%",
         "benchmark-100": "1%", "benchmark-500": "5%"}


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    return resp


def fetch_benchmarks() -> None:
    """벤치마크 sha256 목록을 받아 둔다 (작은 텍스트 파일 976개, 이미 있으면 건너뜀)."""
    os.makedirs(BENCH_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(BENCH_DIR, "benign.txt")):
        with open(os.path.join(BENCH_DIR, "benign.txt"), "w") as fh:
            fh.write(_get(f"{RESOLVE}/benign.txt").text)
    for folder in RATES:
        os.makedirs(os.path.join(BENCH_DIR, folder), exist_ok=True)
        for item in _get(f"{API}/{folder}").json():
            name = item["path"].split("/")[-1]
            path = os.path.join(BENCH_DIR, folder, name)
            if not os.path.exists(path):
                with open(path, "w") as fh:
                    fh.write(_get(f"{RESOLVE}/{folder}/{name}").text)
                time.sleep(0.05)


def tpr_at_fpr(y, s, fpr_max: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())


def precision_at_recall(y, s, recall: float = 0.9) -> float:
    p, r, _ = precision_recall_curve(y, s)
    return float(p[r >= recall].max())


def read_list(path: str) -> list[str]:
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def main() -> None:
    fetch_benchmarks()
    df = pd.read_parquet(SCORES_PATH)
    test = df[df.split == "test"].set_index("sha256")
    lines = ["# v3 평가", "", f"test {len(test):,}행 (피싱 {int(test.label.sum()):,}), PhreshPhish 의 시간 분할 그대로", ""]

    rows = []
    for col, name in MODELS.items():
        y, s = test["label"], test[col]
        rows.append({"모델": name, "ROC-AUC": roc_auc_score(y, s), "AP": average_precision_score(y, s),
                     "탐지율@오탐 1%": tpr_at_fpr(y, s, 0.01), "탐지율@오탐 0.1%": tpr_at_fpr(y, s, 0.001)})
    lines += ["## test 전체", "", pd.DataFrame(rows).set_index("모델").round(4).to_markdown(), ""]

    benign = read_list(os.path.join(BENCH_DIR, "benign.txt"))
    missing = [h for h in benign if h not in test.index]
    benign = [h for h in benign if h in test.index]
    bench_rows, missing_phish = [], 0
    for folder, rate in RATES.items():
        files = sorted(os.listdir(os.path.join(BENCH_DIR, folder)))
        per = {col: {"ap": [], "p90": []} for col in MODELS}
        for f in files:
            phish = read_list(os.path.join(BENCH_DIR, folder, f))
            missing_phish += sum(h not in test.index for h in phish)
            idx = benign + [h for h in phish if h in test.index]
            sub = test.loc[idx]
            for col in MODELS:
                per[col]["ap"].append(average_precision_score(sub.label, sub[col]))
                per[col]["p90"].append(precision_at_recall(sub.label, sub[col]))
        for col, name in MODELS.items():
            bench_rows.append({"모델": name, "피싱 비율": rate, "세트 수": len(files),
                               "AP": np.mean(per[col]["ap"]), "P@R=0.9": np.mean(per[col]["p90"])})
    for paper_model, values in PAPER.items():
        for rate, ap in values.items():
            bench_rows.append({"모델": f"[논문] {paper_model}", "피싱 비율": rate, "AP": ap})
    bench = pd.DataFrame(bench_rows)
    table = bench.pivot_table(index="모델", columns="피싱 비율", values="AP")[list(RATES.values())]
    lines += ["## PhreshPhish 벤치마크 — 평균 AP (높을수록 좋음)", "",
              f"- 정상 {len(benign):,}개 공통, 벤치마크 목록 중 test 에 없는 정상 {len(missing):,}개 / 피싱 {missing_phish:,}건",
              "", table.round(4).to_markdown(), "",
              "### 탐지율 90% 일 때의 정밀도 (P@R=0.9)", "",
              bench.dropna(subset=["P@R=0.9"]).pivot_table(index="모델", columns="피싱 비율", values="P@R=0.9")
              [list(RATES.values())].round(4).to_markdown(), ""]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
