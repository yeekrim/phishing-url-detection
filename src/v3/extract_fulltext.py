"""본문 전체 n-gram 벡터 만들기. (v3 B)

text 블록은 제목 + 본문 앞 2,000자만 쓴다. train 파일 4개로 학습·다른 1개로 검증한 시험에서
본문 전체(최대 200,000자)를 쓰면 기준 점수(피싱 비율 0.05~5% AP 평균)가 0.721 → 0.759 였다.
검증 데이터가 작아 확실하지 않으므로 전체 데이터로 다시 확인하기 위해 만든다.

본문 전체 문자열을 저장하면 디스크가 모자라서(예상 수십 GB), 파일마다
    받기 → 본문 전체 추출(여러 프로세스) → text 블록만 다시 벡터화 → 기존 url·html 블록과 붙여 저장 → 원본 삭제
를 반복한다. 행 순서는 features/<파일>.parquet 의 sha256 순서와 같은지 확인한다.

출력: data/v3/interim/vectors_full/<파일>.npz  (V3_VECTORS 로 학습에 쓴다)
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

import features as F
from extract_phreshphish import BATCH, OUT_DIR as FEATURE_DIR, download
from text_vectors import N_FEATURES, VECTORIZERS

BASE_DIR = os.path.join("data", "v3", "interim", "vectors")
OUT_DIR = os.path.join("data", "v3", "interim", "vectors_full")


def _text(rec: dict) -> str:
    try:
        _, t = F.html_features(rec["html"], rec["url"], text_chars=F.FULL_TEXT_CHARS)
    except Exception:  # noqa: BLE001 — extract_phreshphish 와 같은 처리: 파싱 실패 행은 빈 본문
        return ""
    return t["title"] + " " + t["text"]


def process(name: str, pool: Pool) -> None:
    start = time.time()
    order = pd.read_parquet(os.path.join(FEATURE_DIR, f"{name}.parquet"), columns=["sha256"])["sha256"].to_numpy()
    raw = download(name)
    shas, blocks = [], []
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=BATCH, columns=["sha256", "url", "html"]):
        recs = batch.to_pylist()
        shas += [r["sha256"] for r in recs]
        blocks.append(VECTORIZERS["text"].transform(pool.map(_text, recs, chunksize=16)))
    os.remove(raw)
    if not np.array_equal(np.array(shas), order):
        raise RuntimeError(f"{name}: 행 순서가 피처 파일과 다름")
    base = sp.load_npz(os.path.join(BASE_DIR, f"{name}.npz")).tocsc()[:, : 2 * N_FEATURES]
    X = sp.hstack([base, sp.vstack(blocks)], format="csr", dtype="float32")
    tmp = os.path.join(OUT_DIR, f"{name}.tmp.npz")
    sp.save_npz(tmp, X, compressed=True)
    os.replace(tmp, os.path.join(OUT_DIR, f"{name}.npz"))
    print(f"{name}: {len(shas):,}행, text 블록 행당 {blocks and sum(b.nnz for b in blocks) / len(shas):,.0f}개, "
          f"{time.time() - start:.0f}초", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="본문 전체 text 블록 벡터 만들기")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    names = args.only or sorted(n.removesuffix(".parquet") for n in os.listdir(FEATURE_DIR) if n.endswith(".parquet"))
    todo = [n for n in names if not os.path.exists(os.path.join(OUT_DIR, f"{n}.npz"))]
    print(f"대상 {len(names)}개 중 남은 {len(todo)}개", flush=True)
    with Pool(args.workers) as pool:
        for name in todo:
            process(name, pool)


if __name__ == "__main__":
    main()
