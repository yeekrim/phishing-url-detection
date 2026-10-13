"""PhreshPhish 파일을 하나씩 받아 피처만 뽑고 원본을 지운다. (v3 1단계)

PhreshPhish (Dalton 외, 2025, CC BY 4.0) 는 HTML 원본을 포함해 전체 36.6GB, 파일 60개(각 0.05~0.57GB)다.
디스크에 전체를 둘 수 없으므로 파일 단위로
    받기 → 500행씩 읽으며 피처 계산(여러 프로세스) → 결과 저장 → 원본 삭제
를 반복한다. 결과 파일이 이미 있는 파일은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어진다.

HTML 은 신뢰할 수 없는 데이터라 별도 폴더(data/v3/raw/)에 받고, 텍스트 파싱만 한다.

출력: data/v3/interim/features/<train|test>-NNN.parquet
    sha256, url, label(1 = 피싱), target, date, lang + 숫자 피처 + 텍스트 필드(title, text, html_prefix)

예시:
    python src/v3/extract_phreshphish.py --only train-005      # 파일 하나 시험
    python src/v3/extract_phreshphish.py                       # 전체
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from multiprocessing import Pool

import pandas as pd
import pyarrow.parquet as pq
import requests

import features

REPO = "https://huggingface.co/datasets/phreshphish/phreshphish"
RAW_DIR = os.path.join("data", "v3", "raw")
OUT_DIR = os.path.join("data", "v3", "interim", "features")
USER_AGENT = "phishing-url-detection-v3/0.1 (+https://github.com/yeekrim/phishing-url-detection)"
BATCH = 500
META = ["sha256", "url", "label", "target", "date", "lang"]


def shard_names() -> list[str]:
    resp = requests.get("https://huggingface.co/api/datasets/phreshphish/phreshphish/tree/main/data",
                        headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    names = [f["path"].split("/")[-1].removesuffix(".parquet") for f in resp.json() if f["type"] == "file"]
    return sorted(n for n in names if n.startswith(("train-", "test-")))


def download(name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.parquet")
    url = f"{REPO}/resolve/main/data/{name}.parquet"
    subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-delay", "20", "-A", USER_AGENT, "-o", path, url],
                   check=True)
    return path


def _row(rec: dict) -> dict:
    out = {k: rec[k] for k in META}
    out["label"] = 1 if rec["label"] == "phish" else 0
    try:
        out.update(features.compute(rec["url"], rec["html"]))
        out["parse_error"] = 0
    except Exception:  # noqa: BLE001 — 수십만 개 중 한 행 때문에 멈추지 않기 위함
        out.update(features.compute(rec["url"], None))
        out["parse_error"] = 1
    return out


def process(name: str, pool: Pool) -> None:
    out_path = os.path.join(OUT_DIR, f"{name}.parquet")
    start = time.time()
    raw = download(name)
    rows = []
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=BATCH, columns=META + ["html"]):
        rows.extend(pool.map(_row, batch.to_pylist(), chunksize=16))
    df = pd.DataFrame(rows)
    tmp = out_path + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, out_path)  # 다 쓴 뒤에만 완성 파일로 → 중간에 끊겨도 불완전한 결과가 남지 않음
    os.remove(raw)
    print(f"{name}: {len(df):,}행 (피싱 {int(df.label.sum()):,}), 파싱 실패 {int(df.parse_error.sum())}, "
          f"{time.time() - start:.0f}초", flush=True)


def refresh_url_features() -> None:
    """URL 피처 규칙만 바뀌었을 때: 원본을 다시 받지 않고 저장된 URL 로 URL 피처만 다시 계산한다."""
    for name in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, name)
        df = pd.read_parquet(path)
        clean = df["url"].map(features.strip_query)
        fresh = pd.DataFrame([features.url_features(u) for u in clean], index=df.index)
        df["url_clean"] = clean
        df[fresh.columns] = fresh
        df.to_parquet(path + ".tmp", index=False, compression="zstd")
        os.replace(path + ".tmp", path)
        print(f"{name}: URL 피처 다시 계산", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PhreshPhish 받기 → 피처 추출 → 원본 삭제 반복")
    parser.add_argument("--only", nargs="*", help="이 파일들만 (예: train-005)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh-url", action="store_true", help="저장된 결과의 URL 피처만 다시 계산")
    args = parser.parse_args()
    if args.refresh_url:
        refresh_url_features()
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    names = args.only or shard_names()
    todo = [n for n in names if not os.path.exists(os.path.join(OUT_DIR, f"{n}.parquet"))]
    print(f"대상 {len(names)}개 중 남은 {len(todo)}개", flush=True)
    with Pool(args.workers) as pool:
        for name in todo:
            process(name, pool)


if __name__ == "__main__":
    main()
