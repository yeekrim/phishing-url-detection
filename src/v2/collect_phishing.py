"""공개 피싱 피드에서 피싱 URL 목록을 수집한다. (v2 1단계)

출처
    phishtank         PhishTank 검증 완료·온라인 목록 (verified_online.csv)
    phishing_database Phishing.Database 의 ACTIVE 링크 목록
    openphish         OpenPhish 무료 피드

원본 파일은 data/v2/raw/phishing/<날짜>/ 에 저장하고(공개하지 않음),
정리한 목록은 data/v2/interim/phishing_urls_<날짜>.csv 로 저장한다.
이 스크립트는 목록 파일만 내려받으며, 목록에 있는 피싱 사이트에는 접속하지 않는다.

예시:
    python src/v2/collect_phishing.py                 # 전체 수집
    python src/v2/collect_phishing.py --sample 300    # 출처별 300개만 (시험 수집)
"""

from __future__ import annotations

import argparse
import os
from datetime import date

import pandas as pd
import requests

SOURCES = {
    "phishtank": "https://data.phishtank.com/data/online-valid.csv",
    "phishing_database": "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "openphish": "https://openphish.com/feed.txt",
}

RAW_DIR = os.path.join("data", "v2", "raw", "phishing")
INTERIM_DIR = os.path.join("data", "v2", "interim")


def download(name: str, url: str, out_dir: str) -> str:
    """출처 파일을 내려받아 경로를 반환한다. 같은 날 이미 받았으면 다시 받지 않는다."""
    ext = ".csv" if url.endswith(".csv") else ".txt"
    path = os.path.join(out_dir, name + ext)
    if os.path.exists(path):
        print(f"  [{name}] 오늘 받은 파일 재사용: {path}")
        return path
    print(f"  [{name}] 내려받는 중: {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def parse(name: str, path: str) -> pd.DataFrame:
    """출처 파일을 (url, source, source_time) 표로 바꾼다."""
    if name == "phishtank":
        df = pd.read_csv(path, usecols=["url", "submission_time"])
        df = df.rename(columns={"submission_time": "source_time"})
    else:
        with open(path, encoding="utf-8", errors="ignore") as f:
            urls = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({"url": urls, "source_time": None})
    df["source"] = name
    return df[["url", "source", "source_time"]]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """http(s) URL 만 남기고, 앞뒤 공백 제거 후 같은 URL 은 하나만 남긴다."""
    df = df.copy()
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.match(r"^https?://", case=False)]
    return df.drop_duplicates(subset="url").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="공개 피드에서 피싱 URL 목록 수집")
    parser.add_argument("--sample", type=int, default=0,
                        help="출처별로 무작위 N개만 남김 (시험 수집용, 0이면 전체)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    today = date.today().isoformat()
    raw_dir = os.path.join(RAW_DIR, today)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(INTERIM_DIR, exist_ok=True)

    frames = []
    for name, url in SOURCES.items():
        try:
            df = parse(name, download(name, url, raw_dir))
        except Exception as e:  # 한 출처가 실패해도 나머지는 계속 수집
            print(f"  [{name}] 실패: {e}")
            continue
        df = clean(df)
        if args.sample and len(df) > args.sample:
            df = df.sample(n=args.sample, random_state=args.seed)
        print(f"  [{name}] {len(df):,}개")
        frames.append(df)

    out = clean(pd.concat(frames, ignore_index=True))
    out["collected_at"] = today
    out["Label"] = 1

    suffix = f"_sample{args.sample}" if args.sample else ""
    path = os.path.join(INTERIM_DIR, f"phishing_urls_{today}{suffix}.csv")
    out.to_csv(path, index=False)
    print(f"저장: {path} ({len(out):,}개, 출처 간 중복 제거 후)")
    print(out["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
