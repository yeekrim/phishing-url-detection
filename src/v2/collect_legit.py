"""정상 URL 목록을 수집한다. (v2 1단계)

2024년 데이터는 정상 URL 이 전부 사이트 첫 화면 주소(경로·쿼리 없음)여서,
모델이 "경로가 있으면 피싱"이라는 지름길을 배울 수 있었다.
v2 에서는 Common Crawl 정적 색인(Parquet)에서 실제로 수집된 페이지 URL 을 읽고,
Tranco 상위 100만 도메인에 속한 것만 남긴 뒤 도메인마다 첫 화면 1개 + 안쪽 페이지 여러 개를 뽑는다.
(안쪽 페이지만 뽑으면 반대로 "첫 화면이면 피싱"이라는 지름길이 생긴다.)

수집 원칙
    - 색인 파일은 주소(국가 도메인) 순으로 나뉘어 있으므로 300개 중 일정 간격으로 고른다.
    - Common Crawl 권장 방식대로 파일을 통째로 한 번에 받고(원격 부분 읽기는 503 이 잦음),
      한 번에 한 파일씩, 파일 사이에 쉬며, User-Agent 를 밝힌다.
    - 디스크를 아끼기 위해 파일에서 필요한 URL 만 뽑고 원본 파일은 바로 지운다.
    - 순위 구간(1~1천 / ~1만 / ~10만 / ~100만)별로 도메인을 고르게 뽑되,
      후보가 모자란 구간은 있는 만큼 쓰고 남은 수를 다른 구간에 나눈다.

결과는 data/v2/interim/legit_urls_<날짜>.csv 로, 선택 전 후보 전체는 legit_pool_<날짜>.csv 로 저장한다.

예시:
    python src/v2/collect_legit.py --files 2 --domains 40        # 시험 수집
    python src/v2/collect_legit.py --files 16 --domains 9400     # 본 수집 (약 7만5천 개)
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import random
import time
import zipfile
from datetime import date

import duckdb
import pandas as pd
import requests

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
CC_BASE = "https://data.commoncrawl.org"
CC_CRAWL = "CC-MAIN-2026-39"
USER_AGENT = "phishing-url-detection-v2/0.1 (+https://github.com/yeekrim/phishing-url-detection)"
HEADERS = {"User-Agent": USER_AGENT}

RAW_DIR = os.path.join("data", "v2", "raw", "legit")
INTERIM_DIR = os.path.join("data", "v2", "interim")

RANK_BINS = [(1, 1_000), (1_001, 10_000), (10_001, 100_000), (100_001, 1_000_000)]
# 파일 하나에서 도메인당 남겨 둘 안쪽 페이지 후보 수 (메모리 절약용, 최종 선택은 나중에)
CANDIDATES_PER_DOMAIN = 30
PAUSE_BETWEEN_FILES = 5


def load_tranco(raw_dir: str) -> pd.DataFrame:
    """Tranco 상위 100만 도메인 목록 (rank, domain)."""
    path = os.path.join(raw_dir, "tranco_top1m.csv")
    if not os.path.exists(path):
        print(f"  Tranco 목록 내려받는 중: {TRANCO_URL}")
        resp = requests.get(TRANCO_URL, headers=HEADERS, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open(z.namelist()[0]) as f, open(path, "wb") as out:
                out.write(f.read())
    return pd.read_csv(path, header=None, names=["rank", "domain"])


def cc_parquet_files(crawl: str) -> list[str]:
    """해당 수집본의 색인 Parquet 파일 주소 목록 (주소 순서대로)."""
    resp = requests.get(f"{CC_BASE}/crawl-data/{crawl}/cc-index-table.paths.gz",
                        headers=HEADERS, timeout=60)
    resp.raise_for_status()
    paths = gzip.decompress(resp.content).decode().split()
    return [f"{CC_BASE}/{p}" for p in sorted(paths) if "subset=warc" in p]


def evenly_spaced(files: list[str], n: int, rng: random.Random) -> list[str]:
    """전체 파일에서 일정 간격으로 n개를 고른다 (시작 위치만 무작위)."""
    step = len(files) / n
    offset = rng.random() * step
    return [files[int(offset + i * step)] for i in range(n)]


def download_parquet(file_url: str, out_dir: str, retries: int = 6) -> str:
    """색인 파일을 통째로 내려받는다. 실패하면 간격을 늘려 가며 다시 시도한다."""
    path = os.path.join(out_dir, file_url.rsplit("/", 1)[-1])
    if os.path.exists(path):
        return path
    for attempt in range(retries):
        try:
            with requests.get(file_url, headers=HEADERS, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(path + ".part", "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            os.replace(path + ".part", path)
            return path
        except requests.RequestException as e:
            wait = 20 * (2 ** attempt)
            print(f"    내려받기 실패({e.__class__.__name__}), {wait}초 후 재시도")
            time.sleep(wait)
    raise RuntimeError(f"내려받기 실패: {file_url}")


def extract_candidates(con: duckdb.DuckDBPyConnection, path: str,
                       tranco: pd.DataFrame, seed: int) -> pd.DataFrame:
    """색인 파일에서 Tranco 도메인의 정상 응답(200) HTML 페이지만 골라,
    도메인마다 첫 화면 전부 + 안쪽 페이지 무작위 일부를 후보로 남긴다."""
    con.register("tranco", tranco)
    return con.execute(f"""
        WITH pages AS (
            SELECT p.url, p.url_host_registered_domain AS domain, t.rank,
                   (coalesce(p.url_path, '/') = '/' AND p.url_query IS NULL) AS is_root
            FROM read_parquet('{path}') p
            JOIN tranco t ON t.domain = p.url_host_registered_domain
            WHERE p.fetch_status = 200 AND p.content_mime_detected = 'text/html'
        ),
        ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY domain, is_root ORDER BY hash(url || '{seed}')) AS rn
            FROM pages
        )
        SELECT url, domain, rank, is_root FROM ranked
        WHERE is_root OR rn <= {CANDIDATES_PER_DOMAIN}
    """).df()


def allocate(counts: dict[int, int], total: int) -> dict[int, int]:
    """구간별 후보 도메인 수(counts)에 맞춰 total 을 고르게 나눈다.
    후보가 모자란 구간은 전부 쓰고, 남은 수는 나머지 구간에 다시 나눈다."""
    alloc = {k: 0 for k in counts}
    remaining, open_bins = total, [k for k in counts if counts[k] > 0]
    while remaining > 0 and open_bins:
        share = max(1, remaining // len(open_bins))
        for k in list(open_bins):
            take = min(share, counts[k] - alloc[k], remaining)
            alloc[k] += take
            remaining -= take
            if alloc[k] >= counts[k]:
                open_bins.remove(k)
            if remaining == 0:
                break
    return alloc


def select(pool: pd.DataFrame, n_domains: int, per_domain: int, rng: random.Random) -> pd.DataFrame:
    """순위 구간별로 도메인을 고르고, 도메인마다 첫 화면 1개 + 안쪽 페이지 (per_domain-1)개를 뽑는다."""
    edges = [RANK_BINS[0][0] - 1] + [hi for _, hi in RANK_BINS]
    pool = pool.assign(bin=pd.cut(pool["rank"], edges, labels=False))
    domains_by_bin = {int(b): sorted(g["domain"].unique()) for b, g in pool.groupby("bin")}
    alloc = allocate({b: len(d) for b, d in domains_by_bin.items()}, n_domains)

    by_domain = dict(tuple(pool.groupby("domain")))
    picked = []
    for b, n in sorted(alloc.items()):
        lo, hi = RANK_BINS[b]
        chosen = rng.sample(domains_by_bin[b], n)
        print(f"  순위 {lo:,}~{hi:,}: 후보 도메인 {len(domains_by_bin[b]):,}개 중 {n:,}개 선택")
        for d in chosen:
            rows = by_domain[d]
            roots, inner = rows[rows["is_root"]], rows[~rows["is_root"]]
            take_root = roots.sample(n=min(1, len(roots)), random_state=rng.randrange(1 << 30))
            take_inner = inner.sample(n=min(per_domain - len(take_root), len(inner)),
                                      random_state=rng.randrange(1 << 30))
            picked.append(pd.concat([take_root, take_inner]))
    return pd.concat(picked, ignore_index=True).drop(columns="bin")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tranco + Common Crawl 정적 색인 기반 정상 URL 수집")
    parser.add_argument("--files", type=int, default=2, help="읽을 색인 파일 수 (전체 300개 중 일정 간격)")
    parser.add_argument("--domains", type=int, default=40, help="뽑을 도메인 수")
    parser.add_argument("--per-domain", type=int, default=8, help="도메인당 URL 수 (첫 화면 1 + 안쪽 페이지)")
    parser.add_argument("--keep-files", action="store_true", help="내려받은 색인 파일을 지우지 않음")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    today = date.today().isoformat()
    cc_dir = os.path.join(RAW_DIR, "cc")
    os.makedirs(cc_dir, exist_ok=True)
    os.makedirs(INTERIM_DIR, exist_ok=True)
    rng = random.Random(args.seed)

    tranco = load_tranco(RAW_DIR)
    files = evenly_spaced(cc_parquet_files(CC_CRAWL), args.files, rng)

    con = duckdb.connect()
    frames = []
    for i, f in enumerate(files, start=1):
        path = download_parquet(f, cc_dir)
        df = extract_candidates(con, path, tranco, args.seed)
        if not args.keep_files:
            os.remove(path)
        print(f"  색인 파일 {i}/{len(files)} ({f.rsplit('/', 1)[-1][:10]}): "
              f"후보 {len(df):,}개, 도메인 {df['domain'].nunique():,}개, 첫 화면 {int(df['is_root'].sum()):,}개",
              flush=True)
        frames.append(df)
        if i < len(files):
            time.sleep(PAUSE_BETWEEN_FILES)

    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset="url")
    # 후보 전체도 저장해 둔다. 3단계에서 피싱 쪽 경로 깊이 분포에 맞춰 다시 뽑을 때 쓴다.
    pool_path = os.path.join(INTERIM_DIR, f"legit_pool_{today}_f{args.files}.csv")
    pool.to_csv(pool_path, index=False)
    print(f"후보 저장: {pool_path} ({len(pool):,}개, 도메인 {pool['domain'].nunique():,}개)")
    out = select(pool, args.domains, args.per_domain, rng).rename(columns={"rank": "tranco_rank"})
    out["source"] = "commoncrawl"
    out["collected_at"] = today
    out["cc_crawl"] = CC_CRAWL
    out["Label"] = 0

    path = os.path.join(INTERIM_DIR, f"legit_urls_{today}_d{args.domains}.csv")
    out.to_csv(path, index=False)
    print(f"저장: {path} ({len(out):,}개, 도메인 {out['domain'].nunique():,}개, "
          f"첫 화면 비율 {out['is_root'].mean():.2f})")


if __name__ == "__main__":
    main()
