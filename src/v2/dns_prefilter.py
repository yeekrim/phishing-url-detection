"""피싱 후보 URL 중 도메인이 이미 사라진 것을 DNS 조회만으로 먼저 거른다. (v2 2단계 준비)

시험 계산에서 피싱 후보의 약 절반이 DNS 실패(도메인 소멸)였다. 이런 URL 까지 페이지 접속·인증서·
WHOIS 를 시도하면 시간만 든다. 호스트 이름 단위로 A/AAAA 레코드만 조회하고, 응답이 있는 URL 만 남긴다.
DNS 조회는 피싱 서버에 접속하지 않는다 (DNS 서버에만 묻는다).

입력: data/v2/interim/phishing_urls_<날짜>.csv
출력: data/v2/interim/phishing_resolvable_<날짜>.csv  (dns_checked_at 칼럼 추가)

예시:
    python src/v2/dns_prefilter.py --sample 5000   # 시험
    python src/v2/dns_prefilter.py                 # 전체
"""

from __future__ import annotations

import argparse
import glob
import ipaddress
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import dns.exception
import dns.resolver
import pandas as pd

INTERIM_DIR = os.path.join("data", "v2", "interim")
TIMEOUT, LIFETIME = 3.0, 5.0

_resolver = dns.resolver.Resolver()
_resolver.timeout, _resolver.lifetime = TIMEOUT, LIFETIME


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def resolves(host: str) -> bool:
    """A 또는 AAAA 레코드가 하나라도 있으면 True. IP 주소 자체면 조회 없이 True."""
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    for rtype in ("A", "AAAA"):
        try:
            if _resolver.resolve(host, rtype):
                return True
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers,
                dns.exception.Timeout, dns.name.LabelTooLong, dns.name.EmptyLabel,
                dns.resolver.YXDOMAIN, ValueError):
            continue
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="DNS 조회로 사라진 피싱 도메인 거르기")
    parser.add_argument("--sample", type=int, default=0, help="무작위 N개만 (시험용)")
    parser.add_argument("--workers", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src = sorted(glob.glob(os.path.join(INTERIM_DIR, "phishing_urls_*.csv")))[-1]
    df = pd.read_csv(src, low_memory=False)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
    df["host"] = df["url"].map(host_of)
    hosts = df["host"].unique().tolist()
    print(f"입력 {len(df):,}개 URL, 고유 호스트 {len(hosts):,}개")

    start, alive = time.time(), {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (h, ok) in enumerate(zip(hosts, pool.map(resolves, hosts)), start=1):
            alive[h] = ok
            if i % 20000 == 0 or i == len(hosts):
                print(f"  {i:,}/{len(hosts):,} 호스트  ({i / (time.time() - start):.0f}개/초)", flush=True)

    df["resolvable"] = df["host"].map(alive)
    out = df[df["resolvable"]].drop(columns=["host", "resolvable"])
    out["dns_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    date = os.path.basename(src).replace("phishing_urls_", "").replace(".csv", "")
    suffix = f"_sample{args.sample}" if args.sample else ""
    path = os.path.join(INTERIM_DIR, f"phishing_resolvable_{date}{suffix}.csv")
    out.to_csv(path, index=False)
    print(f"저장: {path} ({len(out):,}개 / {len(df):,}개 = {len(out) / len(df):.2f})")
    print(df.groupby("source")["resolvable"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
