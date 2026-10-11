"""1단계에서 모은 URL 목록으로 피처 DataFrame 을 만든다. (v2 2단계)

정상과 피싱을 같은 코드·같은 실행으로 계산한다. 2024년에는 두 클래스의 피처가
서로 다른 규칙으로 저장되어 피처 하나로 100% 구분되는 문제가 있었다.

출력
    data/v2/interim/features_<태그>.csv    URL 포함 (로컬 전용, 공개하지 않음)
    data/v2/processed/features_<태그>.csv  공개용. URL 대신 SHA-256 해시를 넣고, URL 을 복원할 수 있는
                                           칼럼(Tokenized_url, registered_domain)도 뺀다.

중간에 끊겨도 같은 명령을 다시 실행하면 이미 계산한 URL 은 건너뛴다.
Docker 컨테이너 안에서 실행한다 (docker/run.sh src/v2/build_features.py ...).

예시:
    python src/v2/build_features.py --tag pilot --phishing-sample 2000 --legit-sample 2000
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import tldextract

import features_paper as fp
from fetch import BROWSER_UA, fetch_page, get_certificate, lookup_whois

INTERIM_DIR = os.path.join("data", "v2", "interim")
PROCESSED_DIR = os.path.join("data", "v2", "processed")
CHUNK = 1000
# 공개용 파일에서 뺄 칼럼 (URL 원문을 복원할 수 있는 정보)
PRIVATE_COLUMNS = ["url", "Tokenized_url", "registered_domain"]

# 내장된 공개 접미사 목록만 사용 (실행 중 인터넷에서 받지 않음 → 결과 재현 가능)
_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
_local = threading.local()
# WHOIS 서버(등록 기관)는 요청 제한이 엄격하므로 동시에 몇 개만 조회한다
WHOIS_CONCURRENCY = 8
_whois_slots = threading.Semaphore(WHOIS_CONCURRENCY)


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers["User-Agent"] = BROWSER_UA
        _local.session = s
    return _local.session


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def process(row: dict) -> dict:
    """URL 하나를 처리한다. 예상하지 못한 오류가 나도 전체 실행을 멈추지 않고 그 URL 만 error 로 남긴다."""
    try:
        return _process(row)
    except Exception as e:  # noqa: BLE001 — 수십만 개를 도는 동안 한 URL 때문에 멈추지 않기 위함
        out = {"url": row["url"], "url_sha256": url_hash(row["url"]), "Label": row["Label"],
               "source": row["source"], "collected_at": row["collected_at"],
               "fetch_status": f"crash:{e.__class__.__name__}"}
        out.update(fp.compute(row["url"], None, None, None))
        return out


def _process(row: dict) -> dict:
    """URL 하나: 페이지·인증서·WHOIS 를 조회하고 피처를 계산한다."""
    url = row["url"]
    page = fetch_page(url, _session())
    cert = get_certificate(url)
    ext = _extract(url)
    registered = ext.top_domain_under_public_suffix
    whois = None
    if registered and fp.ip_like(url) == fp.LEGIT:
        with _whois_slots:
            whois = lookup_whois(registered)

    out = {"url": url, "url_sha256": url_hash(url), "Label": row["Label"],
           "source": row["source"], "collected_at": row["collected_at"],
           "registered_domain": registered, "fetch_status": page["status"],
           "http_status": page["http_status"], "n_redirects": page["n_redirects"],
           "cert_issuer": cert["issuer"] if cert else None,
           "cert_verified": cert["verified"] if cert else None,
           "whois_ok": whois is not None,
           "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.update(fp.compute(url, page if page["status"] == "ok" else _partial(page), cert, whois))
    return out


def _partial(page: dict) -> dict:
    """HTML 은 못 받았어도 접속·리디렉션 정보는 있으면 쓴다 (본문 기반 피처만 빈 값)."""
    return {"final_url": page["final_url"], "n_redirects": page["n_redirects"]}


def latest(pattern: str) -> str:
    files = sorted(glob.glob(os.path.join(INTERIM_DIR, pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]


def load_inputs(phishing_sample: int, legit_sample: int, seed: int) -> pd.DataFrame:
    """피싱은 DNS 사전 확인을 거친 목록(phishing_resolvable_*)이 있으면 그것을, 없으면 원래 목록을 쓴다."""
    try:
        phish_path = latest("phishing_resolvable_????-??-??.csv")
    except FileNotFoundError:
        phish_path = latest("phishing_urls_????-??-??.csv")
    print(f"피싱 입력: {phish_path}")
    phish = pd.read_csv(phish_path, low_memory=False)
    legit = pd.read_csv(latest("legit_urls_*_d*.csv"))
    if phishing_sample:
        phish = phish.sample(n=min(phishing_sample, len(phish)), random_state=seed)
    if legit_sample:
        legit = legit.sample(n=min(legit_sample, len(legit)), random_state=seed)
    cols = ["url", "Label", "source", "collected_at"]
    df = pd.concat([phish[cols], legit[cols]], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)  # 클래스를 섞어 같은 시간대에 조회


def main() -> None:
    parser = argparse.ArgumentParser(description="URL 목록 → 피처 DataFrame (정상·피싱 동시 계산)")
    parser.add_argument("--tag", required=True, help="출력 파일 이름에 붙일 태그 (예: pilot)")
    parser.add_argument("--phishing-sample", type=int, default=0)
    parser.add_argument("--legit-sample", type=int, default=0)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    interim_path = os.path.join(INTERIM_DIR, f"features_{args.tag}.csv")
    processed_path = os.path.join(PROCESSED_DIR, f"features_{args.tag}.csv")

    todo = load_inputs(args.phishing_sample, args.legit_sample, args.seed)
    if os.path.exists(interim_path):
        done = set(pd.read_csv(interim_path, usecols=["url"])["url"])
        todo = todo[~todo["url"].isin(done)]
        print(f"이어서 실행: 이미 {len(done):,}개 완료, 남은 {len(todo):,}개")
    print(f"대상 {len(todo):,}개 (피싱 {int(todo['Label'].sum()):,} / 정상 {int((todo['Label'] == 0).sum()):,})")

    start = time.time()
    rows = todo.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i in range(0, len(rows), CHUNK):
            chunk = list(pool.map(process, rows[i:i + CHUNK]))
            out = pd.DataFrame(chunk)
            out.to_csv(interim_path, mode="a", index=False, header=not os.path.exists(interim_path))
            done_n = i + len(chunk)
            rate = done_n / (time.time() - start)
            print(f"  {done_n:,}/{len(rows):,}  ({rate:.1f}개/초)  "
                  f"접속 성공률 {(out['fetch_status'] == 'ok').mean():.2f}", flush=True)

    full = pd.read_csv(interim_path)
    full.drop(columns=PRIVATE_COLUMNS).to_csv(processed_path, index=False)
    print(f"저장: {interim_path} (로컬 전용), {processed_path} (공개용, URL 제외)")
    print(full.groupby("Label")["fetch_status"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
