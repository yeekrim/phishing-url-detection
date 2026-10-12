"""학습/검증/평가 분할. (v2 4단계 준비)

1) 도메인 분할 (split_domain): 같은 사이트의 URL 이 학습과 평가에 동시에 들어가지 않게 한다.
   묶음 기준은 "공개 접미사 목록의 사설 항목까지 반영한 등록 도메인" 이다.
   예) a.github.io 와 b.github.io 는 운영자가 다르므로 다른 묶음. (일반 기준이면 둘 다 github.io 로 묶여
   피싱의 31% 가 상위 10개 호스팅 도메인에 몰린다.)
   묶음 이름의 SHA-256 값으로 70 / 15 / 15 를 정하므로 실행할 때마다 같은 결과가 나온다.

2) 시간 분할 (split_time): 신고 시각(source_time)이 있는 PhishTank 피싱만 대상.
   묶음마다 가장 이른 신고 시각을 기준으로, 오래된 80% 를 train, 최근 20% 를 test 로 둔다.
   정상은 시간 정보가 없어 도메인 분할을 그대로 쓴다. 다른 출처의 피싱은 none(사용 안 함).

출력
    data/v2/interim/split_<데이터셋>.csv    url_sha256, group, source_time, split_domain, split_time (로컬 전용)
    data/v2/processed/split_<데이터셋>.csv  group 을 뺀 공개용

예시:
    python src/v2/split_dataset.py dataset_main dataset_hyperlinks
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os

import pandas as pd
import tldextract

from build_features import INTERIM_DIR, PROCESSED_DIR

TRAIN, VAL = 70, 85  # 해시 % 100 < 70 → train, < 85 → val, 나머지 test
TIME_TEST_SHARE = 0.2

_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)


def group_of(url: str) -> str:
    ext = _extract(url)
    return ext.top_domain_under_public_suffix or ext.domain or url


def domain_split(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if bucket < TRAIN else "val" if bucket < VAL else "test"


def time_split(df: pd.DataFrame) -> pd.Series:
    out = pd.Series("none", index=df.index)
    legit = df["Label"] == 0
    out[legit] = df.loc[legit, "split_domain"].replace({"val": "train"})
    timed = (df["Label"] == 1) & df["source_time"].notna()
    first = df[timed].groupby("group")["source_time"].transform("min")
    cutoff = first.quantile(1 - TIME_TEST_SHARE)
    out[timed] = (first >= cutoff).map({True: "test", False: "train"})
    print(f"시간 분할 기준 시각: {cutoff}")
    return out


def source_times() -> pd.Series:
    path = sorted(glob.glob(os.path.join(INTERIM_DIR, "phishing_urls_????-??-??.csv")))[-1]
    p = pd.read_csv(path, usecols=["url", "source_time"], low_memory=False).drop_duplicates("url")
    return p.set_index("url")["source_time"]


def main() -> None:
    parser = argparse.ArgumentParser(description="도메인 분할 + 시간 분할")
    parser.add_argument("datasets", nargs="+", help="data/v2/interim/<이름>.csv")
    args = parser.parse_args()
    times = source_times()

    for name in args.datasets:
        df = pd.read_csv(os.path.join(INTERIM_DIR, f"{name}.csv"), low_memory=False)
        df["group"] = df["url"].map(group_of)
        df["source_time"] = pd.to_datetime(df["url"].map(times), utc=True)
        df["split_domain"] = df["group"].map(domain_split)
        df["split_time"] = time_split(df)

        cols = ["url_sha256", "group", "source_time", "split_domain", "split_time"]
        df[cols].to_csv(os.path.join(INTERIM_DIR, f"split_{name}.csv"), index=False)
        df[cols].drop(columns=["group"]).to_csv(os.path.join(PROCESSED_DIR, f"split_{name}.csv"), index=False)
        print(f"\n[{name}] 묶음 {df['group'].nunique():,}개")
        print(pd.crosstab(df["split_domain"], df["Label"], margins=True).to_string())
        print(pd.crosstab(df["split_time"], df["Label"]).to_string())


if __name__ == "__main__":
    main()
