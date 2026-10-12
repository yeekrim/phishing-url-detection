"""피처 DataFrame 의 결측 처리 → 학습용 데이터셋. (v2 3단계)

결측은 채우지 않는다. 결측 비율이 클래스마다 크게 달라서(결측 1개 이상인 행: 정상 22% / 피싱 86%)
값을 채우면 "채워진 값 = 피싱" 이라는 지름길이 생기기 때문이다. 대신 피처를 줄이고 빈 행을 뺀다.

    - Mouseover 삭제: 페이지를 받은 134,734개 중 -1 은 32개뿐(정상 28 / 피싱 4).
      window.status 는 요즘 브라우저가 무시해서 피싱에서도 쓰이지 않는다 → 값이 사실상 하나.
    - Hyperlinks 는 주 데이터셋에서 뺀다. 결측의 대부분(피싱 81.5%)이 페이지를 못 받은 경우라
      주 데이터셋에 넣으면 행이 크게 줄어든다. 대신 페이지를 받은 행만 따로 모아 별도 평가한다.
    - IP 주소 URL 은 WHOIS 를 조회할 도메인이 없으므로 Domain_Age·Domain_end 를 NOT_APPLICABLE(2)로 둔다.
      비워 두면 빈 행 제외 단계에서 IP 주소 URL(피싱 9,221개)이 전부 빠져 IP_LIKE 를 평가할 수 없다.

출력
    data/v2/interim/dataset_main.csv          주 데이터셋: 피처 12개, 결측 없는 행 (URL 포함, 로컬 전용)
    data/v2/interim/dataset_hyperlinks.csv    Hyperlinks 평가용: 피처 13개, 결측 없는 행 (로컬 전용)
    data/v2/processed/ 같은 이름               공개용 (URL·Tokenized_url·registered_domain 제외)

예시:
    python src/v2/build_dataset.py --tag full
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

import features_paper as fp
from build_features import INTERIM_DIR, PRIVATE_COLUMNS, PROCESSED_DIR

DROPPED = ["Mouseover"]
SEPARATE = ["Hyperlinks"]
MAIN_FEATURES = [c for c in fp.FEATURE_COLUMNS if c not in DROPPED + SEPARATE]
HYPERLINK_FEATURES = MAIN_FEATURES + SEPARATE
WHOIS_FEATURES = ["Domain_Age", "Domain_end"]
NOT_APPLICABLE = 2  # 결측(-)이 아니라 "해당 없음". 값 체계(1 / 0 / -1)와 겹치지 않는 값
META = ["url", "url_sha256", "Label", "source", "collected_at", "registered_domain",
        "fetch_status", "checked_at", "Tokenized_url"]


def mark_not_applicable(df: pd.DataFrame) -> pd.DataFrame:
    """IP 주소 URL 의 WHOIS 피처를 '해당 없음' 으로 표시한다."""
    df = df.copy()
    is_ip = df["IP_LIKE"] == fp.PHISHING
    for col in WHOIS_FEATURES:
        df.loc[is_ip & df[col].isna(), col] = NOT_APPLICABLE
    print(f"IP 주소 URL {int(is_ip.sum()):,}개의 WHOIS 피처를 해당 없음({NOT_APPLICABLE})으로 표시")
    return df


def subset(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """해당 피처가 모두 있는 행만, 메타 칼럼 + 피처 칼럼으로."""
    return df.dropna(subset=features)[META + features].reset_index(drop=True)


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(os.path.join(INTERIM_DIR, f"{name}.csv"), index=False)
    df.drop(columns=PRIVATE_COLUMNS).to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False)
    counts = df["Label"].value_counts().sort_index()
    print(f"{name}: {len(df):,}행 (정상 {counts.get(0, 0):,} / 피싱 {counts.get(1, 0):,}), "
          f"등록 도메인 {df['registered_domain'].nunique():,}개")


def main() -> None:
    parser = argparse.ArgumentParser(description="결측 처리: 피처 삭제 + 빈 행 제외 (값을 채우지 않음)")
    parser.add_argument("--tag", required=True, help="입력 features_<태그>.csv")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(INTERIM_DIR, f"features_{args.tag}.csv"), low_memory=False)
    print(f"입력 {len(df):,}행, 삭제 피처 {DROPPED}, 별도 평가 {SEPARATE}")
    df = mark_not_applicable(df)
    save(subset(df, MAIN_FEATURES), "dataset_main")
    save(subset(df, HYPERLINK_FEATURES), "dataset_hyperlinks")


if __name__ == "__main__":
    main()
