"""URL 목록으로부터 피처 데이터셋(CSV)을 생성한다.

입력 CSV 는 최소한 'url' 컬럼을 가져야 하며, 있으면 'Label' 컬럼도 그대로 보존한다.
출력은 features.extract_features 로 계산한 15개 피처 + Tokenized_url (+ Label) 이다.

예시:
    # 문자열 기반 피처만 (빠름, 네트워크 불필요)
    python src/build_dataset.py data/raw/normal_data.csv out.csv --no-network

    # 전체 피처 (WHOIS/HTTP 요청 포함, 느림)
    python src/build_dataset.py data/raw/normal_data.csv out.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from features import extract_features


def build(input_csv: str, output_csv: str, include_network: bool = True) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "url" not in df.columns:
        raise ValueError(f"입력 파일에 'url' 컬럼이 없습니다: {input_csv}")

    rows = []
    total = len(df)
    for i, url in enumerate(df["url"].astype(str), start=1):
        feats = extract_features(url, include_network=include_network)
        feats["url"] = url
        rows.append(feats)
        if i % 100 == 0 or i == total:
            print(f"  처리 중 {i}/{total}")

    out = pd.DataFrame(rows)
    if "Label" in df.columns:
        out["Label"] = df["Label"].values

    # url 을 맨 앞으로 정렬
    cols = ["url"] + [c for c in out.columns if c not in ("url", "Label")]
    if "Label" in out.columns:
        cols += ["Label"]
    out = out[cols]

    out.to_csv(output_csv, index=False)
    print(f"저장 완료: {output_csv}  (shape={out.shape})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="URL 목록에서 피처 데이터셋 생성")
    parser.add_argument("input_csv", help="'url' 컬럼을 가진 입력 CSV")
    parser.add_argument("output_csv", help="출력 CSV 경로")
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="WHOIS/HTTP 요청이 필요한 피처를 건너뛴다(문자열 기반 피처만).",
    )
    args = parser.parse_args()
    build(args.input_csv, args.output_csv, include_network=not args.no_network)


if __name__ == "__main__":
    main()
