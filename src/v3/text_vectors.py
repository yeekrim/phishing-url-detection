"""텍스트 필드 → 해시 n-gram 희소 행렬. 파일(shard) 단위로 저장한다. (v3 2단계)

근거: PhreshPhish 의 기준 모델(Linear, FFN)은 URL·HTML 의 n-gram + 손으로 만든 피처를 쓴다.
HashingVectorizer 는 어휘 사전이 필요 없어(상태 없음) 파일마다 따로 변환해도 같은 열을 쓴다.
66만 행 전체를 메모리에 올릴 수 없으므로(Mac 8GB) 파일 단위로 저장해 두고 학습 때 하나씩 읽는다.

블록
    url   : 쿼리를 지운 URL 의 글자 3~5-gram (.jp 접미사는 .com 으로 가림 — features.mask_jp)
    html  : HTML 앞부분 5,000자의 단어 1~2-gram (태그·속성 이름 포함)
    text  : 페이지 제목 + 본문 앞 2,000자의 단어 1~2-gram

출력: data/v3/interim/vectors/<파일>.npz  (url, html, text 블록을 옆으로 붙인 CSR, float32)
"""

from __future__ import annotations

import glob
import os

import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

import features as F

FEATURE_DIR = os.path.join("data", "v3", "interim", "features")
# V3_VECTORS 로 다른 벡터 폴더(예: 본문 전체를 쓴 vectors_full)를 골라 학습할 수 있다.
VECTOR_DIR = os.environ.get("V3_VECTORS", os.path.join("data", "v3", "interim", "vectors"))
N_FEATURES = 2 ** 20
BLOCKS = ("url", "html", "text")

_common = dict(n_features=N_FEATURES, alternate_sign=False, binary=True, norm="l2", dtype="float32")
VECTORIZERS = {
    "url": HashingVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, **_common),
    "html": HashingVectorizer(token_pattern=r"[A-Za-z][A-Za-z0-9_\-]{1,30}", ngram_range=(1, 2), **_common),
    "text": HashingVectorizer(token_pattern=r"(?u)\b\w{2,30}\b", ngram_range=(1, 2), **_common),
}


def block_slice(block: str) -> slice:
    i = BLOCKS.index(block)
    return slice(i * N_FEATURES, (i + 1) * N_FEATURES)


def vectorize(df: pd.DataFrame) -> sp.csr_matrix:
    parts = [
        VECTORIZERS["url"].transform(df["url_clean"].fillna("").map(F.mask_jp)),
        VECTORIZERS["html"].transform(df["html_prefix"].fillna("")),
        VECTORIZERS["text"].transform((df["title"].fillna("") + " " + df["text"].fillna(""))),
    ]
    return sp.hstack(parts, format="csr", dtype="float32")


def main() -> None:
    os.makedirs(VECTOR_DIR, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet"))):
        name = os.path.basename(path).removesuffix(".parquet")
        out = os.path.join(VECTOR_DIR, f"{name}.npz")
        if os.path.exists(out):
            continue
        df = pd.read_parquet(path, columns=["url_clean", "html_prefix", "title", "text"])
        sp.save_npz(out, vectorize(df), compressed=True)
        print(f"{name}: {len(df):,}행", flush=True)


if __name__ == "__main__":
    main()
