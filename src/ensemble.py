"""LightGBM(머신러닝) + CNN(딥러닝) 앙상블 평가.

두 모델의 예측 확률을 평균(soft voting)하여 최종 피싱 확률을 계산한다.

    ensemble_prob = (cnn_prob + lgbm_prob) / 2
    예측 = 1 if ensemble_prob >= 0.5 else 0

발표자료에서는 여기에 '의심 범위'를 추가하여 확률대별로
정상 / 의심 / 피싱 세 구간으로 시각화했다.

이 스크립트는 train_lgbm.py 로 저장한 joblib 모델과
train_cnn.py 로 저장한 keras 모델을 불러와 검증 데이터에 대해 앙상블 성능을 낸다.

예시:
    python src/ensemble.py \
        --data data/final/valid_data.csv \
        --lgbm lgbm_model.joblib \
        --cnn cnn_best.keras
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from features import tokenize_url
from train_cnn import MAX_LEN
from train_lgbm import build_matrix

# 발표자료의 '의심 범위' 구간
SUSPICIOUS_LOW = 0.35
SUSPICIOUS_HIGH = 0.65


def classify_with_suspicion(prob: float) -> str:
    """확률을 정상/의심/피싱 세 구간으로 분류(발표자료 시각화 기준)."""
    if prob < SUSPICIOUS_LOW:
        return "정상"
    if prob < SUSPICIOUS_HIGH:
        return "의심"
    return "피싱"


def main() -> None:
    parser = argparse.ArgumentParser(description="LGBM + CNN 앙상블 평가")
    parser.add_argument("--data", required=True, help="피처 CSV (url, ..., Tokenized_url, Label)")
    parser.add_argument("--lgbm", required=True, help="train_lgbm.py 로 저장한 joblib")
    parser.add_argument("--cnn", required=True, help="train_cnn.py 로 저장한 keras 모델")
    parser.add_argument("--out", default=None, help="예측 결과 CSV 저장 경로(선택)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["url"] = df["url"].astype(str)
    y = df["Label"].values

    # --- LightGBM 예측 ---
    bundle = joblib.load(args.lgbm)
    lgbm, vectorizer = bundle["model"], bundle["vectorizer"]
    X_lgbm = build_matrix(df, vectorizer, fit=False)
    lgbm_prob = lgbm.predict_proba(X_lgbm)[:, 1]

    # --- CNN 예측 ---
    cnn = load_model(args.cnn)
    token_strings = [" ".join(tokenize_url(u)) for u in df["url"]]
    # 학습 때 쓴 토크나이저가 없으면 데이터 기준으로 재구성(근사)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(token_strings)
    X_cnn = pad_sequences(tokenizer.texts_to_sequences(token_strings), maxlen=MAX_LEN)
    cnn_prob = cnn.predict(X_cnn).flatten()

    # --- 앙상블 (soft voting) ---
    ensemble_prob = (cnn_prob + lgbm_prob) / 2
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    acc = float(np.mean(ensemble_pred == y))
    print("=== 앙상블 성능 ===")
    print(f"LightGBM accuracy : {np.mean((lgbm_prob >= 0.5).astype(int) == y):.4f}")
    print(f"CNN accuracy      : {np.mean((cnn_prob >= 0.5).astype(int) == y):.4f}")
    print(f"Ensemble accuracy : {acc:.4f}")

    if args.out:
        result = pd.DataFrame({
            "url": df["url"],
            "Label": y,
            "lgbm_prob": lgbm_prob,
            "cnn_prob": cnn_prob,
            "ensemble_prob": ensemble_prob,
            "prediction": ensemble_pred,
            "verdict": [classify_with_suspicion(p) for p in ensemble_prob],
        })
        result.to_csv(args.out, index=False)
        print(f"결과 저장: {args.out}")


if __name__ == "__main__":
    main()
