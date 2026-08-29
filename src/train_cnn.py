"""딥러닝(1D-CNN) 피싱 탐지 모델 학습·평가.

URL 을 토큰화한 뒤 정수 인코딩 → 패딩 → Embedding + Conv1D 로 학습한다.
LightGBM 과 달리 원본 URL 문자열만 사용한다(수치형 피처 불필요).

입력 CSV 는 'url' 과 'Label' 컬럼을 가지면 된다.

예시:
    python src/train_cnn.py --data data/final/valid_data.csv --model-out cnn_best.keras
"""

from __future__ import annotations

import argparse

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import pandas as pd

from features import tokenize_url

# 하이퍼파라미터 (발표 기준)
MAX_LEN = 560
EMBEDDING_DIM = 32
DROPOUT_RATIO = 0.3
NUM_FILTERS = 32
KERNEL_SIZE = 5
EPOCHS = 10
BATCH_SIZE = 64


def build_model(vocab_size: int) -> Sequential:
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM))
    model.add(Dropout(DROPOUT_RATIO))
    model.add(Conv1D(NUM_FILTERS, KERNEL_SIZE, padding="valid", activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(DROPOUT_RATIO))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    return model


def encode(urls, tokenizer: Tokenizer):
    """URL 리스트 → 토큰화 → 정수 시퀀스 → 패딩."""
    token_strings = [" ".join(tokenize_url(str(u))) for u in urls]
    seqs = tokenizer.texts_to_sequences(token_strings)
    return pad_sequences(seqs, maxlen=MAX_LEN)


def main() -> None:
    parser = argparse.ArgumentParser(description="1D-CNN 피싱 탐지 모델 학습")
    parser.add_argument("--data", required=True, help="'url', 'Label' 컬럼을 가진 CSV")
    parser.add_argument("--model-out", default="cnn_best.keras", help="최적 모델 저장 경로")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["url"] = df["url"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        df["url"], df["Label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["Label"],
    )

    # 훈련 데이터로만 토크나이저 학습
    train_token_strings = [" ".join(tokenize_url(u)) for u in X_train]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_token_strings)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"단어 집합 크기: {vocab_size}")

    X_train_pad = pad_sequences(
        tokenizer.texts_to_sequences(train_token_strings), maxlen=MAX_LEN
    )
    X_test_pad = encode(X_test, tokenizer)

    model = build_model(vocab_size)
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
    mc = ModelCheckpoint(
        args.model_out, monitor="val_acc", mode="max", verbose=1, save_best_only=True
    )
    model.fit(
        X_train_pad, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.2, callbacks=[es, mc],
    )

    test_acc = model.evaluate(X_test_pad, y_test)[1]
    print(f"\n테스트 정확도: {test_acc:.4f}")
    print(f"최적 모델 저장: {args.model_out}")


if __name__ == "__main__":
    main()
