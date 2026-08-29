"""LightGBM · CNN · 앙상블을 동일한 train/test 분할에서 학습·평가하고
세 모델을 한 번에 비교한다.

기존 model_percentage.csv 는 CNN 예측 컬럼이 저장 과정에서 누락되어 있었다.
이 스크립트는 CNN 을 실제로 다시 학습시켜, 세 모델을 완전히 동일한 조건에서 비교한다.

- 데이터: data/final/valid_data.csv (url, 15개 피처, Tokenized_url, Label)
- 분할  : 계층적 8:2 (stratify=Label, random_state=0)
- LightGBM : Tokenized_url(CountVectorizer) + 수치형 피처
- CNN      : url 토큰화 → Embedding + Conv1D
- 앙상블   : 두 모델 예측 확률의 평균(soft voting)

산출물:
  - data/final/model_comparison.csv   : 테스트셋 예측 결과(세 모델 확률/예측)
  - docs/images/metrics_comparison.png : 세 모델 성능 비교
  - docs/images/confusion_matrix.png   : 앙상블 혼동행렬
  - docs/images/probability_distribution.png : 앙상블 예측 확률 분포

    python src/run_comparison.py
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
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

from features import tokenize_url

DATA = os.path.join("data", "final", "valid_data.csv")
OUT_CSV = os.path.join("data", "final", "model_comparison.csv")
IMG_DIR = os.path.join("docs", "images")

RANDOM_STATE = 0
DROP_COLS = ["url", "Label", "Tokenized_url"]

# LightGBM 최적 하이퍼파라미터 (발표 기준)
LGBM_PARAMS = {"learning_rate": 0.1, "max_depth": -1, "n_estimators": 100, "num_leaves": 31}

# CNN 하이퍼파라미터
MAX_LEN = 100        # 이 데이터의 URL 토큰 길이에 맞춰 축소 (원본 560)
EMBEDDING_DIM = 32
DROPOUT = 0.3
NUM_FILTERS = 32
KERNEL_SIZE = 5
EPOCHS = 5
BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------

def load_split():
    df = pd.read_csv(DATA)
    df["url"] = df["url"].astype(str)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_STATE, stratify=df["Label"]
    )
    return df, train_idx, test_idx


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def run_lgbm(df, train_idx, test_idx):
    vectorizer = CountVectorizer()
    text = vectorizer.fit_transform(df["Tokenized_url"].astype(str))
    other = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).values
    X = hstack([text, other]).tocsr()
    y = df["Label"].values

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X[train_idx], y[train_idx])
    prob = model.predict_proba(X[test_idx])[:, 1]
    joblib.dump({"model": model, "vectorizer": vectorizer}, "lgbm_model.joblib")
    return prob


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

def run_cnn(df, train_idx, test_idx):
    token_strings = [" ".join(tokenize_url(str(u))) for u in df["url"]]
    train_texts = [token_strings[i] for i in train_idx]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"[CNN] 단어 집합 크기: {vocab_size}")

    seqs = tokenizer.texts_to_sequences(token_strings)
    padded = pad_sequences(seqs, maxlen=MAX_LEN)
    y = df["Label"].values

    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM),
        Dropout(DROPOUT),
        Conv1D(NUM_FILTERS, KERNEL_SIZE, padding="valid", activation="relu"),
        GlobalMaxPooling1D(),
        Dropout(DROPOUT),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=2,
                       restore_best_weights=True)
    model.fit(padded[train_idx], y[train_idx], epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_split=0.2, callbacks=[es], verbose=2)
    model.save("cnn_best.keras")
    return model.predict(padded[test_idx], verbose=0).flatten()


# ---------------------------------------------------------------------------
# 평가 & 시각화
# ---------------------------------------------------------------------------

def metrics(true, pred):
    tp = int(((pred == 1) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    acc = (tp + tn) / len(true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "cm": np.array([[tn, fp], [fn, tp]])}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for _f in ("AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"):
        if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = _f
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


NAVY, RED, GREEN, GRAY = "#1f3a5f", "#c0392b", "#27ae60", "#95a5a6"
SUSPICIOUS_LOW, SUSPICIOUS_HIGH = 0.35, 0.65


def save_all(true, probs):
    plt = _mpl()
    os.makedirs(IMG_DIR, exist_ok=True)
    preds = {k: (v >= 0.5).astype(int) for k, v in probs.items()}

    # 1) 세 모델 성능 비교
    names = ["Accuracy", "Precision", "Recall", "F1"]
    colors = {"LightGBM": NAVY, "CNN": GREEN, "Ensemble": RED}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.26
    order = ["LightGBM", "CNN", "Ensemble"]
    for i, mname in enumerate(order):
        m = metrics(true, preds[mname])
        vals = [m["acc"], m["prec"], m["rec"], m["f1"]]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=mname, color=colors[mname])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels=names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("점수")
    ax.set_title("모델별 성능 지표 (동일 테스트셋)", fontsize=13, weight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "metrics_comparison.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 2) 앙상블 혼동행렬
    m = metrics(true, preds["Ensemble"])
    cm = m["cm"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["정상 (0)", "피싱 (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("예측 라벨")
    ax.set_ylabel("실제 라벨")
    ax.set_title(f"앙상블 혼동행렬 (정확도 {m['acc']*100:.2f}%)", fontsize=13, weight="bold")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, weight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 3) 앙상블 예측 확률 분포
    prob = probs["Ensemble"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, 1, 51)
    ax.hist(prob[true == 0], bins=bins, color=NAVY, alpha=0.75, label="정상 (실제)")
    ax.hist(prob[true == 1], bins=bins, color=RED, alpha=0.75, label="피싱 (실제)")
    ax.axvspan(SUSPICIOUS_LOW, SUSPICIOUS_HIGH, color=GRAY, alpha=0.25,
               label=f"의심 범위 ({SUSPICIOUS_LOW}~{SUSPICIOUS_HIGH})")
    ax.set_yscale("log")
    ax.set_xlabel("앙상블 예측 피싱 확률")
    ax.set_ylabel("URL 개수 (로그 스케일)")
    ax.set_title(f"앙상블 예측 확률 분포 (테스트 n={len(true):,})", fontsize=13, weight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "probability_distribution.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"이미지 저장: {IMG_DIR}/*.png")


def main():
    df, train_idx, test_idx = load_split()
    true = df["Label"].values[test_idx]
    print(f"학습 {len(train_idx):,} / 테스트 {len(test_idx):,}")

    print("\n=== LightGBM 학습 ===")
    lgbm_prob = run_lgbm(df, train_idx, test_idx)

    print("\n=== CNN 학습 ===")
    cnn_prob = run_cnn(df, train_idx, test_idx)

    ens_prob = (lgbm_prob + cnn_prob) / 2
    probs = {"LightGBM": lgbm_prob, "CNN": cnn_prob, "Ensemble": ens_prob}

    print("\n=== 결과 ===")
    for name in ["LightGBM", "CNN", "Ensemble"]:
        m = metrics(true, (probs[name] >= 0.5).astype(int))
        print(f"{name:9s} acc={m['acc']:.4f} prec={m['prec']:.4f} "
              f"rec={m['rec']:.4f} f1={m['f1']:.4f}")

    pd.DataFrame({
        "url": df["url"].values[test_idx],
        "Label": true,
        "lgbm_prob": lgbm_prob,
        "cnn_prob": cnn_prob,
        "ensemble_prob": ens_prob,
    }).to_csv(OUT_CSV, index=False)
    print(f"예측 결과 저장: {OUT_CSV}")

    save_all(true, probs)
    print("완료")


if __name__ == "__main__":
    main()
