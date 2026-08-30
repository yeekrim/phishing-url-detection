"""앙상블 효과 검증 — 진짜 held-out 테스트 (LightGBM vs CNN vs 앙상블).

run_comparison.py 는 valid_data 를 나눠 평가하므로 학습/평가 분포가 같고,
전체 피처를 쓴 LightGBM 이 정확도 1.0 이 나온다(과적합·누수 의심).

이 스크립트는 완전히 분리된 파일로 일반화 성능을 정직하게 측정한다.
  - 학습: data/final/final_train.csv (url, Label)
  - 평가: data/final/final_test.csv  (url, Label)
  - 두 파일엔 url/Label 만 있어 URL 문자열 신호만 사용(엔지니어링 피처 X)

주의점 두 가지를 그대로 드러낸다.
  1) 데이터 누수: 테스트 URL 상당수가 학습셋에도 존재한다. 학습셋 중복을 제거하고,
     테스트를 '학습과 겹치는 URL' / '처음 보는 URL' 로 나눠 각각 평가한다.
  2) 피싱 URL 재사용: 공개 피싱 목록은 재사용이 많아 '처음 보는' 피싱은 극소수다.
     따라서 처음 보는 정상 URL 에 대한 '오탐(FP)' 이 가장 신뢰할 수 있는 clean 지표다.

산출물:
  - data/final/holdout_predictions.csv   : 테스트셋 예측 결과(세 모델 확률 + is_unseen)
  - docs/images/holdout_metrics.png       : held-out 성능 비교(전체 테스트)
  - docs/images/holdout_false_positives.png : 처음 보는 정상 URL 오탐률

    python src/holdout_experiment.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import CountVectorizer
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

TRAIN = os.path.join("data", "final", "final_train.csv")
TEST = os.path.join("data", "final", "final_test.csv")
OUT_CSV = os.path.join("data", "final", "holdout_predictions.csv")
IMG_DIR = os.path.join("docs", "images")

MAX_LEN = 100
LGBM_PARAMS = {"learning_rate": 0.1, "max_depth": -1, "n_estimators": 100, "num_leaves": 31}
ORDER = ["LightGBM", "CNN", "Ensemble"]
NAVY, GREEN, RED = "#1f3a5f", "#27ae60", "#c0392b"
COLORS = {"LightGBM": NAVY, "CNN": GREEN, "Ensemble": RED}


def metrics(true, pred):
    tp = int(((pred == 1) & (true == 1)).sum()); tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum()); fn = int(((pred == 0) & (true == 1)).sum())
    acc = (tp + tn) / len(true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "cm": (tn, fp, fn, tp)}


def report(tag, true, probs):
    print(f"\n########## {tag}  (n={len(true):,}) ##########")
    print(f"{'model':10s} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7}   (tn,fp,fn,tp)")
    out = {}
    for name in ORDER:
        m = metrics(true, (probs[name] >= 0.5).astype(int))
        out[name] = m
        print(f"{name:10s} {m['acc']:7.4f} {m['prec']:7.4f} {m['rec']:7.4f} "
              f"{m['f1']:7.4f}   {m['cm']}")
    return out


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


def plot_metrics(full_metrics, n):
    plt = _mpl()
    names = ["Accuracy", "Precision", "Recall", "F1"]
    keys = ["acc", "prec", "rec", "f1"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.26
    for i, mname in enumerate(ORDER):
        vals = [full_metrics[mname][k] for k in keys]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=mname, color=COLORS[mname])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels=names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("점수")
    ax.set_title(f"held-out 테스트 성능 (URL 문자열만, n={n:,})", fontsize=13, weight="bold")
    ax.legend()
    ax.text(0.5, -0.16,
            "완전히 분리된 final_train → final_test · 엔지니어링 피처 미사용",
            transform=ax.transAxes, ha="center", fontsize=8, color="#7f8c8d")
    fig.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, "holdout_metrics.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"이미지 저장: {IMG_DIR}/holdout_metrics.png")


def plot_false_positives(true_unseen, probs_unseen):
    plt = _mpl()
    n_normal = int((true_unseen == 0).sum())
    fig, ax = plt.subplots(figsize=(7, 5))
    fps, rates = [], []
    for mname in ORDER:
        pred = (probs_unseen[mname] >= 0.5).astype(int)
        fp = int(((pred == 1) & (true_unseen == 0)).sum())
        fps.append(fp)
        rates.append(fp / n_normal * 100)
    bars = ax.bar(ORDER, rates, color=[COLORS[m] for m in ORDER], width=0.55)
    for b, fp, r in zip(bars, fps, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}%\n({fp}건)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("오탐률 (정상을 피싱으로 오판)")
    ax.set_ylim(0, max(rates) * 1.3)
    ax.set_title(f"처음 보는 정상 URL 오탐률 (n={n_normal:,})", fontsize=13, weight="bold")
    ax.text(0.5, -0.13, "낮을수록 좋음 · 앙상블이 가장 보수적(오탐 최소)",
            transform=ax.transAxes, ha="center", fontsize=8, color="#7f8c8d")
    fig.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, "holdout_false_positives.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"이미지 저장: {IMG_DIR}/holdout_false_positives.png")


def main():
    # --- 데이터 ---
    tr = pd.read_csv(TRAIN, usecols=["url", "Label"]).dropna()
    te = pd.read_csv(TEST, usecols=["url", "Label"]).dropna()
    tr["url"] = tr["url"].astype(str); te["url"] = te["url"].astype(str)
    tr = tr.drop_duplicates("url").reset_index(drop=True)
    print(f"학습 {len(tr):,} (중복 제거) / 테스트 {len(te):,}")

    tr_tok = [" ".join(tokenize_url(u)) for u in tr["url"]]
    te_tok = [" ".join(tokenize_url(u)) for u in te["url"]]
    y_tr = tr["Label"].values
    y_te = te["Label"].values

    train_url_set = set(tr["url"])
    unseen = np.array([u not in train_url_set for u in te["url"]])
    print(f"테스트 중 처음 보는 URL: {unseen.sum():,} / 학습과 겹치는 URL: {(~unseen).sum():,} "
          f"({(~unseen).mean()*100:.1f}% 누수)")

    # --- LightGBM (URL 토큰 BoW) ---
    print("\n[LightGBM 학습]")
    vec = CountVectorizer(dtype=np.float32)
    X_tr = vec.fit_transform(tr_tok)
    X_te = vec.transform(te_tok)
    lgbm = LGBMClassifier(**LGBM_PARAMS)
    lgbm.fit(X_tr, y_tr)
    lgbm_prob = lgbm.predict_proba(X_te)[:, 1]

    # --- CNN ---
    print("[CNN 학습]")
    tok = Tokenizer(); tok.fit_on_texts(tr_tok)
    Xc_tr = pad_sequences(tok.texts_to_sequences(tr_tok), maxlen=MAX_LEN)
    Xc_te = pad_sequences(tok.texts_to_sequences(te_tok), maxlen=MAX_LEN)
    vocab = len(tok.word_index) + 1
    model = Sequential([
        Embedding(vocab, 32), Dropout(0.3),
        Conv1D(32, 5, padding="valid", activation="relu"),
        GlobalMaxPooling1D(), Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(Xc_tr, y_tr, epochs=4, batch_size=256, validation_split=0.1,
              callbacks=[es], verbose=2)
    cnn_prob = model.predict(Xc_te, verbose=0).flatten()

    # --- 앙상블 ---
    ens_prob = (lgbm_prob + cnn_prob) / 2
    probs = {"LightGBM": lgbm_prob, "CNN": cnn_prob, "Ensemble": ens_prob}

    # --- 결과 ---
    full_metrics = report("전체 테스트셋 (학습과 겹치는 URL 포함 = 누수)", y_te, probs)
    report("처음 보는 URL만 (진짜 held-out; 피싱 재사용으로 양성 극소수)",
           y_te[unseen], {k: v[unseen] for k, v in probs.items()})

    # --- 예측 결과 저장 ---
    pd.DataFrame({
        "url": te["url"].values,
        "Label": y_te,
        "is_unseen": unseen.astype(int),
        "lgbm_prob": lgbm_prob,
        "cnn_prob": cnn_prob,
        "ensemble_prob": ens_prob,
    }).to_csv(OUT_CSV, index=False)
    print(f"\n예측 결과 저장: {OUT_CSV}")

    # --- 시각화 ---
    plot_metrics(full_metrics, len(y_te))
    plot_false_positives(y_te[unseen], {k: v[unseen] for k, v in probs.items()})
    print("완료")


if __name__ == "__main__":
    main()
