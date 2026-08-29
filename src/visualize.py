"""검증 데이터 예측 결과(model_percentage.csv)로 결과 시각화 이미지를 생성한다.

생성물 (docs/images/):
  - probability_distribution.png : 예측 확률 분포 + 의심 범위(0.35~0.65)
  - confusion_matrix.png         : 앙상블 모델 혼동행렬
  - metrics_comparison.png       : 모델별 성능 지표 비교

참고: model_percentage.csv 의 'CNN 예측값' 컬럼은 저장 과정에서 전부 0(정상)으로
누락되어 있어 신뢰할 수 없으므로, 비교 그래프에서는 LightGBM 과 앙상블만 사용한다.

    python src/visualize.py
"""

from __future__ import annotations

import csv
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# macOS 한글 폰트
for _f in ("AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"):
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

DATA = os.path.join("data", "final", "model_percentage.csv")
OUT_DIR = os.path.join("docs", "images")

# 의심 범위 (ensemble.py 와 동일)
SUSPICIOUS_LOW = 0.35
SUSPICIOUS_HIGH = 0.65

NAVY = "#1f3a5f"
RED = "#c0392b"
GRAY = "#95a5a6"


def load():
    true, lgbm, ens, prob = [], [], [], []
    with open(DATA, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            true.append(int(r["실제값"]))
            lgbm.append(int(r["LightGBM"]))
            ens.append(int(r["앙상블 예측값"]))
            prob.append(float(r["예측 확률"].strip("[]")))
    return (np.array(true), np.array(lgbm), np.array(ens), np.array(prob))


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


def plot_probability_distribution(true, prob):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, 1, 51)
    ax.hist(prob[true == 0], bins=bins, color=NAVY, alpha=0.75, label="정상 (실제)")
    ax.hist(prob[true == 1], bins=bins, color=RED, alpha=0.75, label="피싱 (실제)")
    ax.axvspan(SUSPICIOUS_LOW, SUSPICIOUS_HIGH, color=GRAY, alpha=0.25,
               label=f"의심 범위 ({SUSPICIOUS_LOW}~{SUSPICIOUS_HIGH})")
    ax.set_yscale("log")
    ax.set_xlabel("예측 피싱 확률")
    ax.set_ylabel("URL 개수 (로그 스케일)")
    ax.set_title("검증 데이터 예측 확률 분포 (n=70,000)", fontsize=13, weight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, "probability_distribution.png")


def plot_confusion_matrix(true, ens):
    m = metrics(true, ens)
    cm = m["cm"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["정상 (0)", "피싱 (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("예측 라벨")
    ax.set_ylabel("실제 라벨")
    ax.set_title(f"앙상블 혼동행렬 (정확도 {m['acc']*100:.2f}%)",
                 fontsize=13, weight="bold")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, weight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


def plot_metrics_comparison(true, lgbm, ens):
    models = {"LightGBM": lgbm, "Ensemble": ens}
    names = ["Accuracy", "Precision", "Recall", "F1"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.35
    colors = [NAVY, RED]
    for i, (mname, pred) in enumerate(models.items()):
        m = metrics(true, pred)
        vals = [m["acc"], m["prec"], m["rec"], m["f1"]]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=mname, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels=names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("점수")
    ax.set_title("모델별 성능 지표 (검증 데이터)", fontsize=13, weight="bold")
    ax.legend()
    ax.text(0.5, -0.16,
            "※ model_percentage.csv 의 CNN 예측 컬럼은 저장 누락으로 제외",
            transform=ax.transAxes, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    _save(fig, "metrics_comparison.png")


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {path}")


def main():
    true, lgbm, ens, prob = load()
    plot_probability_distribution(true, prob)
    plot_confusion_matrix(true, ens)
    plot_metrics_comparison(true, lgbm, ens)
    print("완료")


if __name__ == "__main__":
    main()
