"""머신러닝(LightGBM) 피싱 탐지 모델 학습·평가.

입력 데이터는 build_dataset.py 로 만든 피처 CSV 형식을 가정한다.
컬럼: url, IP_LIKE ... Domain_Cons, Tokenized_url, Label

- Tokenized_url 은 CountVectorizer 로 벡터화하여 텍스트 피처로 사용
- 나머지 수치형 피처와 결합한 뒤 LGBMClassifier 로 학습
- GridSearchCV 로 찾은 최적 하이퍼파라미터를 기본값으로 사용

예시:
    python src/train_lgbm.py --data data/final/valid_data.csv --model-out lgbm.joblib
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# 발표 당시 GridSearchCV 로 찾은 최적 하이퍼파라미터
BEST_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": -1,
    "n_estimators": 100,
    "num_leaves": 31,
}

DROP_COLS = ["url", "Label", "Tokenized_url"]


def build_matrix(df: pd.DataFrame, vectorizer: CountVectorizer, fit: bool):
    """텍스트 피처(Tokenized_url) + 수치형 피처를 결합한 희소 행렬을 만든다."""
    if fit:
        text = vectorizer.fit_transform(df["Tokenized_url"].astype(str))
    else:
        text = vectorizer.transform(df["Tokenized_url"].astype(str))
    other = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).values
    return hstack([text, other])


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM 피싱 탐지 모델 학습")
    parser.add_argument("--data", required=True, help="피처 CSV (Tokenized_url, Label 포함)")
    parser.add_argument("--model-out", default="lgbm_model.joblib", help="모델 저장 경로")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y = df["Label"]

    vectorizer = CountVectorizer()
    X = build_matrix(df, vectorizer, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model = LGBMClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== LightGBM 테스트 성능 ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump({"model": model, "vectorizer": vectorizer}, args.model_out)
    print(f"\n모델 저장: {args.model_out}")

    # 상위 피처 중요도 출력
    text_names = vectorizer.get_feature_names_out()
    other_names = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).columns
    names = np.concatenate([text_names, other_names])
    imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    print("\n상위 15개 피처 중요도:")
    print(imp.sort_values("importance", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
