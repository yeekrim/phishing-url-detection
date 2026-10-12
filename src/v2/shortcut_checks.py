"""지름길(shortcut) 검사. 모델을 학습하기 전에, 피처가 "피싱다움" 이 아니라
수집 방식의 차이로 클래스를 맞히고 있지 않은지 확인한다. (v2 4단계 준비)

검사 항목
    1. 분할 누수: train / val / test 사이에 같은 묶음(사이트)이 있는지
    2. 단일 피처 AUC: 피처 하나로 클래스를 얼마나 가르는지 (1 에 가까우면 지름길 의심)
       — 전체, 그리고 피싱 출처별로
    3. 피처 벡터 중복: 같은 피처 값인데 클래스가 다른 행, 평가 행 중 학습에 똑같은 벡터가 있는 비율
    4. 얕은 규칙: 깊이 2 결정 트리가 train → test 에서 얼마나 맞히는지, 어떤 규칙인지
    5. 수집 방식 차이: 루트 페이지 비율, 쿼리 비율, http 비율, 양쪽 클래스에 다 있는 묶음
    6. 메타데이터: 학습에 쓰지 않는 fetch_status 가 클래스를 얼마나 가르는지 (참고)

결과는 판정하지 않고 숫자만 남긴다. 해석은 결과를 보고 따로 한다.

예시:
    python src/v2/shortcut_checks.py dataset_main dataset_hyperlinks
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from build_dataset import HYPERLINK_FEATURES, MAIN_FEATURES
from build_features import INTERIM_DIR

REPORT_DIR = os.path.join("reports", "v2")


def auc(y, x) -> float:
    """방향과 무관한 AUC (0.5 = 구분 못 함, 1.0 = 완전히 구분)."""
    if pd.Series(x).nunique() < 2 or pd.Series(y).nunique() < 2:
        return float("nan")
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(INTERIM_DIR, f"{name}.csv"), low_memory=False)
    split = pd.read_csv(os.path.join(INTERIM_DIR, f"split_{name}.csv"), low_memory=False)
    return df.merge(split, on="url_sha256", validate="one_to_one")


def check(name: str, features: list[str]) -> list[str]:
    df = load(name)
    out = [f"## {name}", "", f"{len(df):,}행, 피처 {len(features)}개", ""]

    # 1. 분할 누수
    groups = {s: set(df.loc[df["split_domain"] == s, "group"]) for s in ("train", "val", "test")}
    out += ["### 1. 분할 누수 (묶음 겹침)", "",
            f"- train∩val {len(groups['train'] & groups['val'])}, train∩test {len(groups['train'] & groups['test'])}, "
            f"val∩test {len(groups['val'] & groups['test'])}",
            f"- 시간 분할 train∩test 묶음: "
            f"{len(set(df.loc[df.split_time == 'train', 'group']) & set(df.loc[df.split_time == 'test', 'group']))}", ""]

    # 2. 단일 피처 AUC
    train = df[df["split_domain"] == "train"]
    legit = train[train["Label"] == 0]
    rows = []
    for f in features:
        row = {"피처": f, "전체": auc(train["Label"], train[f])}
        for src in sorted(train.loc[train["Label"] == 1, "source"].unique()):
            part = pd.concat([legit, train[(train["Label"] == 1) & (train["source"] == src)]])
            row[src] = auc(part["Label"], part[f])
        rows.append(row)
    table = pd.DataFrame(rows).set_index("피처").round(3)
    out += ["### 2. 단일 피처 AUC (train, 방향 무관)", "", table.to_markdown(), ""]

    # 3. 피처 벡터 중복
    key = df[features].astype(str).agg("|".join, axis=1)
    labels_per_key = df.groupby(key)["Label"].nunique()
    conflict = key.map(labels_per_key) > 1
    test = df["split_domain"] == "test"
    in_train = key[test].isin(set(key[df["split_domain"] == "train"]))
    out += ["### 3. 피처 벡터 중복", "",
            f"- 서로 다른 피처 벡터: {key.nunique():,}개 / {len(df):,}행",
            f"- 같은 벡터가 두 클래스에 다 있는 행: {conflict.mean():.3f} "
            f"(정상 {conflict[df.Label == 0].mean():.3f} / 피싱 {conflict[df.Label == 1].mean():.3f})",
            f"- test 행 중 train 에 똑같은 벡터가 있는 비율: {in_train.mean():.3f}", ""]

    # 4. 얕은 규칙
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(train[features], train["Label"])
    te = df[test]
    prob = tree.predict_proba(te[features])[:, 1]
    acc = (tree.predict(te[features]) == te["Label"]).mean()
    out += ["### 4. 깊이 2 결정 트리 (train → test)", "",
            f"- test 정확도 {acc:.3f}, AUC {roc_auc_score(te['Label'], prob):.3f} "
            f"(다수 클래스만 찍을 때 정확도 {max(te.Label.mean(), 1 - te.Label.mean()):.3f})",
            "", "```", export_text(tree, feature_names=features).rstrip(), "```", ""]

    # 5. 수집 방식 차이
    by = df.groupby("Label")
    diff = pd.DataFrame({
        "루트 페이지(깊이 0, 쿼리 없음)": by.apply(lambda g: ((g.URL_Depth == 0) & (g.Query == 0)).mean(), include_groups=False),
        "쿼리 있음": by["Query"].apply(lambda s: (s > 0).mean()),
        "http (Is_Https=-1)": by["Is_Https"].apply(lambda s: (s == -1).mean()),
        "URL_Depth 중앙값": by["URL_Depth"].median(),
    }).T.round(3)
    both = set(df.loc[df.Label == 0, "group"]) & set(df.loc[df.Label == 1, "group"])
    out += ["### 5. 수집 방식 차이", "", diff.to_markdown(), "",
            f"- 두 클래스에 다 있는 묶음: {len(both)}개, {df['group'].isin(both).sum():,}행", ""]

    # 6. 메타데이터
    codes = df["fetch_status"].astype("category").cat.codes
    out += ["### 6. 메타데이터 (학습에 쓰지 않음, 참고)", "",
            f"- fetch_status 단일 AUC: {auc(df['Label'], codes):.3f}",
            "", pd.crosstab(df["fetch_status"], df["Label"], normalize="columns").round(3).to_markdown(), ""]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="학습 전 지름길 검사")
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    features = {"dataset_main": MAIN_FEATURES, "dataset_hyperlinks": HYPERLINK_FEATURES}
    lines = ["# v2 지름길 검사", ""]
    for name in args.datasets:
        lines += check(name, features[name])
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "shortcut_checks.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
