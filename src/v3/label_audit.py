"""라벨 범위 점검. (v3 한계 1번)

label_groups.py 의 묶음(blogspot_google·casino·betting)이 나머지 피싱과 성격이 다른지 재고,
그 묶음을 뺀 벤치마크 점수를 공식 점수와 나란히 계산한다. 공식 점수는 바꾸지 않는다.

    1) 묶음별 행 수 (train / test 피싱)
    2) 묶음별 성격: 비밀번호 입력란·폼이 있는 비율, 본문 50자 미만 비율, 제목에 공격 대상 이름이 나오는 비율
    3) 묶음별 탐지율 (최종 스태킹 모델, test 전체에서 오탐률 0.1% 가 되는 임계값)
    4) 벤치마크 평균 AP: 공식 vs 묶음의 피싱을 세트에서 뺀 값
       논문 모델은 이 부분집합으로 다시 잴 수 없으므로 논문 수치와는 비교하지 않는다.

출력: reports/v3/label_audit.md
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

from evaluate import BENCH_DIR, MODELS, RATES, read_list
from label_groups import GROUPS, label_group
from text_vectors import FEATURE_DIR
from train import SCORES_PATH

REPORT = os.path.join("reports", "v3", "label_audit.md")
COLUMNS = ["sha256", "label", "target", "url_clean", "title", "text_len", "n_forms", "n_password_inputs"]


def brand_in_title(target: str, title: str) -> bool:
    """target 이름(또는 4자 이상 조각)이 제목에 나오는가. 공백·기호는 무시한다."""
    norm = re.sub(r"[^a-z0-9]", "", (title or "").lower())
    target = target if isinstance(target, str) else ""
    title = title if isinstance(title, str) else ""
    parts = [re.sub(r"[^a-z0-9]", "", p) for p in re.split(r"[\s_]+", target.lower())]
    whole = "".join(parts)
    return bool(whole) and (whole in norm or any(len(p) >= 4 and p in norm for p in parts))


def load() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(p, columns=COLUMNS).assign(split=os.path.basename(p)[:4])
                    for p in sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet")))])
    df["split"] = df["split"].map({"trai": "train", "test": "test"})
    df["group"] = np.where(df.label == 1, [label_group(t, u) for t, u in zip(df.target, df.url_clean)], "benign")
    df["brand_in_title"] = [brand_in_title(t, x) for t, x in zip(df.target, df.title)]
    scores = pd.read_parquet(SCORES_PATH).drop(columns=["split", "label", "target"], errors="ignore")
    return df.merge(scores[["sha256"] + list(MODELS)], on="sha256", how="left")


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        part = df[df.split == split]
        for group in ("benign", "other") + GROUPS:
            g = part[part.group == group]
            if not len(g):
                rows.append({"분할": split, "묶음": group, "행 수": 0})
                continue
            rows.append({"분할": split, "묶음": group, "행 수": len(g),
                         "비밀번호 입력란": (g.n_password_inputs > 0).mean(), "폼 있음": (g.n_forms > 0).mean(),
                         "본문 50자 미만": (g.text_len < 50).mean(),
                         "제목에 대상 이름": g.brand_in_title.mean() if group != "benign" else np.nan})
    return pd.DataFrame(rows)


def detection(test: pd.DataFrame) -> pd.DataFrame:
    col = "score_lgbm_stacked"
    fpr, _, th = roc_curve(test.label, test[col])
    thr = th[np.where(fpr <= 0.001)[0][-1]]
    phish = test[test.label == 1]
    missed = int((phish[col] < thr).sum())
    rows = []
    for group in ("other",) + GROUPS:
        g = phish[phish.group == group]
        miss = int((g[col] < thr).sum())
        rows.append({"묶음": group, "피싱": len(g), "탐지율": (g[col] >= thr).mean(), "놓침": miss,
                     "놓친 것 중 비율": miss / missed})
    return pd.DataFrame(rows)


def benchmark(test: pd.DataFrame, exclude: set[str]) -> tuple[pd.DataFrame, dict]:
    t = test.set_index("sha256")
    benign = [h for h in read_list(os.path.join(BENCH_DIR, "benign.txt")) if h in t.index]
    rows, info = [], {"removed": 0, "kept": 0, "empty_sets": 0}
    for folder, rate in RATES.items():
        per = {col: [] for col in MODELS}
        for f in sorted(os.listdir(os.path.join(BENCH_DIR, folder))):
            phish = [h for h in read_list(os.path.join(BENCH_DIR, folder, f)) if h in t.index]
            keep = [h for h in phish if t.at[h, "group"] not in exclude]
            info["removed"] += len(phish) - len(keep)
            info["kept"] += len(keep)
            if not keep:
                info["empty_sets"] += 1
                continue
            sub = t.loc[benign + keep]
            for col in MODELS:
                per[col].append(average_precision_score(sub.label, sub[col]))
        for col, name in MODELS.items():
            rows.append({"모델": name, "피싱 비율": rate, "AP": np.mean(per[col])})
    table = pd.DataFrame(rows).pivot_table(index="모델", columns="피싱 비율", values="AP")[list(RATES.values())]
    return table, info


def main() -> None:
    df = load()
    test = df[df.split == "test"]
    lines = ["# v3 라벨 범위 점검", "",
             "묶음 규칙은 `src/v3/label_groups.py` (평가 전에 커밋). 공식 점수는 묶음을 빼지 않은 값이다.", ""]

    prof = profile(df)
    lines += ["## 1·2) 묶음별 행 수와 성격", "",
              "비율은 해당 묶음 안에서의 비율. '제목에 대상 이름' 은 PhreshPhish target 이름(또는 4자 이상 조각)이 "
              "페이지 제목에 나오는 비율 — 브랜드를 사칭하는 페이지인지 보는 대략적인 지표다.", "",
              prof.round(3).to_markdown(index=False), ""]

    det = detection(test)
    lines += ["## 3) 묶음별 탐지율 (최종 스태킹, test 전체 오탐률 0.1% 임계값)", "",
              det.round(3).to_markdown(index=False), ""]

    official, info0 = benchmark(test, set())
    lines += ["## 4) 벤치마크 평균 AP", "", "### 공식 (묶음을 빼지 않음)", "", official.round(4).to_markdown(), ""]
    for name, exclude in (("blogspot_google 제외", {"blogspot_google"}),
                          ("casino·betting 제외", {"casino", "betting"}),
                          ("세 묶음 모두 제외", set(GROUPS))):
        table, info = benchmark(test, exclude)
        lines += [f"### {name}", "",
                  f"세트에서 뺀 피싱 {info['removed']:,}건 / 남은 피싱 {info['kept']:,}건, "
                  f"피싱이 하나도 남지 않아 건너뛴 세트 {info['empty_sets']}개", "",
                  table.round(4).to_markdown(), ""]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
