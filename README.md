# 머신러닝·딥러닝 앙상블을 활용한 피싱 사이트 탐지

WHS 2기 **떡밥팀** 프로젝트를 코드·데이터 중심으로 정리한 저장소입니다.
URL 에서 추출한 15개 피처와 텍스트(URL 토큰)를 이용해 **LightGBM(머신러닝)** 과
**1D-CNN(딥러닝)** 을 학습하고, 두 모델의 예측 확률을 평균한 **앙상블**로 피싱 여부를 판별합니다.

> 발표자료: [`docs/떡밥팀_최종발표.pdf`](docs/떡밥팀_최종발표.pdf)

## 프로젝트 개요
- **문제** — 블랙리스트 방식만으로는 피싱 공격의 약 20%를 막지 못하고, 수법이 계속 다양해짐
- **접근** — URL 자체의 특징 + 실제 페이지의 특징을 피처로 만들어 지도학습으로 탐지
- **데이터** — 정상 URL 약 150,000개 + 피싱 URL 약 139,994개, 검증용 별도 수집(정상 50,000 / 피싱 20,000)
- **모델** — LightGBM, CNN, 그리고 둘의 soft-voting 앙상블

## 저장소 구조
```
phishing-url-detection/
├── src/
│   ├── features.py        # 15개 URL 피처 추출 함수 (핵심 재사용 모듈)
│   ├── build_dataset.py   # URL 목록 → 피처 데이터셋(CSV) 생성
│   ├── train_lgbm.py      # LightGBM 학습·평가
│   ├── train_cnn.py       # 1D-CNN 학습·평가
│   └── ensemble.py        # LGBM + CNN 앙상블 평가 (+ 의심 범위 분류)
├── data/
│   ├── raw/               # 원본 URL 목록
│   ├── features/          # 피처 추출 중간 산출물
│   └── final/             # 최종 학습·검증 데이터 및 결과
│                          #   (자세한 설명은 data/README.md)
└── docs/
    └── 떡밥팀_최종발표.pdf
```

## 피처 15종
| # | 컬럼 | 의미 | 계산 방식 |
| --- | --- | --- | --- |
| 1 | `IP_LIKE` | URL 에 IP 주소 형식 포함 여부 | 문자열 |
| 2 | `AT` | `@` 포함 여부 | 문자열 |
| 3 | `URL_Depth` | 경로(`/`) 깊이 | 문자열 |
| 4 | `Redirection` | 프로토콜 이후 `//` 리디렉션 의심 | 문자열 |
| 5 | `Is_Https` | https 사용 여부 | 문자열 |
| 6 | `TINY_URL` | 단축 URL 서비스 사용 여부 | 문자열 |
| 7 | `Check_Hyphen` | `-` 포함 여부 | 문자열 |
| 8 | `Query` | 쿼리 파라미터 개수 | 문자열 |
| 9 | `Domain_Age` | 도메인 생성일(2년 기준) | WHOIS |
| 10 | `Domain_end` | 도메인 만료일(5개월 기준) | WHOIS |
| 11 | `Mouseover` | `onmouseover` 태그 존재 | HTTP |
| 12 | `Web_forwards` | 웹 포워딩(리디렉션) 횟수 | HTTP |
| 13 | `Hyperlinks` | 페이지 내 `<a>` 태그 수 | HTTP |
| 14 | `Domain_Cons` | 접속 전후 도메인 일관성 | HTTP |
| — | `Tokenized_url` | URL 토큰열(텍스트 모델 입력) | 문자열 |

> 값 규칙(대체로): `0` = 의심, `-1` = 정상 간주, `1` = 피싱 의심. 네트워크 계열은 실패 시 `-1`.

## 사용법
### 1. 설치
```bash
pip install -r requirements.txt
```

### 2. URL 목록에서 피처 데이터셋 만들기
```bash
# 문자열 기반 피처만 (빠름, 네트워크 불필요)
python src/build_dataset.py data/raw/normal_data.csv out.csv --no-network

# 전체 피처 (WHOIS/HTTP 요청 포함, 느림)
python src/build_dataset.py data/raw/normal_data.csv out.csv
```

### 3. 모델 학습
```bash
# 머신러닝 (LightGBM)
python src/train_lgbm.py --data data/final/valid_data.csv --model-out lgbm_model.joblib

# 딥러닝 (CNN)
python src/train_cnn.py  --data data/final/valid_data.csv --model-out cnn_best.keras
```

### 4. 앙상블 평가
```bash
python src/ensemble.py \
    --data data/final/valid_data.csv \
    --lgbm lgbm_model.joblib \
    --cnn  cnn_best.keras \
    --out  ensemble_result.csv
```

## 결과 요약
`valid_data.csv` 를 계층적 8:2 로 나눠(학습 56,000 / 테스트 14,000) **LightGBM · CNN · 앙상블**을
동일한 테스트셋에서 학습·평가했습니다. 아래 그래프는 한 번의 실행으로 재현할 수 있습니다.

```bash
python src/run_comparison.py   # 세 모델 학습 + docs/images/*.png 생성
```

| 모델 | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| LightGBM | 1.000 | 1.000 | 1.000 | 1.000 |
| CNN | 0.963 | 0.983 | 0.885 | 0.932 |
| **앙상블** | **0.9999** | **0.9998** | **1.000** | **0.9999** |

### 모델별 성능 비교
CNN 을 다시 학습시켜 세 모델을 동일 조건에서 비교했습니다. LightGBM 이 가장 높고,
앙상블은 두 모델의 예측 확률을 평균하여 CNN 단독보다 재현율을 끌어올립니다.

![모델별 성능 비교](docs/images/metrics_comparison.png)

### 앙상블 혼동행렬
테스트 14,000건에 대해 오분류는 1건으로 정확도 약 100%입니다.

![앙상블 혼동행렬](docs/images/confusion_matrix.png)

### 예측 확률 분포
정상/피싱이 확률 양극단으로 뚜렷하게 분리되며, 가운데 회색 구간이 **의심 범위(0.35~0.65)** 입니다.
확률대별로 **정상 / 의심 / 피싱** 세 구간으로 분류합니다.

![예측 확률 분포](docs/images/probability_distribution.png)

> 재학습한 세 모델의 테스트셋 예측값은 [`data/final/model_comparison.csv`](data/final/model_comparison.csv) 에 저장됩니다.
> (발표 당시 결과 원본은 [`data/final/model_percentage.csv`](data/final/model_percentage.csv) — 이 파일의 `CNN 예측값` 컬럼은
> 저장 과정에서 누락되어 있어, 위 비교는 CNN 을 새로 학습시킨 결과를 사용합니다.)

## 팀
WHS 2기 떡밥팀 — 박찬민, 김주령, 김해찬, 김현진, 이경림, 이소영, 홍혜원 (멘토: 손승호)

## 라이선스
학습·연구 목적의 프로젝트 아카이브입니다.
