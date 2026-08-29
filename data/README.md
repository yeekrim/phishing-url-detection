# 데이터 설명

수집·전처리 과정에서 나온 데이터를 단계별로 정리했다.

## `raw/` — 원본 URL 목록
| 파일 | 설명 |
| --- | --- |
| `normal_data.csv` / `normal_data.xls` | 정상 URL 목록 |
| `gov_normal_sites.csv` | 정부기관 정상 사이트 목록 |
| `aws_normal_sites.csv` | AWS 등에서 수집한 정상 사이트 목록 |

> 피싱 URL 은 공개 피싱 URL 데이터셋(예: PhishTank / OpenPhish 계열)에서 수집했으며,
> 최종 데이터셋에는 정상 약 150,000개 + 피싱 약 139,994개가 포함된다.

## `features/` — 피처 추출 중간 산출물
`raw/` 의 URL 에 `src/features.py` 의 함수를 단계적으로 적용하며 저장한 중간 파일들.
| 파일 | 설명 |
| --- | --- |
| `base_1.csv` | URL 문자열 기반 피처(IP_LIKE ~ Query)까지 계산한 기본 데이터 |
| `0623_*.csv`, `0624_*.csv` | WHOIS/HTTP 기반 피처를 날짜별로 이어 붙인 산출물 |
| `Hyperlinks112.csv` | 하이퍼링크 수 피처 계산 샘플 |
| `5.30_20000.xls` | 초기 20,000건 스냅샷 |

## `final/` — 최종 학습·검증 데이터 및 결과
| 파일 | 설명 |
| --- | --- |
| `valid_data.csv` | **15개 피처 + Tokenized_url + Label 이 모두 포함된 검증용 데이터** (모델 입력 형식의 표준 예시) |
| `final_train.csv` | 학습용 (정상 100,000 + 피싱 100,000) — `url`, `Label` |
| `final_test.csv` | 테스트용 (정상 50,000 + 피싱 39,994) — `url`, `Label` |
| `model_percentage.csv` | 검증 데이터에 대한 LightGBM / CNN / 앙상블 예측 확률 결과 |

## 피처 컬럼 (valid_data.csv 기준)
`url, IP_LIKE, AT, URL_Depth, Redirection, Is_Https, TINY_URL, Check_Hyphen, Query,
Domain_Age, Domain_end, Mouseover, Web_forwards, Hyperlinks, Domain_Cons, Tokenized_url, Label`

각 피처의 의미는 [`../src/features.py`](../src/features.py) 의 docstring 을 참고.
