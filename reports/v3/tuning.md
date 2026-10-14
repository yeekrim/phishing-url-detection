# v3 설정 고르기

평가(test) 데이터는 쓰지 않았다. 기준 점수 = 피싱 비율 0.05/0.1/0.5/1/5% 에서의 AP 평균.

## n-gram 모델 (train 교차 예측, 기준 점수 = 피싱 비율 0.05~5% AP 평균)

| 모델 | alpha | 가중치 | 기준 점수 |
|---|---|---|---|
| url | 1e-07 | 1 | 0.5689 |
| url | 3e-07 | 1 | 0.6096 |
| url | 1e-06 | 1 | 0.6115 |
| url | 3e-06 | 1 | 0.5845 |
| url | 1e-05 | 1 | 0.5366 |
| all | 1e-07 | 1 | 0.7199 |
| all | 3e-07 | 1 | 0.8278 |
| all | 1e-06 | 1 | 0.8455 |
| all | 3e-06 | 1 | 0.8328 |
| all | 1e-05 | 1 | 0.7986 |
| url | 1e-06 | 5 | 0.6085 |
| url | 1e-06 | 20 | 0.6001 |
| url | 1e-06 | 50 | 0.6254 |
| all | 1e-06 | 5 | 0.8446 |
| all | 1e-06 | 20 | 0.8278 |
| all | 1e-06 | 50 | 0.7494 |

선택: {'url': {'alpha': 1e-06, 'hosted_weight': 50, 'oof_score': 0.6253751849076467}, 'all': {'alpha': 1e-06, 'hosted_weight': 1, 'oof_score': 0.8454874042346778}}

## LightGBM [features] (시간 검증: 학습 228,875행 / 검증 77,520행, 검증의 호스팅 정상 158행)

| 설정 | 가중치 | 반복 | 기준 점수 |
|---|---|---|---|
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 107 | 0.3945 |
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 181 | 0.4188 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 137 | 0.4192 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 128 | 0.4396 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 87 | 0.4257 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 178 | 0.4445 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 50 | 0.4726 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 96 | 0.4307 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 86 | 0.4648 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 157 | 0.4665 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 57 | 0.4842 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 115 | 0.4406 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 5 | 57 | 0.4939 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 20 | 72 | 0.4791 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 50 | 30 | 0.4758 |

선택: {'num_leaves': 255, 'min_data_in_leaf': 300, 'feature_fraction': 0.5}, 가중치 5, 반복 57, 기준 점수 0.4939 (사이트 이름 피처를 빼면 0.5107)

## LightGBM [stacked] (시간 검증: 학습 228,875행 / 검증 77,520행, 검증의 호스팅 정상 158행)

| 설정 | 가중치 | 반복 | 기준 점수 |
|---|---|---|---|
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 601 | 0.8712 |
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 280 | 0.8688 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 435 | 0.8755 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 438 | 0.8698 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 423 | 0.8729 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 867 | 0.8794 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 307 | 0.8704 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 143 | 0.8666 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 412 | 0.8667 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 325 | 0.8785 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 345 | 0.8648 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 452 | 0.8745 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 5 | 489 | 0.8810 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 20 | 265 | 0.8738 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 50 | 374 | 0.8749 |

선택: {'num_leaves': 127, 'min_data_in_leaf': 50, 'feature_fraction': 0.8}, 가중치 5, 반복 489, 기준 점수 0.8810 (사이트 이름 피처를 빼면 0.8796)
