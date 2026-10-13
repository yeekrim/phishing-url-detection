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
| all | 1e-07 | 1 | 0.7767 |
| all | 3e-07 | 1 | 0.8174 |
| all | 1e-06 | 1 | 0.8393 |
| all | 3e-06 | 1 | 0.8266 |
| all | 1e-05 | 1 | 0.7905 |
| url | 1e-06 | 5 | 0.6085 |
| url | 1e-06 | 20 | 0.6001 |
| url | 1e-06 | 50 | 0.6254 |
| all | 1e-06 | 5 | 0.8389 |
| all | 1e-06 | 20 | 0.8236 |
| all | 1e-06 | 50 | 0.7881 |

선택: {'url': {'alpha': 1e-06, 'hosted_weight': 50, 'oof_score': 0.6253751849076467}, 'all': {'alpha': 1e-06, 'hosted_weight': 1, 'oof_score': 0.8392564762585204}}

## LightGBM [features] (시간 검증: 학습 228,875행 / 검증 77,520행, 검증의 호스팅 정상 158행)

| 설정 | 가중치 | 반복 | 기준 점수 |
|---|---|---|---|
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 125 | 0.3857 |
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 117 | 0.3756 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 102 | 0.3972 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 116 | 0.3720 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 106 | 0.4420 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 135 | 0.4218 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 92 | 0.4477 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 116 | 0.4207 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 179 | 0.4685 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 253 | 0.4424 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 15 | 0.4560 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 231 | 0.4573 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 5 | 74 | 0.5020 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 20 | 70 | 0.4973 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 50 | 74 | 0.5039 |

선택: {'num_leaves': 255, 'min_data_in_leaf': 50, 'feature_fraction': 0.5}, 가중치 50, 반복 74, 기준 점수 0.5039 (사이트 이름 피처를 빼면 0.4924)

## LightGBM [stacked] (시간 검증: 학습 228,875행 / 검증 77,520행, 검증의 호스팅 정상 158행)

| 설정 | 가중치 | 반복 | 기준 점수 |
|---|---|---|---|
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 713 | 0.8445 |
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 280 | 0.8536 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 225 | 0.8445 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 170 | 0.8468 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 424 | 0.8491 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 135 | 0.8504 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 5 | 0.8349 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 136 | 0.8488 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 509 | 0.8466 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 374 | 0.8570 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 5 | 0.8279 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 136 | 0.8514 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 5 | 380 | 0.8622 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 20 | 415 | 0.8548 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 50 | 278 | 0.8528 |

선택: {'num_leaves': 255, 'min_data_in_leaf': 50, 'feature_fraction': 0.8}, 가중치 5, 반복 380, 기준 점수 0.8622 (사이트 이름 피처를 빼면 0.8488)
