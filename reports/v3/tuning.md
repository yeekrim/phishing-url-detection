# v3 설정 고르기

평가(test) 데이터는 쓰지 않았다. 기준 점수 = 피싱 비율 0.05/0.1/0.5/1/5% 에서의 AP 평균.

## n-gram 모델 (train 교차 예측, 기준 점수 = 피싱 비율 0.05~5% AP 평균)

| 모델 | alpha | 가중치 | 기준 점수 |
|---|---|---|---|
| url | 1e-07 | 1 | 0.6642 |
| url | 3e-07 | 1 | 0.6944 |
| url | 1e-06 | 1 | 0.6907 |
| url | 3e-06 | 1 | 0.6624 |
| url | 1e-05 | 1 | 0.6120 |
| all | 1e-07 | 1 | 0.8603 |
| all | 3e-07 | 1 | 0.8696 |
| all | 1e-06 | 1 | 0.8810 |
| all | 3e-06 | 1 | 0.8719 |
| all | 1e-05 | 1 | 0.8463 |
| url | 3e-07 | 5 | 0.6936 |
| url | 3e-07 | 20 | 0.7026 |
| url | 3e-07 | 50 | 0.6994 |
| all | 1e-06 | 5 | 0.8802 |
| all | 1e-06 | 20 | 0.8745 |
| all | 1e-06 | 50 | 0.8326 |

선택: {'url': {'alpha': 3e-07, 'hosted_weight': 20, 'oof_score': 0.7025708251965392}, 'all': {'alpha': 1e-06, 'hosted_weight': 1, 'oof_score': 0.8809986440872848}}

## LightGBM [features] (시간 검증: 학습 269,788행 / 검증 77,520행, 검증의 호스팅 정상 506행)

| 설정 | 가중치 | 반복 | 기준 점수 |
|---|---|---|---|
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 159 | 0.4133 |
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 162 | 0.4284 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 165 | 0.4422 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 149 | 0.4170 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 231 | 0.4652 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 257 | 0.4535 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 186 | 0.4586 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 157 | 0.4405 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 162 | 0.4766 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 189 | 0.4886 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 203 | 0.4754 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 169 | 0.4647 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 5 | 257 | 0.4773 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 20 | 100 | 0.4923 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 50 | 257 | 0.5171 |

선택: {'num_leaves': 255, 'min_data_in_leaf': 50, 'feature_fraction': 0.8}, 가중치 50, 반복 257, 기준 점수 0.5171 (사이트 이름 피처를 빼면 0.4769)

## LightGBM [stacked] (시간 검증: 학습 269,788행 / 검증 77,520행, 검증의 호스팅 정상 506행)

| 설정 | 가중치 | 반복 | 기준 점수 |
|---|---|---|---|
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 669 | 0.8725 |
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 310 | 0.8673 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 535 | 0.8704 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 170 | 0.8658 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 734 | 0.8837 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 139 | 0.8697 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 268 | 0.8758 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 440 | 0.8804 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 646 | 0.8776 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 114 | 0.8666 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 557 | 0.8779 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 94 | 0.8740 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 5 | 353 | 0.8717 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 20 | 403 | 0.8722 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 50 | 429 | 0.8705 |

선택: {'num_leaves': 127, 'min_data_in_leaf': 50, 'feature_fraction': 0.5}, 가중치 1, 반복 734, 기준 점수 0.8837 (사이트 이름 피처를 빼면 0.8800)
