# v3 설정 고르기

평가(test) 데이터는 쓰지 않았다. 기준 점수 = 피싱 비율 0.05/0.1/0.5/1/5% 에서의 AP 평균.

## n-gram 모델 (train 교차 예측, 기준 점수 = 피싱 비율 0.05~5% AP 평균)

| 모델 | alpha | 가중치 | 기준 점수 |
|---|---|---|---|
| url | 1e-07 | 1 | 0.6642 |
| url | 3e-07 | 1 | 0.6946 |
| url | 1e-06 | 1 | 0.6915 |
| url | 3e-06 | 1 | 0.6637 |
| url | 1e-05 | 1 | 0.6135 |
| all | 1e-07 | 1 | 0.7856 |
| all | 3e-07 | 1 | 0.8734 |
| all | 1e-06 | 1 | 0.8811 |
| all | 3e-06 | 1 | 0.8719 |
| all | 1e-05 | 1 | 0.8464 |
| url | 3e-07 | 5 | 0.6939 |
| url | 3e-07 | 20 | 0.7018 |
| url | 3e-07 | 50 | 0.6853 |
| all | 1e-06 | 5 | 0.8803 |
| all | 1e-06 | 20 | 0.8723 |
| all | 1e-06 | 50 | 0.8383 |

선택: {'url': {'alpha': 3e-07, 'hosted_weight': 20, 'oof_score': 0.7017615245034524}, 'all': {'alpha': 1e-06, 'hosted_weight': 1, 'oof_score': 0.8810526389801572}}

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
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 651 | 0.8807 |
| {"num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 316 | 0.8734 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 828 | 0.8785 |
| {"num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 255 | 0.8708 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 401 | 0.8759 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 669 | 0.8809 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 379 | 0.8761 |
| {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 102 | 0.8727 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.5} | 1 | 912 | 0.8798 |
| {"num_leaves": 255, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 1 | 386 | 0.8798 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.5} | 1 | 396 | 0.8768 |
| {"num_leaves": 255, "min_data_in_leaf": 300, "feature_fraction": 0.8} | 1 | 237 | 0.8788 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 5 | 883 | 0.8861 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 20 | 122 | 0.8686 |
| {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.8} | 50 | 936 | 0.8848 |

선택: {'num_leaves': 127, 'min_data_in_leaf': 50, 'feature_fraction': 0.8}, 가중치 5, 반복 883, 기준 점수 0.8861 (사이트 이름 피처를 빼면 0.8585)
