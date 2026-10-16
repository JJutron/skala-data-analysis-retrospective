# 사용 기법과 모델 구조

두 타깃은 입력 데이터가 같아도 잘 맞는 패턴이 달랐다. 금융은 변화량과 시장 간 차이를 중심으로, 건강식품은 요일과 과거 수요를 중심으로 모델을 분리했다. 아래 파라미터와 혼합식은 저장소의 `recipe.json`에 고정된 4회차 재현 구성 기준이다.

## 1. 금융 특성: 수준에서 변화로

미국 3대 지수(S&P 500·NASDAQ·Dow)는 1·2·3일 차분을 만들었다. 환율(원/달러·원/엔·원/위안), 금, WTI는 1일 차분을 사용했다. 원본 값이 표준화된 실습 데이터라 가격 변화율 대신 차분으로 표현했다.

추가로 다음 변수 간 차이의 1일 변화를 사용했다.

- 회사채 3년물 − CD 91일물
- 구리 − 금
- NASDAQ − Dow

구리·금 특성의 코드 이름은 `Ratio_Copper_Gold_d1`이지만 실제 계산은 두 표준화 값의 차이를 다시 차분한 것이다. 원시 가격 비율로 해석하지 않는다. 검색어는 마스크·코로나의 3일 차분을 추가하고, NCA와 보조 모델의 core 입력에는 요일 원핫 변수를 결합했다.

구현: [features.py](../src/features.py), [feature_frames](../src/clean_causal.py).

## 2. 금융 기준 모델: 규제 로지스틱 회귀

```text
금융 모멘텀 특성
  → 수치 결측 중앙값 대체 + RobustScaler
  → 요일 OneHotEncoder
  → L2 LogisticRegression(C=0.03)
```

초기 기준 모델은 미국 지수 1·3일 차분, 원/달러·금·WTI 1일 차분과 요일을 사용한다. 작은 데이터에서 규제가 있는 선형 모델로 출발하고, 시계열 개발 구간에서 입력 표현의 효과를 확인했다.

구현: [make_model](../src/validate.py), [momentum 특성 목록](../src/features.py).

## 3. 금융 보조 모델: teacher의 신호를 student로 추정

금융 변화량의 분류 신호를 현재 특성에서 추정하기 위해 **LogisticRegression teacher + Ridge student** 구성을 사용했다.

1. 학습 구간 내부의 미래 금융 변화량으로 `StandardScaler + LogisticRegression(C=0.1)` teacher를 학습한다.
2. teacher의 분류 점수에서 현재 이미 관측한 미국 지수 변화의 기여를 뺀다.
3. 나머지 점수를 현재·과거 core 특성으로 예측하도록 `중앙값 대체 + StandardScaler + Ridge(alpha=1000)` student를 학습한다.
4. 추론에서는 관측된 기여와 student가 예측한 나머지를 더한다. 실제 미래 관측은 읽지 않는다.

미국 지수의 시차 가정을 offset 1과 2로 나눠 두 모델을 학습하고 같은 비중으로 평균한다. 점수는 학습 구간의 평균·표준편차로 표준화한 뒤 `0.5 + arctan(z) / π`로 0~1에 매핑한다. 이는 순위를 유지하는 변환이며 확률 보정 모델을 따로 학습한 것은 아니다.

학습 데이터의 마지막 3행은 완성되지 않은 D+3 라벨 때문에 제외한다. 미래 변화량은 학습용 목표 구성에만 쓰는 것이 핵심이다.

구현: [fit_proxy·predict_proxy·finance_predict](../src/clean_causal96_stage1.py).

## 4. 금융 비선형 모델: NCA + KNN

```text
core 금융 특성 + 요일 원핫
  → SimpleImputer(strategy="median")
  → StandardScaler
  → NCA(n_components=4, init="pca", max_iter=80, random_state=42)
  → KNN(n_neighbors=90, weights="distance")
```

NCA(Neighborhood Components Analysis)는 분류에 유리한 투영을 학습하는 거리 학습 기법이다. 변환된 4차원 공간에서 가까운 과거 사례 90개를 찾고, 가까운 사례일수록 큰 가중치를 주어 예측한다.

KNN 출력은 logit으로 점수화하고 학습 구간 통계로 표준화한 뒤 arctan 매핑을 적용한다. 반복 상한 80회는 최적화 설정이며 수렴 보장을 뜻하지 않는다.

구현: [estimator·fit·predict_model](../src/clean_causal96_stage2.py).

## 5. 건강식품 특성: 요일과 과거 수요

건강식품은 요일별 상승 비율 차이가 컸다. Weekday는 CatBoost의 범주형 입력으로 처리하고, 식품·생활건강 쇼핑 변수에 달력 수요 특성을 추가했다.

| 특성 | 계산 |
|---|---|
| D+3 요일 예상 수요 | 과거 같은 요일 값의 EWMA, alpha=0.25 |
| calendar_delta | 예상 수요 − 현재 값 |
| calendar_ratio | 예상 수요 / (현재 값의 절댓값 + 0.005) |
| trend7 | 현재 값 − 7행 전 값 |
| season_ratio | 4·11·18·25행 전 평균 / (현재 값의 절댓값 + 0.005) |

주기가 7행이면 D+3과 D−4·D−11·D−18·D−25의 요일이 같다. 미래 요일의 실제 수요 대신, 이미 관측한 같은 요일의 값으로 예상 수준을 만들었다. EWMA 이력도 한 행 뒤로 밀어 과거 관측만 사용했다.

구현: [feature_frames의 food_calendar](../src/clean_causal.py).

## 6. 건강식품 모델: CatBoost 두 개의 평균

| 구성 | 입력 | iterations | depth | learning_rate | l2_leaf_reg |
|---|---|---:|---:|---:|---:|
| 기준 CatBoost | 원본 입력 | 300 | 4 | 0.03 | 3 |
| 달력 수요 CatBoost | 식품·생활건강 원값 + 달력 수요 특성 + 요일 | 300 | 5 | 0.03 | 10 |

두 모델 모두 `random_seed=42`, `cat_features=["Weekday"]`로 학습한다. 예측 확률을 50:50으로 평균한다.

```text
건강식품 예측 = 0.5 × 기준 CatBoost + 0.5 × 달력 수요 CatBoost
```

구현: [estimator·fit_recipe·predict_recipe](../src/clean_causal.py).

## 7. 금융 앙상블의 실제 혼합식

`recipe.json`의 `two_offsets_half`와 `half_blend=true`가 순서대로 적용된다.

```text
보조 신호 = (offset 1 예측 + offset 2 예측) / 2
prior = 0.5 × 변화량 로지스틱 회귀 + 0.5 × 보조 신호
금융 예측 = 0.5 × prior + 0.5 × NCA·KNN
          = 0.25 × 변화량 로지스틱 회귀
          + 0.25 × 보조 신호
          + 0.50 × NCA·KNN
```

보조 신호의 두 offset을 각각 풀어 쓰면 각 12.5%다. 별도 메타 모델을 학습하는 스태킹 대신 고정 비율의 예측 평균을 사용한다.

구현: [recipe.json](../src/recipe.json), [predict_finance](../src/clean_causal96_stage2.py).

## 8. 시간순 검증과 재현

학습→검증의 시간 순서를 지키고 D+3 경계를 고려해 gap=3을 둔다. 100행·150행 검증 구간을 함께 비교했으며, 스케일러·결측 대체·NCA·모델은 학습 구간에서만 적합한다. 전체 924행 재학습에서는 최종 타깃 학습 시점을 921로 제한한다.

합성 데이터 검사로 미래 입력을 제거하거나 변경해도 앞선 예측이 같은지, Target 열 변경이 특성에 영향을 주지 않는지, 미성숙 라벨이 학습에서 제외되는지 확인한다. 재학습 후에는 Public 100행·Private 200행을 동결 예측과 비교한다.

[실행 방법](REPRODUCIBILITY.md) · [검증 및 실험 점수](EXPERIMENTS.md)
