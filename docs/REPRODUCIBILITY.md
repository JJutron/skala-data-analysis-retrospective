# 코드 실행과 재현 범위

## 포함한 코드

기존 제출 패키지에서 stage2 후보의 고정 설정 재학습·추론에 필요한 파일을 추렸다. 원본 탐색 스크립트의 `main()`은 과거 실험 폴더를 참조하므로, 이 저장소에서는 아래 두 진입점만 사용한다.

- `train_selected.py`: `recipe.json`의 고정 설정으로 학습한다. 새로운 탐색을 수행하지 않는다.
- `predict_submission.py`: 학습 모델로 Public·Private를 예측하고 `expected/`의 동결 예측과 비교한다.

금융은 보조 변화량 예측을 활용하는 prior와 NCA·KNN의 혼합이고, 건강식품은 기존 기준 모델과 달력 수요 모델의 혼합이다. 정확한 구성은 [recipe.json](../src/recipe.json)에 있다.

## 환경과 데이터

Python 3.12와 [기록된 의존성 버전](../requirements.txt)을 사용한다. 제공받은 다음 파일을 `src/data/`에 둔다.

```text
train.csv
test_public.csv
test_private.csv
sample_submission_public.csv
sample_submission_private.csv
```

원본 데이터 인코딩은 `cp949`다. 원본 데이터와 평가 정답은 Git에 올리지 않으며, `expected/`에는 재현 비교용 예측만 포함한다.

## 실행 순서

저장소 최상위에서:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd src
python -m unittest test_clean_causal test_clean_causal96_stage2
python train_selected.py
python predict_submission.py --model-dir models_retrained
```

학습 모델은 `src/models_retrained/`, 예측은 `src/outputs/`에 생긴다. 기존 모델 폴더가 있으면 재학습은 덮어쓰지 않고 중단한다. 재실행할 때는 새 폴더를 지정한다.

```sh
python train_selected.py --output-dir models_retrained_2
python predict_submission.py --model-dir models_retrained_2
```

원본 패키지의 저장 모델은 이 저장소에 없으므로 `predict_submission.py`를 기본 옵션으로 바로 실행하지 않는다.

## 검사하는 내용

합성 데이터 검사에서는 미래 입력을 변경·제거했을 때 앞선 특성과 예측이 같은지, Target 열 변경이 입력에 영향을 주지 않는지, 마지막 3행의 미성숙 라벨이 학습에서 제외되는지 등을 확인한다. 평가 정답과 금지된 과거 모델 읽기를 차단하는 검사도 포함한다.

예측 진입점은 Public 100행·Private 200행을 처리하고, 동결 예측과 절대 오차 `1e-9` 이내로 일치하는지 확인한다. 확률의 유한성과 0~1 범위도 점검한다. 이 비교에는 평가 정답이 필요하지 않다.

## 해석 범위

재현 대상의 로컬 Public 평균은 `0.832397`이며, 참여자가 제공한 공식 4회차의 세 점수와 소수점 여섯 자리까지 일치한다. [공식 제출 이력](../evidence/official_submissions.json). 업로드 CSV의 해시 대조는 아직 하지 않았다. 참여자가 전달한 최종 1등 평균 `0.856276`과는 다른 기록이다. 정확한 최종 제출 파일·설정이 연결되기 전에는 최종 우승 모델의 재현이라고 표현하지 않는다.

실제 미래 변수를 사용하지 않는다는 검사와 독립적인 성능 평가는 다르다. 과거 Public 정답을 본 이력이 있으므로 전체 연구가 완전히 블라인드였다고 주장하지 않는다.

## 저장소 정리 후 확인

2026-10-16에 포함 코드로 자동 검사 9개를 통과했고 고정 설정을 다시 학습했다. Public 100행·Private 200행이 모두 동결 예측과 일치했다. 문서 내부 링크 19개, 최종 평균 계산, 공식 1~4회차 평균의 반올림 일치도 확인했다. [확인 기록](../evidence/repository_verification.json)
