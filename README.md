# ko-sbert-experiment
ko-RoBERTa-base 모델을 기반으로 Pooling Layer와 Siamese Network를 추가하여 s-BERT를 구현하기 위한 프로젝트임.
sentence-BERT 논문을 확인해보면, STS로 테스트 하는 경우와 NLI->STS로 테스트 하는 경우 2가지가 소개되어있는데
Fine-Tuning 시 사용하는 데이터셋의 순서가 성능에 영향을 미치는 지 확인해보고자 본 레포지토리를 개설함.

## 실험 구조

1. **STS Only**  
   - KorSTS 데이터로 CosineSimilarityLoss 학습  
   - 평가: STS dev (학습 중), STS test (최종)

2. **NLI Only**  
   - KorNLI 데이터로 MultipleNegativesRankingLoss 학습  
   - 평가: STS dev (학습 중), STS test (최종)

3. **STS -> NLI**  
   - 위 (1)에서 학습한 STS 모델을 불러와, KorNLI 데이터로 추가 학습  
   - 평가: STS dev/test

4. **NLI -> STS**  
   - 위 (2)에서 학습한 NLI 모델을 불러와, KorSTS 데이터로 추가 학습  
   - 평가: STS dev/test

## 폴더 구조
```
my-sbert-experiments/
├── README.md
├── requirements.txt
├── data/
│   ├── kor-nlu-datasets/                # (옵션) 직접 포함할 수도, 또는 git clone 방식으로 받을 수도 있음
│   │   ├── KorNLI/
│   │   └── KorSTS/
├── notebooks/
│   ├── 1_STS_Only.ipynb                # Colab notebook - 실험1
│   ├── 2_NLI_Only.ipynb                # Colab notebook - 실험2
│   ├── 3_STS_then_NLI.ipynb            # Colab notebook - 실험3
│   └── 4_NLI_then_STS.ipynb            # Colab notebook - 실험4
└── utils/
    └── evaluation.py                    # evaluate_on_sts_test 등 평가 함수
```
