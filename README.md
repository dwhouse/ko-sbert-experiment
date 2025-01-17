# ko-sBERT-experiment

This project aims to implement **Sentence-BERT (S-BERT)** based on the `klue/roberta-base` model by adding a pooling layer and a Siamese network structure. Inspired by the original [Sentence-BERT paper](https://arxiv.org/abs/1908.10084), we explore how the order of fine-tuning datasets affects performance, specifically comparing STS-only, NLI-only, STS → NLI, and NLI → STS training strategies.

The experiments use publicly available Korean datasets:
- [KorNLI](https://github.com/kakaobrain/KorNLUDatasets) for Natural Language Inference (NLI)
- [KorSTS](https://github.com/kakaobrain/KorNLUDatasets) for Semantic Textual Similarity (STS)

## Experimental Setup

1. **STS Only**
   - Training: KorSTS dataset with `CosineSimilarityLoss`
   - Evaluation: STS dev set during training, STS test set as the final evaluation

2. **NLI Only**
   - Training: KorNLI dataset with `MultipleNegativesRankingLoss` (MNR Loss)
   - Evaluation: STS dev set during training, STS test set as the final evaluation

3. **STS → NLI**
   - Training: Load the pre-trained STS model (from experiment 1) and fine-tune with the KorNLI dataset
   - Evaluation: STS dev and test sets

4. **NLI → STS**
   - Training: Load the pre-trained NLI model (from experiment 2) and fine-tune with the KorSTS dataset
   - Evaluation: STS dev and test sets

### Technical Details

- **Siamese Network Architecture**: The architecture employs a shared encoder with a pooling layer. Mean pooling is used to derive fixed-size sentence embeddings from the token-level representations.
- **Loss Functions**:
  - `CosineSimilarityLoss` for STS tasks
  - `MultipleNegativesRankingLoss` for NLI tasks
- **Base Model**: `klue/roberta-base` ([Hugging Face Model Card](https://huggingface.co/klue/roberta-base))

## Results

The following table summarizes the test scores for each experiment. Performance is evaluated using Pearson and Spearman correlation coefficients on the STS test set.

| Experiment       | Pearson (Cosine)  | Spearman (Cosine) |
|------------------|-------------------|-------------------|
| **STS Only**     | 0.816             | 0.814             |
| **NLI Only**     | 0.827             | 0.838             |
| **STS → NLI**    | 0.838             | 0.849             |
| **NLI → STS**    | 0.849             | 0.855             |

### Analysis

The results indicate that the **NLI → STS** approach achieves the highest performance, as suggested in the original Sentence-BERT paper. This can be attributed to the fact that NLI tasks involve diverse and nuanced relationships between sentence pairs, which may better prepare the model for capturing semantic similarity during the STS task.

## Repository Structure

```
.
├── notebooks/
│   ├── Experiment_Visualization.ipynb
│   ├── kosbert_experiment_1_sts_only.ipynb
│   ├── kosbert_experiment_2_nli_only.ipynb
│   ├── kosbert_experiment_3_sts_to_nli.ipynb
│   ├── kosbert_experiment_4_nli_to_sts.ipynb
├── README.md
├── outputs/
│   ├── nli-to-sts-2025-01-17/
│   ├── nli-only-klue-roberta-base-01-17/
│   ├── sts-to-nli-01-17/
│   ├── sts-only-klue-roberta-base-01-17/
├── data/
│   ├── KorNLI/
│   └── KorSTS/

```

## Citation

Please cite the following works if you use this repository or its results:

- **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
  ```
  @inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
   }
   ```
- **KorNLI and KorSTS Datasets**: [Kakao Brain](https://github.com/kakaobrain/KorNLUDatasets)
  ```
  @article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
  }
  ```
- **Base Model**: [KLUE RoBERTa-base](https://huggingface.co/klue/roberta-base)
  ```
  @misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
