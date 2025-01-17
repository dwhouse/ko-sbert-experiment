---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:568640
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: 청바지에 빨간 셔츠를 입은 여자가 주먹을 꽉 쥐고 발을 뻗은 채 포즈를 취하고 있다.
  sentences:
  - 그 여자는 불꽃놀이로 만들어졌다.
  - 한 어린 소녀가 호수 근처에 위치한 웅덩이를 피하기 위해 길을 깡충깡충 뛰어다닌다.
  - 여자는 빨간 셔츠를 입고 포즈를 취했다.
- source_sentence: 아이가 밖에 있다.
  sentences:
  - 한 고위 정치 시위자가 사회보장제도의 폐지를 요구하는 철자를 잘못 쓴 팻말을 들고 있다.
  - 분홍색 코트를 입은 어린 아이가 보도 위의 간판을 지우고 있다.
  - 한 남성이 포스터 벽 앞 의자에 앉아 있다.
- source_sentence: 저가 경구 수화는 탈수를 치료할 수 있다.
  sentences:
  - 예를 들어, 저렴한 구강 수화는 개발 도상국에서 설사 질환에 대한 탈수 치료를 위해 연구되고 사용되었습니다.
  - 저가의 구강 수화는 의학적인 이점이 없다.
  - 구덩이에는 물이 있다.
- source_sentence: 만약 유대인들이 이슬람교와 무함마드라는 공식적인 캠페인을 시작했다고 상상해보라. 당신은 그 지방이 만들어 낼 것이라고
    상상할 수 있는가?
  sentences:
  - 유대인들은 이슬람교와 무함마드가 많은 지방을 생산할 수 있다고 말하는 캠페인을 시작했다.
  - 만약 유대인들이 무함마드가 이슬람과 아무 관련이 없다고 말하는 공식 캠페인을 시작했다면, 그것은 아마도 어떤 종류의 새로운 이슬람 판결이나
    법을 초래하지 않을 것이다.
  - (a) LSC와 정부, 또한 청원인은 Rust v.
- source_sentence: 오르간 조끼를 입은 노동자가 삽을 사용하고 있다.
  sentences:
  - 어떤 토끼도 오렌지색 조끼를 입고 삽을 사용하지 않는다.
  - 오렌지색 조끼를 입은 노동자가 삽을 사용하고 있다.
  - 콜트를 얼마나 잘 치료했는지 쉽게 알 수 있다.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: sts dev
      type: sts-dev
    metrics:
    - type: pearson_cosine
      value: 0.8651853943363769
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.8674529174406812
      name: Spearman Cosine
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '오르간 조끼를 입은 노동자가 삽을 사용하고 있다.',
    '오렌지색 조끼를 입은 노동자가 삽을 사용하고 있다.',
    '어떤 토끼도 오렌지색 조끼를 입고 삽을 사용하지 않는다.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `sts-dev`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.8652     |
| **spearman_cosine** | **0.8675** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 568,640 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | sentence_2                                                                        |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                            |
  | details | <ul><li>min: 4 tokens</li><li>mean: 18.94 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 19.07 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 14.59 tokens</li><li>max: 44 tokens</li></ul> |
* Samples:
  | sentence_0                            | sentence_1                                   | sentence_2                             |
  |:--------------------------------------|:---------------------------------------------|:---------------------------------------|
  | <code>한 아이가 현미경으로 표본을 관찰하고 있다.</code> | <code>아이가 현미경을 사용하고 있다.</code>               | <code>한 아이가 창문 밖으로 현미경을 던지고 있다.</code> |
  | <code>던져진 공을 잡는 큰 검은 개</code>         | <code>큰 검은 개가 공을 잡기 위해 공중으로 뛰어오르고 있다.</code> | <code>큰 검은 개가 막대기를 잡고 있다</code>        |
  | <code>사람들이 환승 열차에 앉아 있다.</code>       | <code>사람들이 기차를 타고 있다.</code>                 | <code>사람들은 외계인이다.</code>               |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `num_train_epochs`: 1
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | sts-dev_spearman_cosine |
|:------:|:----:|:-------------:|:-----------------------:|
| 0.0563 | 500  | 0.873         | -                       |
| 0.1125 | 1000 | 0.6455        | 0.8557                  |
| 0.1688 | 1500 | 0.5673        | -                       |
| 0.2251 | 2000 | 0.5399        | 0.8670                  |
| 0.2814 | 2500 | 0.5122        | -                       |
| 0.3376 | 3000 | 0.4973        | 0.8576                  |
| 0.3939 | 3500 | 0.4743        | -                       |
| 0.4502 | 4000 | 0.4645        | 0.8604                  |
| 0.5065 | 4500 | 0.4467        | -                       |
| 0.5627 | 5000 | 0.4308        | 0.8610                  |
| 0.6190 | 5500 | 0.4228        | -                       |
| 0.6753 | 6000 | 0.4211        | 0.8675                  |


### Framework Versions
- Python: 3.11.11
- Sentence Transformers: 3.4.0.dev0
- Transformers: 4.47.1
- PyTorch: 2.5.1+cu121
- Accelerate: 1.2.1
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
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

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->