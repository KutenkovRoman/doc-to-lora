# Doc-to-LoRA: Code Overview

## 1. Что такое D2L и общая идея

**Doc-to-LoRA (D2L)** — гиперсеть, которая за один forward pass читает документ и генерирует матрицы LoRA для целевой LLM. Модель может отвечать на вопросы по тексту, не имея его в своём контекстном окне.

Математически цель — минимизировать KL-дивергенцию:
$$J = \text{KL}\left(p_\theta(y|x,c) \;\|\; p_{\theta + H_\phi(c)}(y|x)\right)$$
где $H_\phi(c)$ — функция гиперсети, предсказывающая LoRA-матрицы по контексту $c$.

**Схема прохода данных во время обучения:**

```
Документ (context)
       ↓
   ctx_encoder      (первые L//4 слоёв базовой LLM, заморожен)
       ↓
  [bs, n_layers, seq_len, hidden]  — активации слоёв
       ↓
   Perceiver aggregator            (trainable)
       ↓
  [bs, n_layers, n_modules, r, latent_size]
       ↓
   ResMLPBlock layers + Head       (trainable)
       ↓
  LoRA weights {A, B} per layer per module
       ↓
  apply_lora_to_layers(base_model)  — patch forward() у Linear-слоёв
       ↓
  base_model(question)             (заморожен)
       ↓
  KL(teacher_logits, student_logits) — loss только по обучаемым параметрам гиперсети
```

---

## 2. Структура репозитория

```
train.py                        — точка входа обучения
run_eval.py                     — точка входа оценки
watcher.py                      — фоновый watcher для перезапуска обучения
install.sh                      — установка зависимостей
accelerate_config.yaml          — конфиг Accelerate (multi-GPU / FSDP)
pyproject.toml / setup.py       — пакет ctx_to_lora
configs/                        — YAML-конфиги экспериментов
scripts/                        — shell-скрипты запуска
data/                           — скрипты генерации датасетов
chat_templates/                 — jinja-шаблоны чатов для разных моделей
src/ctx_to_lora/                — основной пакет
    configs.py
    trainer.py
    model_loading.py
    utils.py
    metrics.py
    eval_utils.py
    pooling.py
    modeling/
        hypernet.py             ← ГЛАВНЫЙ файл архитектуры
        aggregator.py           ← Perceiver aggregator
        ctx_encoder.py          ← Encoder контекста
        idefics2.py             ← Реализация Perceiver attention
        lora_layer.py           ← LoRA forward pass
        lora_merger.py          ← Chunking / merge LoRA чанков
        text_to_lora.py / text_to_lora_impl.py
        generative_adapter.py
        context_distillation.py
    data/
        processing.py
        collator.py
        packing.py
        preprocessing_fn.py
        definitions.py
        q_generation_template.py
        self_gen_template.py
```

---

## 3. Файлы верхнего уровня

### `train.py`
Главная точка входа, запускается через `accelerate launch`.

**Что делает:**
1. Парсит аргументы из YAML-конфига и CLI через `ArgumentParser`.
2. Загружает базовую LLM (`base_model`) через PEFT (с LoRA-обёрткой, но адаптер сразу отключается — `disable_adapter_layers()`).
3. Определяет `ctx_encoder_model_config` — конфиг модели, которая будет кодировать контекст.
4. Создаёт `HypernetConfig` из гиперпараметров.
5. Инстанциирует `ModulatedPretrainedModel` — главный объект, содержащий base_model + ctx_encoder + hypernet.
6. Загружает и токенизирует датасеты; применяет sequence packing.
7. Компилирует модули через `torch.compile`.
8. Запускает `train_model(...)` → `DistillationTrainer.train(...)`.

**Ключевые ветки:**
- `ctx_args.from_pretrained_checkpoint` — продолжить обучение с чекпоинта.
- `ctx_args.use_kl_loss = True` → использует `DistillationTrainer` (KL-дивергенция); `False` → `CrossEntropyTrainer` (обычный CE).

### `run_eval.py`
Аналогичная точка входа для оценки. Загружает модель из чекпоинта и прогоняет генеративную оценку на тестовых датасетах. Использует `eval_utils.py` и `metrics.py`.

### `watcher.py`
Простой loop, который перезапускает обучение при падении процесса. Полезно в долгих запусках на кластере.

### `install.sh`
```bash
# Устанавливает uv, создаёт venv Python 3.10
# Устанавливает torch 2.6 / CUDA 12.4
# Устанавливает flash-attention 2.7.4
# Скачивает SQuAD, строит compact-версии датасетов
uv pip install torch==2.6.0 ...
uv pip install flash_attn-2.7.4.post1...
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download rajpurkar/squad ...
uv run data/build_{drop,pwc,ropes,squad}_compact.py
```

### `accelerate_config.yaml`
Конфиг для `accelerate launch` — задаёт multi-GPU DDP или FSDP, число процессов, mixed precision (bf16).

---

## 4. `src/ctx_to_lora/configs.py` — все датаклассы аргументов

Один из важнейших файлов для понимания пайплайна. Содержит все dataclass-аргументы, которые парсятся из YAML + CLI.

### `ArgumentParser`
Расширяет `HfArgumentParser`. Умеет читать YAML-файл и перебивать его значения аргументами командной строки (`--key=value`). Вызов: `python train.py config.yaml --key=val`.

### `TrainingArguments`
Расширяет `transformers.TrainingArguments`. Ключевые дефолты:
- `bf16=True`, `tf32=True`
- `optim="adamw_torch_fused"`, `lr=4e-5`
- `lr_scheduler_type="cosine_with_min_lr"`
- `gradient_accumulation_steps` — задаётся в YAML
- `per_device_train_batch_size=1` — жёстко, т.к. данные упакованы (sequence packing)

### `ModelArguments`
- `model_name_or_path` — путь к базовой LLM (например, `google/gemma-2-2b-it`)
- `use_flash_attn` — включить flash attention 2

### `LoRAArguments`
- `lora_r` — ранг LoRA (базовый $r$ для одного чанка). По умолчанию 8.
- `lora_dropout`
- `target_modules` — список модулей для применения LoRA (например, `["down_proj"]`)

Важно: `lora_alpha` устанавливается автоматически как $r^{3/2} \cdot 2$ в `model_loading.py`.

### `CtxTrainingArguments`
- `exp_setup` — всегда `ExperimentSetup.HYPERLORA`
- `use_kl_loss` — `True`: KL-лосс (дистилляция); `False`: CE-лосс
- `use_per_ctx_average_loss` — усреднять лосс по контекстам, а не по токенам
- `gen_lora_l1_reg_coef` — коэффициент L1-регуляризации для сгенерированных LoRA
- `max_ctx_chunk_len` — максимальная длина чанка контекста в токенах
- `min_ctx_chunk_len` — минимальная длина чанка (для аугментации)
- `num_chunk_probs` — распределение вероятностей по числу чанков `{1: 0.5, 2: 0.3, 4: 0.2}`
- `max_ctx_chunk_num` — максимальное число чанков
- `max_packed_inp_len` / `max_packed_ctx_len` — длины упакованных последовательностей

### `HypernetArguments`
Гиперпараметры самой гиперсети. **Ключевые для модификации архитектуры:**
- `latent_size` — размерность латентного пространства (выход агрегатора и вход head). Дефолт: 512.
- `per_rank_gen` — если `True`, агрегатор производит отдельный вектор на каждый rank $r$, т.е. выход имеет форму `[bs, n_layers, n_modules, r, latent_size]`; если `False` — один вектор на модуль, потом расширяется.
- `num_pre_head_layers` — количество `ResMLPBlock` между агрегатором и финальным head.
- `per_layer_processing` — если `True`, `ResMLPBlock` заменяется на `ResMLPBlockPerLayer` (разные веса MLP на каждый слой LLM).
- `use_bias` — добавить data-independent LoRA bias (параметры `bias_A`, `bias_B`, обучаемые постоянные матрицы, прибавляются к предсказанным LoRA и образуют приращение при чанкинге).
- `dropout_rate` — dropout в `ResMLPBlock`.

### `AggregatorArguments`
Гиперпараметры агрегатора (Perceiver). **Ключевые для изменения архитектуры Perceiver:**
- `aggregator_type` — `"perceiver"` (единственный поддерживаемый вариант сейчас)
- `n_latent_queries` — число latent queries в Perceiver (стандартно 208 = 26 слоёв × 8 rank). Это $Q_m$ из статьи.
- `num_blocks` — число блоков cross-attention в Perceiver. Дефолт: 8 (в конфиге YAML можно задать 9).
- `num_self_attn_per_block` — число self-attention слоёв внутри каждого блока. 0 = только cross-attention.
- `shared_weights` — разделять ли веса между блоками Perceiver.

### `CtxEncoderArguments`
- `ctx_encoder_model_name_or_path` — если `None`, используется та же модель, что и base_model.
- `ctx_encoder_type` — тип энкодера:
  - `"per_layer_activations"` — **основной режим** (используется в main_exp). Берёт активации всех слоёв до `ctx_encoder_last_layer`. Выход: `[bs, n_layers, seq_len, hidden]`.
  - `"early_exit"` — выход только с `layer_idx`-го слоя. Выход: `[bs, seq_len, hidden]`.
  - `"embed_only"` — только embedding-слой.
- `layer_idx` — для `early_exit`: номер слоя (дефолт `L//4`).
- `ctx_encoder_last_layer` — для `per_layer_activations`: jusqu'à этого слоя берём активации.
- `quantize_ctx_encoder` — 4-bit квантизация энкодера через bitsandbytes.

---

## 5. `src/ctx_to_lora/modeling/` — архитектура модели

### ★★★ `hypernet.py` — **ГЛАВНЫЙ файл для модификации**

Содержит три ключевых класса:

---

#### `HypernetConfig`
Dataclass-конфиг, создаётся функцией `get_hypernet_config(...)`. Аккумулирует все параметры гиперсети:
- `latent_size`, `per_rank_gen`, `use_bias`, `per_layer_processing`, `num_pre_head_layers` — из `HypernetArguments`
- `layer_indices` — тензор индексов слоёв LLM (все от 0 до N-1)
- `feature_sizes` — словари `{module_name: d_in}` и `{module_name: d_out}` для каждого target_module LoRA
- `aggregator_config` — создаётся через `get_aggregator_config(...)`

`get_hypernet_config` считает число модулей LoRA и инициализирует `AggregatorConfig`.

---

#### `HyperLoRA` — **сердце гиперсети**

```
Агрегатор (Perceiver)
    вход: [bs, seq_len, feature_dim]  или  [bs, n_layers, seq_len, feature_dim]
    выход: [bs, n_layers, n_modules, r, latent_size]
          ↓
ResMLPBlock × num_pre_head_layers  (или ResMLPBlockPerLayer если per_layer_processing=True)
          ↓
L2 нормализация векторов
          ↓
head (EinMix): [bs, n_layers, n_modules, r, latent_size] → [bs, n_layers, n_modules, r, d_lora]
          ↓
_to_lora_dict():  разбить по модулям, разрезать на A и B, умножить на scaler_A / scaler_B
          ↓
{module: {"A": [bs, n_layers, r, d_in], "B": [bs, n_layers, r, d_out]}}
```

**Параметры `HyperLoRA`:**

- `self.aggregator` — агрегатор (инстанс `Perceiver` из `aggregator.py`)
- `self.layers` — `nn.Sequential` из `ResMLPBlock` или `ResMLPBlockPerLayer`. Это «предголовочные» слои.
- `self.head` — финальный линейный проектор (EinMix). Веса `head.weight` имеют форму `[n_layers, n_modules, d_latent, d_lora]` (при `n_modules > 1`). **Разные веса для каждого слоя и каждого модуля** — это реализация отдельных $W_A^{(l)},\, W_B^{(l)}$ из статьи.
- `self.bias_A`, `self.bias_B` — data-independent смещения LoRA (`ParameterDict` по модулям). Форма: `[n_layers, r, d_in/d_out]`. Добавляются в `combine_lora()`.
- `self.scaler_A`, `self.scaler_B` — обучаемые скаляры, умножаемые поэлементно на A и B (форма `[1, n_layers, r, 1]`). Инициализируются: `scaler_A = 1`, `scaler_B = 0` — т.е. инициально B-компонент нулевой (LoRA стартует с нуля).

**Метод `_to_lora_dict`:**
Принимает плоский тензор `[bs, n_layers, n_modules, r, max_io_dim]`, разрезает вдоль оси `n_modules` (через `einops.unpack`), затем для каждого модуля разрезает последнюю ось на A-часть (`d_in` элементов) и B-часть (`d_out` элементов).

**`ResMLPBlock`:**
```
LayerNorm → Dropout → Linear(d, 4d) → SiLU → Dropout → Linear(4d, d) → LayerNorm
```
+ residual-соединение. Применяется ко всем слоям и модулям одинаково.

**`ResMLPBlockPerLayer`:**
То же, но использует `EinMix` — отдельные матрицы для каждого слоя LLM. Tensor shape: `[bs, n_layers, n_modules, r, d]`. Включается флагом `per_layer_processing=True`.

---

#### `ModulatedPretrainedModel` — **внешний контейнер всей системы**

Владеет тремя под-моделями:
- `self.base_model` — целевая LLM (PeftModel с LoRA-обёрткой, адаптер отключён)
- `self.ctx_encoder` — энкодер контекста (один из вариантов `CTX_ENCODER_CLS`)
- `self.hypernet` — инстанс `HyperLoRA`

**Инициализация:**
1. `base_model.disable_adapter_layers()` — отключает PEFT LoRA-адаптер (мы генерируем свои веса)
2. `HyperLoRA(hypernet_config)` — создаёт гиперсеть
3. `patch_lora_forward()` — **патчит** `forward()` у каждого Linear-модуля, на котором стоит LoRA. Заменяет стандартный `peft` forward на кастомный `lora_forward` или `lora_forward_packed`. Новый forward принимает дополнительные аргументы `A`, `B`, `n_qs`, `seq_lens`.
4. Создаёт `ctx_encoder` нужного типа.

**Forward pass:**
```python
def forward(ctx_ids, ctx_attn_mask, ..., model_inputs_kwargs):
    # 1. Прогнать контекст через ctx_encoder
    ctx_features = self.ctx_encoder(ctx_ids, ...)
    # [bs, seq_len, hidden] или [bs, n_layers, seq_len, hidden]

    # 2. Прогнать через HyperLoRA
    lora_dict, _ = self.hypernet.generate_weights(ctx_features, ...)
    # {module: {"A": ..., "B": ...}}

    # 3. combine_lora — объединить чанки, добавить bias
    lora_dict = combine_lora(lora_dict, n_ctx_chunks, lora_bias=...)

    # 4. Вставить LoRA в слои base_model через partial
    apply_lora_to_layers(base_model, layer_indices, lora_dict, n_qs, position_ids)

    # 5. Прогнать вопрос через base_model (теперь каждый Linear имеет LoRA)
    outputs = self.base_model(**model_inputs_kwargs)
    return outputs
```

**`state_dict()` / `load_state_dict()`:**
Сохраняет **только** веса `hypernet` + метаданные (`base_model_name_or_path`, `hypernet_config`, `ctx_encoder_args`). `base_model` и `ctx_encoder` не сохраняются — они замороженные.

**`internalize(ctx_str)` / `reset()`:**
API для использования после обучения. `internalize` запускает контекст через гиперсеть и запоминает LoRA. `reset()` удаляет патч и сбрасывает состояние.

---

### ★★★ `aggregator.py` — Perceiver aggregator

Содержит класс `Perceiver` — реализацию механизма сжатия переменной длины.

**Архитектура `Perceiver`:**

```
ctx_features [bs, seq_len, feature_dim]
       ↓
Idefics2Perceiver(
    encoder: num_blocks блоков cross-attention,
             n_latents=n_latent_queries (latent queries, обучаемые),
             input_size=feature_dim, hidden_size=latent_size
    decoder: 1 блок cross-attention,
             n_latents=n_layers * n_modules * r  (или n_modules * r для layer_to_layer),
             задача — «спроецировать» в нужные размерности
)
       ↓
x: [bs, n_layers * n_modules * r, latent_size]
       ↓
rearrange → [bs, n_layers, n_modules, r, latent_size]
```

**Два режима (`layer_to_layer`):**
- `layer_to_layer = False` (стандартный): принимает хвостовые активации одного слоя или усреднённые.
- `layer_to_layer = True` (main_exp): принимает активации **всех** слоёв `[bs, n_layers, seq_len, hidden]`. Перед Perceiver они разворачиваются: `rearrange("bs n_layers seq_len d -> (n_layers bs) seq_len d")`. Это позволяет Perceiver'у обработать все слои независимо.

**Параметры `Perceiver`:**
- `self.perceiver` — инстанс `Idefics2Perceiver` из `idefics2.py`. Состоит из:
  - *encoder part*: `num_blocks` блоков, каждый = cross-attention (latents attend to context)
  - *decoder part*: 1 блок cross-attention с `n_output_queries` latent vectors → финальная компрессия в нужную форму

---

### ★★★ `idefics2.py` — реализация Perceiver attention

Взята из HuggingFace Idefics2 и модифицирована. Содержит полную реализацию блоков Perceiver.

**Ключевые классы:**

**`Idefics2PerceiverConfig`:**
Конфиг блока Perceiver. Важные поля:
- `input_size` — размерность входа (из контекста)
- `hidden_size` — размерность latent space
- `n_latents` — число latent queries
- `n_heads` / `num_key_value_heads` — MHA параметры (GQA)
- `head_dim` — размерность головы
- `num_blocks` / `num_self_attn_per_block` / `shared_weights`

**`Idefics2PerceiverAttention`:**
Базовый attention-модуль Perceiver. В проекте фактически используется его flash-attention вариант `Idefics2PerceiverFlashAttention2`, но смысл одинаковый.

Нужно различать два объекта:

- **`context`** — активации, пришедшие от `ctx_encoder`.
    Обычно сначала это
    $$[bs, seq\_len, hidden\_dim\_{ctx}]$$
    а затем через `modality_projection` они переводятся во внутреннее пространство Perceiver:
    $$[bs, seq\_len, d\_{perc}]$$
- **`latents`** — обучаемые latent queries, то есть специальные векторы-слоты, которыми Perceiver читает контекст.
    Их форма:
    $$[bs, n\_{latents}, d\_{perc}]$$

То есть латенты — это **не токены текста** и не активации base model. Это отдельные обучаемые векторы, которые играют роль фиксированного набора запросов к длинному контексту.

#### Cross-attention

В cross-attention:

- `Q` строится из `latents`
- `K, V` строятся из `context`

Математически:

$$
Q = \text{latents} \cdot W_Q,
\quad
K = \text{context} \cdot W_K,
\quad
V = \text{context} \cdot W_V
$$

Если обозначить:

- batch size = $B$
- число латентов = $M$
- длина контекста = $N$
- число attention heads = $H$
- число key/value heads = $H_{kv}$
- размерность головы = $d_h$

то формы такие:

$$
Q \in \mathbb{R}^{B \times M \times (H d_h)}
$$
$$
K,V \in \mathbb{R}^{B \times N \times (H_{kv} d_h)}
$$

После reshape:

- $Q$: $[B, H, M, d_h]$
- $K$: $[B, H, N, d_h]$ после `repeat_kv`
- $V$: $[B, H, N, d_h]$ после `repeat_kv`

Attention scores:

$$
A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)
\in \mathbb{R}^{B \times H \times M \times N}
$$

Далее каждый из $M$ латентов получает взвешенную сумму по всем $N$ позициям контекста:

$$
Z_{attn} = A V
$$

Итоговая идея: **длинная последовательность активаций документа сжимается в фиксированное число latent-векторов**.

#### Self-attention

В self-attention источник один и тот же: **текущие латенты**.

$$
Q = \text{latents} \cdot W_Q,
\quad
K = \text{latents} \cdot W_K,
\quad
V = \text{latents} \cdot W_V
$$

Если текущие латенты имеют форму $[B, M, d_{perc}]$, то:

$$
Q \in \mathbb{R}^{B \times M \times (H d_h)}
$$
$$
K,V \in \mathbb{R}^{B \times M \times (H_{kv} d_h)}
$$

После reshape attention scores имеют форму:

$$
\mathbb{R}^{B \times H \times M \times M}
$$

То есть теперь **латенты смотрят друг на друга**, а не на текст напрямую. Интуитивно:

- cross-attention = латенты читают документ;
- self-attention = латенты обмениваются информацией между собой.

Поддерживается Flash Attention 2 (в проекте именно он и используется).

**`Idefics2PerceiverLayer`:**
Один слой Resampler. На практике он может работать в двух режимах:

- `is_cross_attn=True` → `Q` из латентов, `K/V` из контекста;
- `is_cross_attn=False` → `Q/K/V` все из латентов.

После attention идут residual-соединение, LayerNorm и MLP.

**`Idefics2PerceiverResampler`:**
Стек слоёв `Idefics2PerceiverLayer`. Внутри него есть обучаемые latent queries `latents_q` формы `[n_latents, hidden_size]`, которые на forward расширяются до batch-размера.

Каждый блок Resampler устроен так:

1. один cross-attention слой;
2. затем `num_self_attn_per_block` self-attention слоёв.

Если обозначить:

- число блоков = $b = \text{num\_blocks}$
- число self-attn на блок = $s = \text{num\_self\_attn\_per\_block}$

то суммарно в одном Resampler:

$$
\#\text{cross-attention} = b
$$
$$
\#\text{self-attention} = b \cdot s
$$

Пример:

- `num_blocks=3`
- `num_self_attn_per_block=2`

даёт такую структуру:

```
Block 1: Cross → Self → Self
Block 2: Cross → Self → Self
Block 3: Cross → Self → Self
```

В текущем NIAH-эксперименте `num_self_attn_per_block=0`, поэтому внутри encoder-resampler только cross-attention.

**`Idefics2Perceiver`:**
Финальный класс, используемый в `aggregator.py`. Объединяет:
- `self.modality_projection` — сначала переводит входные активации из размерности `input_size` (например 2304 у Gemma) во внутреннюю размерность Perceiver `hidden_size` (например 512);
- `self.encoder = Idefics2PerceiverResampler(encoder_config)` — первый Resampler, который сжимает длинный контекст в фиксированное число латентов;
- `self.decoder = Idefics2PerceiverResampler(decoder_config)` — второй Resampler, который преобразует encoder-latents в нужное число выходных latent slots (`n_output_queries`).

Нормальный «живой» путь с attention внутри выглядит так:

```
1) ctx_encoder activations
    X_ctx: [bs, seq_len_ctx, hidden_ctx]

2) modality projection (обучаемый MLP)
    X = modality_projection(X_ctx)
    X: [bs, seq_len_ctx, d_perc]

3) encoder resampler (num_blocks раз)
    стартовые латенты L0 = latents_q: [bs, n_latent_queries, d_perc]  # latents_q обучаемые

    каждый блок:
      CrossAttn(L, X):
         Q = L W_Q
         K = X W_K
         V = X W_V
         L <- L + Attn(Q,K,V)   # + norm/mlp/residual внутри слоя

      затем SelfAttn(L, L, L) повторяется num_self_attn_per_block раз:
         Q = L W_Q
         K = L W_K
         V = L W_V
         L <- L + Attn(Q,K,V)

    выход encoder:
    L_enc: [bs, n_latent_queries, d_perc]

4) decoder resampler
    берёт L_enc как context и свои decoder-латенты как queries
    после cross-attention (и опциональных self-attention):
    L_out: [bs, n_output_queries, d_perc]

5) aggregator reshape
    [bs, n_output_queries, d_perc]
    -> [bs, n_layers, n_modules, r, d_latent]

6) HyperLoRA head
    генерирует LoRA A/B для каждого слоя/модуля
```

#### Подробно про пункт 4: `decoder resampler`

Это **не один линейный слой**, а второй Perceiver-Resampler (структурно такой же класс, как encoder-resampler):

- вход `context` = `L_enc` с формы `[bs, n_latent_queries, d_perc]`
- вход `queries` = decoder `latents_q` с формы `[bs, n_output_queries, d_perc]`

Далее внутри блока:

1. Cross-attention (обязательно):
    - `Q = latents_dec * W_Q`
    - `K = L_enc * W_K`
    - `V = L_enc * W_V`
    - `latents_dec <- latents_dec + Attn(Q,K,V)`
2. Затем `num_self_attn_per_block` self-attention шагов (если > 0):
    - `Q,K,V` все строятся из текущих decoder-латентов.
3. После каждого attention-шага есть residual/norm/MLP как в `Idefics2PerceiverLayer`.

Результат decoder-resampler:

- `L_out: [bs, n_output_queries, d_perc]`

где обычно
`n_output_queries = n_layers * (n_modules * r + n_extra_modules)`
(или упрощённый вариант в layer-to-layer режиме).

Обучаемые параметры **именно decoder-resampler**:

- `latents_q` decoder-а
- attention-проекции: `W_Q`, `W_K`, `W_V`, `W_O`
- MLP внутри Perceiver-слоёв: `gate_proj`, `up_proj`, `down_proj`
- RMSNorm параметры

#### Подробно про пункт 6: `HyperLoRA head`

На вход head приходит тензор после reshape:

- `Z: [bs, n_layers, n_modules, r, d_latent]`

Head (`EinMix`) делает линейную проекцию в «плоский» LoRA-вектор:

- `flat: [bs, n_layers, n_modules, r, d_lora]`
- где `d_lora = max(d_in[module] + d_out[module])`

Важно:

- в head уже есть ось `n_layers`, поэтому веса head **слой-специфичные**;
- при `n_modules > 1` веса также модуль-специфичные;
- это функционально эквивалентно идее отдельных `W_A^(l), W_B^(l)`,
  но реализация технически делает один общий head и потом разрезает его выход.

Как из `flat` получаются `A` и `B`:

1. Для каждого module берётся его часть по последней оси.
2. Эта часть режется на:
    - `A: [bs, n_layers, r, d_in[module]]`
    - `B: [bs, n_layers, r, d_out[module]]`
3. Потом применяются обучаемые скейлеры `scaler_A`, `scaler_B`.

Итог пункта 6:

- head **генерирует LoRA A/B на каждый слой и модуль**,
- но делает это через один слой-параметризованный проектор + split, а не двумя независимыми модулями `head_A/head_B`.

Какие матрицы здесь обучаются (в части Perceiver):

- `modality_projection`: `gate_proj`, `up_proj`, `down_proj`
- `latents_q` в каждом Resampler (encoder и decoder)
- в каждом attention-слое:
  - `W_Q` (`q_proj`)
  - `W_K` (`k_proj`)
  - `W_V` (`v_proj`)
  - `W_O` (`o_proj`)
- в каждом Perceiver-слое MLP:
  - `gate_proj`, `up_proj`, `down_proj`
- все `RMSNorm` параметры

После этого в `aggregator.py` выход уже приводится к
`[bs, n_layers, n_modules, r, d_latent]` и уходит в `HyperLoRA head` для генерации матриц `A` и `B`.

---

### `lora_layer.py` — патч LoRA forward pass

**`lora_forward(x, n_qs, tot_q, A, B, ...)`:**
Кастомная forward-функция для линейных слоёв в режиме без sequence packing.
```
base_out = Linear.forward(x)          # стандартный линейный слой
delta_x = einsum(A, x, "n_ctx r d_in, n_ctx s d_in -> n_ctx s r")
delta_x = einsum(B, delta_x, ...) * scaling
return base_out + delta_x
```
`A` имеет форму `[n_ctx, r, d_in]`, `B` — `[n_ctx, r, d_out]`. Каждый элемент батча имеет свою LoRA.

**`lora_forward_packed(x, n_qs, tot_q, seq_lens, tot_len, A, B, ...)`:**
Версия для упакованных последовательностей (sequence packing). Вход имеет форму `[1, tot_len, d_in]`. Используется `repeat_interleave` для развёртки A и B: сначала по числу questions на контекст, затем по длинам последовательностей.

**`apply_lora_to_layers(model, layer_indices, generated_loras, n_qs, position_ids)`:**
Патчит все нужные Linear-модули в указанных слоях через `partial`. Для каждого модуля:
```python
module.forward = partial(module.forward, n_qs=n_qs, tot_q=tot_q, A=A[:, layer_idx], B=B[:, layer_idx])
```
Таким образом при следующем вызове `base_model(question)` каждый Linear автоматически применит свою LoRA.

---

### `lora_merger.py` — механизм чанкинга

**`combine_lora(generated_loras, n_chunks, lora_bias, scalers, bias_scaler)`:**

Реализует формулу конкатенации из статьи:
$$A_l = \begin{bmatrix} A_l^{(1)} \\ \vdots \\ A_l^{(K)} \end{bmatrix}, \quad B_l = \begin{bmatrix} B_l^{(1)} & \cdots & B_l^{(K)} \end{bmatrix}$$

Логика:
1. `n_chunks` — вектор `[n_ctx]`, сколько чанков у каждого контекста в батче.
2. Все чанки сначала сливаются в `[tot_chunks, n_layers, r, dim]`.
3. Для каждого контекста g берётся его срез по оси rank_dim и пишется в `combined[g, :, :combined_rank, :]`.
4. Если есть `lora_bias` (data-independent смещение), он добавляется **после** чанков по оси rank: `combined[g, :, combined_rank:combined_rank+r, :] = bias`.
5. Итоговый ранг: `(n_chunks + 1) * r` (это rank чанков плюс rank bias).

Важно: матрица заполняется нулями до `max_rank_needed`, что позволяет батчить контексты с разным числом чанков.

---

### `ctx_encoder.py` — энкодер контекста

Три варианта энкодера, выбирается через `ctx_encoder_type`:

**`EarlyExit`:**
Берёт первые `layer_idx` слоёв базовой LLM. Выход: `[bs, seq_len, hidden]` — `last_hidden_state`. Не вычисляет градиенты (`@torch.no_grad()`).

**`EmbeddingOnly`:**
Только embedding-слой (первый `hidden_state`). Дешёво, но теряет контекстуальную информацию.

**`PerLayerActivations`** (**используется в main_exp**):
Берёт активации всех слоёв до `ctx_encoder_last_layer - 1`. Использует `output_hidden_states=True`. Выход: `torch.stack(outputs.hidden_states, dim=1)` → форма `[bs, n_layers, seq_len, hidden]`. Передаёт все слои на вход Perceiver (режим `layer_to_layer`).

---

### `modeling/text_to_lora.py` / `text_to_lora_impl.py`

Обёртки для inference-API через сервер (FastAPI). `text_to_lora.py` содержит сервер, `text_to_lora_impl.py` — реализацию. Используются в `demo/app.py`.

### `generative_adapter.py`

Клиент HTTP-сервера для генерации. `GenerativeAdapter` — thin wrapper, который перехватывает `.generate()` вызов и отправляет HTTP-запрос на сервер.

### `context_distillation.py`

Базслайн: классическая дистилляция контекста (без гиперсети). Используется для сравнения в экспериментах.

---

## 6. `src/ctx_to_lora/trainer.py` — тренировочный цикл

### `DistillationTrainer` — **главный тренер для KL-лосса**

Наследуется от `ModulatedModelTrainer → Trainer`.

**`compute_loss(model, inputs, return_outputs, num_items_in_batch)`:**

1. Извлекает `labels`, `logprobs_vals` (log-вероятности токенов от teacher-модели), `logprobs_indices` (индексы top-k токенов).
2. Делает `model.forward(**inputs, return_generated_lora=True)` → получает логиты и сгенерированные LoRA.
3. Вычисляет KL-лосс:
   ```python
   outputs_logits = outputs.logits[label_pos]  # [N, vocab_size]
   logq_full_denom = logsumexp(outputs_logits)
   selected_logits = gather(outputs_logits, indices)  # [N, K] top-K токенов
   logq_selected = selected_logits - logq_full_denom
   loss = -(p * logq_selected).sum(dim=-1)  # KL между teacher и student на top-K токенах
   ```
4. Опционально применяет `per_ctx_loss_kl(...)` — усредняет по контекстам.
5. Добавляет L1-регуляризацию на нормы A и B матриц: `l1_norm = mean(|A| + |B|)`.
6. Логирует `kl_loss` и `gen_lora_l1_norm` в WandB.

**`per_ctx_loss_kl(inputs, labels, loss)`:**
Производит усреднение лосса сначала по токенам каждого Q, потом по Q каждого контекста. Нужно для правильного взвешивания, когда у разных контекстов разное число вопросов.

### `CrossEntropyTrainer`

Как `DistillationTrainer`, но с обычным CE-лоссом. Используется когда нет teacher-логитов (режим `use_kl_loss=False`).

### `train_model(...)`

Определяет тип тренера (`DistillationTrainer` vs `CrossEntropyTrainer`), применяет monkey-patch `get_decay_parameter_names` (исключает normalization layers, biases, и latent queries `latents_q` из weight decay), запускает `trainer.train()`.

**`get_decay_parameter_names`:**
Важно: latent queries Perceiver (`latents_q`) исключены из weight decay — они обучаются без штрафа.

---

## 7. `src/ctx_to_lora/data/` — пайплайн данных

### `processing.py` — основной пайплайн

**`get_tokenized_dataset(ds_name, split, ...)`:**
Главная функция подготовки данных. Делает:
1. `load_and_process_dataset(...)` → загружает датасет, нормализует столбцы (context, prompts, responses)
2. Применяет `preprocessing_fn` — датасет-специфическую нормализацию
3. Токенизирует: каждый контекст нарезается на чанки длиной `max_ctx_chunk_len`, каждый вопрос-ответ склеивается в chat-формат
4. Если `use_kl_loss=True`, рядом с токенами лежат `logprobs_vals` и `logprobs_indices` (teacher logits)

**Чанкинг контекста:**
Контекст разрезается по словесным границам. Вокруг каждого чанка добавляются специальные токены (из `CTX_AFFIXES` в `definitions.py`) — chat-template обёртка. Число чанков сэмплируется из `num_chunk_probs` — так происходит аугментация числом чанков.

**`pack(datasets, max_packed_inp_len, max_packed_ctx_len, ...)`:**
Упаковывает несколько коротких примеров в один длинный (sequence packing). Вызывает `pack_data_points_by_length` для нахождения групп, затем `pack_data_points_FA` для физической упаковки. Токены разных примеров конкатенируются, `position_ids` сбрасываются в 0 на границе каждого примера.

### `packing.py` — sequence packing

**`pack_data_points_by_length(lens, ctx_lens, max_packed_inp_len, max_packed_ctx_len)`:**
Жадный алгоритм: находит максимальный префикс примеров, не выходящий за лимиты длин. Возвращает список пар `(i_start, i_end)`.

**`pack_data_points_FA(batch)`:**
Физически склеивает данные в один numpy-массив. Заполняет `position_ids`: для каждого примера начинается с 0. Flash Attention 2 использует `position_ids` для определения границ примеров — zero-crossing означает новый пример.

### `collator.py` — батч-коллатор

**`flatten_if_not_packed(inp_list)`:**
Единственный коллатор, используемый при обучении. Если данные упакованы (есть `position_ids`) — просто передаёт единственный элемент батча как тензор. Важно: при sequence packing `batch_size=1` обязателен.

**`eval_collator`** / **`generation_collator`:**
Коллаторы для валидации. Паддят последовательности, собирают `ctx_ids` и `ctx_attn_mask`.

### `definitions.py` — константы

- `CTX_AFFIXES` — словарь токенов-обрамлений чанков контекста для каждой модели (Gemma, Mistral, Qwen). Нужны для правильного chat-форматирования каждого чанка.
- `DS_KWARGS` — словарь путей и параметров для каждого датасета (squad, drop, ropes, pwc, и др.)
- `SELF_GEN_DATA_DIR`, `RAW_DATA_DIR`, `TRANSFORMED_DATA_DIR` — пути к директориям данных.

### `preprocessing_fn.py`

Датасет-специфические функции нормализации: приводят разные форматы данных к единому формату `{context, prompts, responses}`.

### `self_gen_template.py` / `q_generation_template.py`

Jinja/f-string шаблоны для форматирования промптов. `QA_PROMPT_TEMPLATE` — шаблон `Context: {context}\nQuestion: {question}` для вопрос-ответного inference.

---

## 8. `src/ctx_to_lora/model_loading.py`

**`get_model(model_name_or_path, train, requires_grad, ...)`:**
Загружает модель через `AutoModelForCausalLM.from_pretrained`. Важные детали:
- `attn_implementation="flash_attention_2"` по умолчанию
- Если указан `peft_config`, оборачивает в `PeftModel`
- `requires_grad=False` для base_model и ctx_encoder — только hypernet обучается
- Для vision-моделей (Gemma-3) берёт `model.language_model`

**`get_lora_config(model_dir, **kwargs)`:**
Создаёт `LoraConfig`. Важно: `lora_alpha = r^(3/2) * 2` — нестандартное значение, выбранное авторами. LoRA применяется только к указанным `target_modules`.

**`get_tokenizer(...)`:**
Загружает токенайзер. Если найден файл `chat_templates/{model_name}.jinja`, применяет его как `chat_template`.

---

## 9. `src/ctx_to_lora/pooling.py`

Простые функции пулинга для варианта агрегатора `"pooler"` (не используется в main_exp, но доступен):
- `mean_pool` — среднее по seq_len
- `max_pool` — максимум
- `last_token_pool` — последний токен (или первый при left padding)

---

## 10. `src/ctx_to_lora/utils.py`

Вспомогательные утилиты:
- `get_layers(model)` — рекурсивно достаёт `model.layers` (работает для вложенных PeftModel)
- `get_peft_modules(layer, peft_config)` — итерирует по LoRA-обёрнутым модулям слоя
- `get_peft_in_out_features(model, peft_config)` — извлекает `d_in` и `d_out` для каждого target_module
- `compile_linear(model)` — компилирует все Linear-слои через `torch.compile`
- `setup_logging` / `save_yaml` / `extract_cli_args` — логирование и сохранение конфигов

---

## 11. `configs/` — YAML-конфиги

### `configs/main_exp/self_gen_lv1_closed_qa_1_l2l.yaml` (Gemma)
Главный конфиг для Gemma-2-2b-it:
```yaml
lora_r: 8
target_modules: [down_proj]
use_kl_loss: true
ctx_encoder_type: per_layer_activations
n_latent_queries: 8
num_blocks: 9
num_self_attn_per_block: 0
gradient_accumulation_steps: 11
max_packed_inp_len: 6144
max_packed_ctx_len: 6144
train_ds_names:
  - self_gen/google/gemma-2-2b-it_.../fw_qa_v2/...*level_1*.parquet
  - self_gen/.../pwc_compact
  - ...
```

Используется с CLI-аргументами из `1-train.sh`:
- `--per_rank_gen=True` — отдельный вектор на каждый rank
- `--per_layer_processing=True` — ResMLPBlockPerLayer после агрегатора
- `--gen_lora_l1_reg_coef=0.1` — L1 регуляризация
- `--max_steps=80000`
- `--quantize_ctx_encoder=True` — 4-bit encoder

### `configs/main_exp/qwen/self_gen_lv1_closed_qa_1_l2l.yaml`
Аналогичный конфиг для Qwen3-4B.

### `configs/niah_exp/ctx_magic_number_32_256.yaml`
Конфиг для Needle-in-a-Haystack эксперимента. Обучение на синтетических задачах поиска числа в контексте.

---

## 12. `data/` — скрипты генерации данных

### `data/self_generate_qa.py`
Генератор синтетических QA-пар. Использует `vLLM` для быстрой генерации.

**Алгоритм:**
1. Загружает контексты из датасета (FineWeb-Edu, PwC, SQuAD и др.)
2. Для каждого контекста генерирует N вопросов через большую модель (`gemma-3-12b-it` или аналог)
3. Для каждого (контекст, вопрос) запускает целевую модель (`gemma-2-2b-it`) с контекстом в промпте
4. Сохраняет `logprobs_vals` и `logprobs_indices` (top-K логиты) — они станут teacher-сигналом при обучении

Ключевые аргументы: `--base_model_name`, `--teacher_model_name`, `--ds_names`, `--closed_qa_prob` (вероятность closed-QA режима), `--temperature`.

### `data/generate_fw_edu_qa_v2.py`
Генерация QA на базе FineWeb-Edu. Параллельная версия с поддержкой batch inference.

### `data/build_{squad,drop,ropes,pwc}_compact.py`
Предобрабатывают соответствующие датасеты: нормализуют формат, отфильтровывают примеры, сохраняют в parquet.

### `data/generate_ctx_magic_number.py`
Генерирует синтетические датасеты `ctx_magic_number` — тексты со вставленным числом в случайном месте. Используется для NIAH-экспериментов.

---

## 13. `scripts/` — скрипты запуска

### `scripts/main_exp/0-download_data.py`
```python
snapshot_download("SakanaAI/self_gen_qa_d2l", repo_type="dataset", ...)
```
Скачивает готовые данные (~100 ГБ на модель) из HuggingFace Hub.

### `scripts/main_exp/1-train.sh`
```bash
uv run accelerate launch --config_file accelerate_config.yaml \
  --num_processes=8 --gpu_ids all train.py \
  configs/main_exp/self_gen_lv1_closed_qa_1_l2l.yaml \
  --model_name_or_path=google/gemma-2-2b-it \
  --target_modules=down_proj --lora_r=8 \
  --per_rank_gen=True --per_layer_processing=True \
  --gen_lora_l1_reg_coef=0.1 --max_steps=80000 \
  --use_kl_loss=True --quantize_ctx_encoder=True
```

### `scripts/main_exp/2-train-chunk.sh`
Второй этап обучения: добавляется аугментация с несколькими чанками (`num_chunk_probs`). Инициализируется с чекпоинта первого этапа.

### `scripts/main_exp/eval/`
Скрипты оценки для каждого метода: `d2l.sh` (D2L), `cd.sh` (context distillation), `base_model.sh` (baseline).

---

## 14. Как запустить: полный пайплайн

### Шаг 1: Создание окружения

```bash
# Требования: Linux, CUDA 12.4, Python 3.10

# Установить uv (менеджер пакетов, быстрее pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # или reload shell

# Клонировать репозиторий и перейти в него
cd /path/to/doc-to-lora

# Запустить install.sh — создаёт venv, ставит все зависимости
bash install.sh
# После этой команды создан .venv/ с Python 3.10, torch, flash-attn и т.д.

# Опционально: авторизоваться в HuggingFace (нужно для Gemma — gated модель)
uv run huggingface-cli login

# Опционально: WandB для логирования
wandb login
export WANDB_PROJECT=doc_to_lora
```

Что делает `install.sh`:
1. Устанавливает `uv` — аналог `pip`/`poetry` на Rust
2. Создаёт venv `.venv` через `uv venv --python 3.10`
3. Ставит PyTorch 2.6 для CUDA 12.4
4. Синхронизирует зависимости из `pyproject.toml` через `uv sync`
5. Ставит flash-attention из precompiled wheel
6. Скачивает SQuAD и строит compact-версии датасетов

### Шаг 2: Скачать обучающие данные

```bash
# Вариант А: скачать готовые данные (~100 ГБ для Gemma, ~328 ГБ для всех трёх моделей)
uv run python scripts/main_exp/0-download_data.py
# Данные сохраняются в data/raw_datasets/self_gen/

# Вариант Б: сгенерировать с нуля (долго, нужны GPU)
# Генерация QA на FineWeb-Edu через vLLM
uv run python data/generate_fw_edu_qa_v2.py \
    --base_model google/gemma-2-2b-it \
    --teacher_model google/gemma-3-12b-it \
    --output_dir data/raw_datasets/self_gen/
```

### Отдельно: как запускать NIAH (если хочешь начать именно с него)

Для NIAH тебе **не нужен main-exp self-gen датасет**. Здесь используется синтетическая задача `ctx_magic_number`: в длинный контекст вставляется «магическое число», и модель должна его восстановить.

Это самый удобный старт для знакомства с пайплайном, потому что:
- данные генерируются локально;
- запуск проще;
- быстрее видно, работает ли гиперсеть и чанкинг;
- легче дебажить архитектурные изменения.

#### Что используется в NIAH-конфиге

Файл: `configs/niah_exp/ctx_magic_number_32_256.yaml`

Обучение идёт на двух синтетических диапазонах длины:
- `ctx_magic_number_32_128`
- `ctx_magic_number_128_256`

То есть гиперсеть обучается на коротких контекстах, а потом проверяется на гораздо более длинных. Именно это и демонстрирует ключевую идею статьи: **обобщение на длины, которых не было на обучении**.

#### Шаг N1: сгенерировать данные для NIAH

Ты `install.sh` уже выполнил, поэтому дальше просто из корня репозитория:

```bash
uv run bash scripts/niah/0-gen_data.sh
```

Этот скрипт запускает:

```bash
uv run data/generate_ctx_magic_number.py
```

Что он делает:
- создаёт наборы `ctx_magic_number_*` в `data/raw_datasets/`;
- готовит train / val / test диапазоны разных длин;
- эти датасеты потом автоматически подхватываются через `DS_KWARGS` из `src/ctx_to_lora/data/definitions.py`.

#### Шаг N2: запустить обучение NIAH

```bash
uv run bash scripts/niah/1-train.sh
```

Этот запуск отличается от main exp:
- используется `ctx_encoder_type=early_exit`, а не `per_layer_activations`;
- используется `use_kl_loss=False`, то есть обучение идёт по **cross-entropy**, а не по teacher-logits;
- включён чанкинг уже во время обучения через `num_chunk_probs`;
- train/val данные синтетические и маленькие по сравнению с main exp.

Фактически `scripts/niah/1-train.sh` запускает примерно такой режим:

```bash
uv run train.py \
  configs/niah_exp/ctx_magic_number_32_256.yaml \
  --model_name_or_path=google/gemma-2-2b-it \
  --target_modules=down_proj \
  --aggregator_type=perceiver \
  --num_blocks=8 \
  --n_latent_queries=208 \
  --lora_r=8 \
  --per_rank_gen=True \
  --per_layer_processing=True \
  --ctx_encoder_type=early_exit \
  --use_kl_loss=False \
  --max_ctx_chunk_len=512 \
  --num_chunk_probs='{"1":"0.5", "2":"0.125", "3":"0.0625", "4":"0.0625", "5":"0.0625", "6":"0.0625", "7":"0.0625", "8":"0.0625"}'
```

#### Подробный разбор параметров `scripts/niah/1-train.sh` (и сравнение с `main_exp/1-train.sh`)

Ниже — смысл каждого параметра из NIAH-запуска:

- `--model_name_or_path=google/gemma-2-2b-it` — базовая LLM (26 слоёв).
- `--target_modules=down_proj` — LoRA вешается только на MLP `down_proj` каждого слоя.
- `--lora_r=8` — базовый LoRA rank для одного контекстного чанка.
- `--aggregator_type=perceiver` — агрегатор гиперсети = Perceiver (не pooler).
- `--num_blocks=8` — число блоков Resampler в encoder-Perceiver.
- `--num_self_attn_per_block=0` — внутри каждого блока нет self-attention на латентах (только cross-attention).
- `--n_latent_queries=208` — число latent queries encoder-части Perceiver.
- `--per_rank_gen=True` — генерируются отдельные латенты на каждый rank (а не один латент на модуль).
- `--per_layer_processing=True` — после агрегатора включается `ResMLPBlockPerLayer` (разные веса MLP на каждый слой LLM).
- `--num_pre_head_layers=1` — один pre-head residual MLP блок перед `head`.
- `--ctx_encoder_type=early_exit` — вход гиперсети: `[bs, seq_len, hidden]` (а не пер-слойные активации).
- `--use_kl_loss=False` — `CrossEntropyTrainer` (без teacher KL-дистилляции).
- `--gen_lora_l1_reg_coef=1.5` — сильная L1-регуляризация на сгенерированные LoRA-веса.
- `--use_sequence_packing=True`, `--max_packed_inp_len=4096`, `--max_packed_ctx_len=4096` — packing для ускорения и экономии памяти.
- `--max_ctx_chunk_len=512`, `--min_ctx_chunk_len=25` — ограничения на длину контекстных чанков.
- `--gradient_accumulation_steps=16`, `--per_device_train_batch_size=-1` — effective batch через accumulation, размер микро-батча подбирается пайплайном.

##### Что делает `per_layer_processing=True` (и что НЕ делает)

После агрегатора получаем латенты вида
$Z \in \mathbb{R}^{B \times L \times M_{mod} \times r \times d_{latent}}$,
где $L$ — число слоёв целевой LLM.

`per_layer_processing=True` означает:

- используется `ResMLPBlockPerLayer`;
- для каждого слоя $\ell \in [1..L]$ свои веса MLP;
- математически это семейство функций $f_\ell(\cdot)$:
$$
Z'_{\ell} = Z_{\ell} + f_{\ell}(Z_{\ell}).
$$

Важно: этот флаг **не увеличивает количество Z-токенов**.
Количество латентных слотов задаётся агрегатором (Perceiver + reshape), а `per_layer_processing`
меняет только преобразование уже полученных латентов перед `head`.

##### Отличия NIAH 1-train от main_exp 1-train

- **Распределение вычислений**: NIAH запускается на 1 GPU, `main_exp/1-train.sh` — через `accelerate` на 8 GPU.
- **Тип входа гиперсети**: NIAH `early_exit`; main exp `per_layer_activations`.
- **Perceiver queries**: NIAH `n_latent_queries=208`; main exp (Gemma l2l) `n_latent_queries=8`.
- **Глубина Perceiver**: NIAH `num_blocks=8`; main exp `num_blocks=9`.
- **Loss**: NIAH `use_kl_loss=False` (CE); main exp `use_kl_loss=True` (дистилляция + teacher logprobs).
- **Регуляризация LoRA**: NIAH `gen_lora_l1_reg_coef=1.5`; main exp `0.1`.
- **Тренировочные данные**: NIAH — синтетика `ctx_magic_number_*`; main exp — self-generated QA + compact benchmark datasets.
- **Цель эксперимента**: NIAH — отладка/проверка работы гиперсети на контролируемой задаче; main exp — полноразмерное обучение и сравнение с baseline-методами.

#### Что важно понимать про этот NIAH training setup

1. **Это не тот же режим, что в main exp.**  
    В main exp гиперсеть учится имитировать teacher через KL-loss на self-generated QA. В NIAH — это более «чистая» supervised задача на восстановление информации из контекста.

2. **Контекстный энкодер проще.**  
    `early_exit` берёт ранние слои модели и отдаёт один набор признаков `[bs, seq_len, hidden]`. Это дешевле и проще, чем `per_layer_activations`.

3. **Чанкинг встроен прямо в обучение.**  
    Параметр `num_chunk_probs` учит модель работать не только с одним чанком, а сразу с несколькими. Это важно для последнего zero-shot обобщения на очень длинные контексты.

4. **Оценка идёт на длинах сильно больше train.**  
    Train: до 256 токенов. Eval: вплоть до `131072` токенов в готовом скрипте.

#### Шаг N3: запустить оценку NIAH

После обучения запусти:

```bash
CHECKPOINT_PATH=train_outputs/runs/<твой_run>/pytorch_model.bin \
uv run bash scripts/niah/2-eval.sh
```

Если хочешь прогнать только сокращённый набор длин, используй:

```bash
CHECKPOINT_PATH=train_outputs/runs/<твой_run>/pytorch_model.bin \
uv run bash scripts/niah/2-eval-test.sh
```

Смысл двух скриптов:
- `scripts/niah/2-eval.sh` — длинный полный прогон по множеству диапазонов вплоть до очень больших длин;
- `scripts/niah/2-eval-test.sh` — короткий smoke-test на нескольких диапазонах.

#### Практический порядок действий для тебя сейчас

Раз ты уже сделал `bash install.sh`, дальше минимальный маршрут такой:

```bash
uv run bash scripts/niah/0-gen_data.sh
uv run bash scripts/niah/1-train.sh
CHECKPOINT_PATH=train_outputs/runs/<run_name>/pytorch_model.bin uv run bash scripts/niah/2-eval-test.sh
```

А затем уже полный eval:

```bash
CHECKPOINT_PATH=train_outputs/runs/<run_name>/pytorch_model.bin uv run bash scripts/niah/2-eval.sh
```

#### Где смотреть результаты NIAH

- чекпоинты: `train_outputs/runs/<run_name>/`
- аргументы запуска: `train_outputs/runs/<run_name>/args.yaml`
- CLI аргументы: `train_outputs/runs/<run_name>/cli_args.yaml`
- debug-лог: `train_outputs/runs/<run_name>/debug.log`

#### Если захочешь модифицировать архитектуру именно под NIAH

Самые полезные ручки для первых экспериментов:
- `--num_blocks` — глубина Perceiver;
- `--n_latent_queries` — число latent queries;
- `--lora_r` — базовый ранг LoRA;
- `--per_layer_processing=True/False` — включать ли отдельный post-MLP на каждый слой;
- `--ctx_encoder_type=early_exit` vs `per_layer_activations` — самый важный архитектурный выбор на входе гиперсети;
- `--max_ctx_chunk_len` и `--num_chunk_probs` — поведение на длинных контекстах.

Если цель — сначала быстро проверить свою модификацию гиперсети, **начинать с NIAH действительно правильнее, чем с main exp**.

### Шаг 3: Этап 1 — обучение на 1 чанке

```bash
# Запуск на 8 GPU через Accelerate
uv run bash scripts/main_exp/1-train.sh

# Или вручную с кастомными параметрами:
uv run accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_port 29051 \
    --num_processes=8 \
    --gpu_ids all \
    train.py configs/main_exp/self_gen_lv1_closed_qa_1_l2l.yaml \
    --model_name_or_path=google/gemma-2-2b-it \
    --target_modules=down_proj \
    --lora_r=8 \
    --per_rank_gen=True \
    --per_layer_processing=True \
    --gen_lora_l1_reg_coef=0.1 \
    --max_steps=80000 \
    --use_kl_loss=True \
    --quantize_ctx_encoder=True \
    --use_per_ctx_average_loss=True \
    --gradient_accumulation_steps=8

# Результаты сохраняются в train_outputs/runs/{run_name}/
```

**Для одной GPU (не официальный режим, но работает):**
```bash
uv run python train.py configs/main_exp/self_gen_lv1_closed_qa_1_l2l.yaml \
    --model_name_or_path=google/gemma-2-2b-it \
    --target_modules=down_proj --lora_r=8 \
    --per_rank_gen=True --per_layer_processing=True \
    --max_steps=80000 --use_kl_loss=True
```

### Шаг 4: Этап 2 — обучение с чанкингом

```bash
# Продолжение с чекпоинта, добавление аугментации чанками
uv run bash scripts/main_exp/2-train-chunk.sh
# В скрипте задаётся --from_pretrained_checkpoint=train_outputs/runs/{run_name}/pytorch_model.bin
# и добавляются --num_chunk_probs='{"1": 0.5, "2": 0.3, "4": 0.2}'
```

### Шаг 5: Оценка

```bash
# D2L оценка
uv run bash scripts/main_exp/eval/d2l.sh

# Базовая модель (без контекста)
uv run bash scripts/main_exp/eval/base_model.sh

# Или через run_eval.py напрямую
uv run python run_eval.py \
    --from_pretrained_checkpoint=train_outputs/runs/{run_name}/pytorch_model.bin \
    --test_ds_names=squad --max_test_samples_per_ds=500
```

### Шаг 6: Qwen / Mistral

```bash
# Для Qwen3-4B аналогично, но другой конфиг
uv run accelerate launch --config_file accelerate_config.yaml \
    --num_processes=8 --gpu_ids all \
    train.py configs/main_exp/qwen/self_gen_lv1_closed_qa_1_l2l.yaml \
    --model_name_or_path=Qwen/Qwen3-4B-Instruct-2507 \
    --target_modules=down_proj --lora_r=8 \
    --per_rank_gen=True --per_layer_processing=True \
    --use_kl_loss=True
```

---

## 15. Где что менять для модификации архитектуры гиперсети

Все ключевые точки для экспериментов:

| Что хочешь изменить | Файл | Что трогать |
|---|---|---|
| Архитектура Perceiver (глубина, ширина) | `aggregator.py`, `idefics2.py` | `n_latent_queries`, `num_blocks`, `num_self_attn_per_block`, `hidden_size` в `Idefics2PerceiverConfig` |
| Добавить новый тип агрегатора | `aggregator.py` | Новый класс, добавить в `AGGREGATOR_CLS` |
| Изменить структуру pre-head слоёв | `hypernet.py`, `HyperLoRA._init_model` | `ResMLPBlock`, `ResMLPBlockPerLayer`, `num_pre_head_layers` |
| Изменить head (как latent → LoRA матрица) | `hypernet.py`, `HyperLoRA._init_model` | `self.head` (EinMix) |
| Изменить инициализацию весов | `hypernet.py`, `ModulatedPretrainedModel._bias_hyper_init` | `nn.init.*` на `head.weight` |
| Изменить механизм чанкинга | `lora_merger.py`, `combine_lora` | Логику конкатенации A и B |
| Изменить как LoRA применяется к слоям | `lora_layer.py`, `lora_forward` / `lora_forward_packed` | Формула применения delta_x |
| Изменить функцию потерь | `trainer.py`, `DistillationTrainer.compute_loss` | KL-лосс, регуляризация |
| Изменить энкодер контекста | `ctx_encoder.py` | Новый класс, добавить в `CTX_ENCODER_CLS` |
| Изменить гиперпараметры через конфиг | `configs.py` | `HypernetArguments`, `AggregatorArguments` |
| Добавить новый target_module LoRA | YAML конфиг | `target_modules: [down_proj, up_proj]` |
| Изменить ранг LoRA | YAML конфиг / CLI | `lora_r=16` |

### Пример: добавить новый тип агрегатора (например, Cross-Attention с CLS-токеном)

1. В `aggregator.py` написать новый класс, например `CLS_Aggregator(nn.Module)`.
2. Добавить в `AGGREGATOR_TYPE` enum: `CLS = "cls"`.
3. Добавить в `AGGREGATOR_CLS`: `AGGREGATOR_TYPE.CLS: CLS_Aggregator`.
4. Убедиться, что `AggregatorConfig` содержит все нужные поля (или добавить новые в `AggregatorArguments` в `configs.py`).
5. В `configs/` создать YAML с `aggregator_type: cls`.

### Пример: изменить Perceiver — добавить self-attention внутри блоков

В `aggregator.py` при создании `Idefics2PerceiverConfig` передать `num_self_attn_per_block > 0`, или в YAML задать `num_self_attn_per_block: 2`.

### Пример: изменить head — вместо линейной проекции использовать MLP

В `hypernet.py`, `HyperLoRA._init_model`, заменить:
```python
self.head = Mix("bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora", ...)
```
на кастомный `nn.Module` с MLP-структурой. Нужно учесть, что `head` должен обрабатывать batch-размерности через `EinMix` или `einops`.
