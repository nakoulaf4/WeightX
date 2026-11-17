# Архитектура WeightX: глубокое техническое описание

Этот документ описывает внутреннее устройство харнесса **WeightX**, математические алгоритмы редактирования и защиты весов, а также механизмы изоляции и верификации.

---

## 1. Обзор трехслойной изоляции

Ключевой принцип **WeightX** — **разделение привилегий между планирующей LLM («мозгом») и исполняющей средой**:

1. **Provider Layer (`src/weightx/providers/`)** отвечает исключительно за транспорт сообщений и преобразование JSON-схем инструментов в формат конкретного API или CLI-агента.
2. **Tool Layer (`src/weightx/tools/`)** экспортирует строго типизированные дескрипторы инструментов (`ToolDefinition`). Модель видит только имена тензоров, агрегированные профили и запрошенные срезы в формате **TOON**. Она не имеет доступа к объекту `torch.nn.Module`, файловой системе хоста или отложенному тестовому набору (`held_out_split`).
3. **Execution & Verification Layer (`src/weightx/execution/`, `src/weightx/safety/`)** применяет изменения в транзакционном рабочем пространстве `WeightWorkspace`, защищает важные веса через диагональ Фишера (**IWCKEdit**), исполняет код в изолированном процессе (**Sandbox**) и верифицирует результат на скрытых тестах (**Vespasian**).

---

## 2. Экономия контекста через формат TOON (`src/weightx/toon_io.py`)

Таблицы тензоров и метрики слоев содержат десятки однотипных объектов. В JSON ключи (`name`, `shape`, `dtype`, `norm_l2`, `mean`, `std`) повторяются в каждой строке.

**WeightX** сериализует все ответы инструментов в формат **TOON (Token-Oriented Object Notation)**:
```text
tensors[38]{name,layer,role,shape,dtype,numel,norm_l2,mean,std,max_abs}:
  model.embed_tokens.weight,-1,embed,[128, 64],float32,8192,90.4152,-0.00211,0.99873,3.8412
  model.layers.0.input_layernorm.weight,0,norm,[64],float32,64,8.0,1.0,0.0,1.0
  ...
```
На реальной таблице тензоров модели формат TOON сокращает объем текста и количество токенов на **58.3%** по сравнению с эквивалентным JSON.

---

## 3. Уровень L1: Понимание архитектуры (`src/weightx/l1_arch/autoarch.py`)

Модуль `inspect_architecture` интегрирует движок **AutoArch** из библиотеки `omnia-surgery` (MIT) и расширяет его поддержкой произвольных `torch.nn.Module` и `.safetensors`:

- **Классификация семейств**: автоматически распознаются `llama`, `mistral`, `qwen`, `gemma`, `gpt2`, `gpt_neox`, `falcon`, `phi`, `bert` и др.
- **Семантические роли тензоров**: каждый тензор размечается ролью (`embed`, `q`, `k`, `v`, `qkv`, `o`, `gate`, `up`, `down`, `mlp_fc`, `mlp_proj`, `norm`, `lm_head`).
- **Калибровочное разрешение неоднозначностей**: если по одним только формам матриц `[d_model, d_model]` невозможно однозначно определить число голов внимания `n_heads` или базу `rope_theta`, `AutoArch` перебирает кандидатов на калибровочном батче и выбирает конфигурацию с минимальным **Negative Log-Likelihood (NLL)**.

---

## 4. Уровень L2: Профилирование слоев и карта знаний (`src/weightx/l2_profile/`)

### 4.1. 13 метрик Uni-Layer и консенсус Борда (`unilayer.py`)

Для каждого трансформерного блока вычисляются 13 метрик в 4 категориях:
1. **Спектральные (SVD)**: `effective_rank` ($\exp(H(\sigma))$), `stable_rank` ($\|W\|_F^2 / \sigma_{\max}^2$), `condition_number` ($\sigma_{\max}/\sigma_{\min}$).
2. **Статистические (веса)**: `weight_l2_norm`, `weight_sparsity`, `weight_outlier_ratio` (доля элементов $> 3\sigma$), `weight_kurtosis` (избыточный эксцесс).
3. **Активационные**: `block_influence` ($1 - \cos(h_{\text{in}}, h_{\text{out}})$), `activation_entropy`, `neuron_outlier_ratio`.
4. **Градиентно-атрибутивные**: `fisher_information`, `gradient_norm`, `wanda_score` ($\|W_{ij}\| \cdot \|X_j\|_2$).

Все метрики агрегируются методом **Borda Count** в единый балл важности слоя `consensus_importance` $\in [0, 1]$ и автоматически выявляют:
- **Бутылочные слои (`bottleneck_layers`)** — слои с максимальной концентрацией сигнала или аномальными нейронами-выбросами (`outlier_neurons`).
- **Избыточные слои (`redundant_layers`)** — кандидаты на сжатие или прунинг.

### 4.2. Карта знаний и Индекс концентрации знаний (`knowledge_map.py`)

Вместо пакета `egora-diagnostics` (имеющего коммерчески несовместимую лицензию AGPLv3 + патентную оговорку) в **WeightX** реализован чистый GPLv3-движок `compute_knowledge_map`:
- **Нормализованный Logit Lens**: к промежуточному состоянию $h^{(l)}$ каждого слоя применяется финальная нормализация модели (`RMSNorm` / `LayerNorm`) перед проекцией через `lm_head`, что устраняет систематическое смещение норм на ранних слоях.
- **Knowledge Concentration Index (KCI)**: вычисляется как нормированный коэффициент Джини распределения вклада слоев $\Delta p_l$:
  $$\text{KCI} = \frac{1}{L - 1} \left( L + 1 - 2 \frac{\sum_{i=1}^{L} (L + 1 - i) w_{(i)}}{\sum_{i=1}^{L} w_{(i)}} \right) \in [0, 1]$$
  Высокий $\text{KCI}$ означает, что знание узко локализовано в 1–2 слоях (идеально для точечной правки); низкий $\text{KCI}$ указывает на распределенное знание (требуется многослойный `EAMIT`).

---

## 5. Уровень L3: Декларативные YAML-планы и пакетное редактирование

### 5.1. Декларативный движок BrainSurgery-WX (`src/weightx/declarative/plan_engine.py`)

Агенты генерируют декларативные планы правок в формате YAML:
```yaml
transforms:
  - assert:
      all:
        - exists: { of: "model.layers.2.mlp.down_proj.weight" }
        - shape: { of: "model.layers.2.mlp.down_proj.weight", is: [64, 172] }
        - dtype: { of: "model.layers.2.mlp.down_proj.weight", is: "float32" }
        - finite: { of: "model.layers.2.mlp.down_proj.weight" }
  - rescale_outlier_rows_:
      target: "model.layers.2.mlp.down_proj.weight"
      threshold_factor: 2.5
  - assert:
      finite: { of: "model.layers.2.mlp.down_proj.weight" }
```
**Гарантии безопасности BrainSurgery-WX:**
- Любой план **обязан** содержать минимум один блок `assert`.
- Все операции из исходного `brainsurgery`, обращающиеся к диску (`load`, `save`, `dump`, `execute`), жестко заблокированы.
- Если любой `assert` до или после трансформации завершается ошибкой, все изменения плана **атомарно откатываются**.

### 5.2. Пакетное редактирование EAMIT (`src/weightx/l3_weights/eamit.py`, `rust/crates/weightx-core/src/eamit.rs`)

Для одновременной записи батча из $B$ фактов $(K \in \mathbb{R}^{d_{\text{in}} \times B}, V^* \in \mathbb{R}^{d_{\text{out}} \times B})$ в набор слоев $S = \{l_1, \dots, l_M\}$ сначала выполняется **выравнивание ключевых и остаточных эмбеддингов** (по алгоритму EAMET, arXiv:2505.11876) для предотвращения коллизий представлений при масштабных правках:

1. Строятся матрицы попарных косинусных сходств $S_k^{(ij)} = \cos(k_i, k_j)$ и $S_r^{(ij)} = \cos(r_i, r_j)$ и распределения $P_k^{(i)}, P_r^{(i)} = \text{softmax}(S_{\cdot}^{(i, j \neq i)} / \tau)$.
2. Исходная невязка $R_{\text{raw}} = V^* - \sum_{l \in S} W_l K$ оптимизируется по целевому функционалу EAMIT:
   $$\mathcal{L}(R) = \mathcal{L}_{\text{fid}}(R, R_{\text{raw}}) + \lambda_{\text{KL}} \, \text{KL}(P_r \parallel P_k) + \frac{\lambda_{\text{MSE}}}{B M_{\text{top}}} \sum_{i=1}^B \sum_{j \in \text{Top-}M(i)} \left(S_r^{(ij)} - S_k^{(ij)}\right)^2$$
3. Выровненная невязка $R$ равномерно распределяется по целевым слоям с регуляризацией ковариации ключей и защитой Фишера:
   $$\Delta W_l = \frac{R_l}{M - idx(l)} K_l^\top \left( K_l K_l^\top + \lambda_{\text{cov}} I + \text{diag}(F_l) \right)^{-1}$$

Это позволяет редактировать сотни и тысячи ассоциаций (включая факты с общим субъектом и близкими ключами) за один матричный шаг без коллапса представлений и интерференции.

---

## 6. Защита весов, изоляция и верификация

### 6.1. Защита критических весов IWCKEdit (`src/weightx/safety/iwck_fisher.py`)

Перед применением поправки $\Delta W = W_{\text{proposed}} - W_{\text{orig}}$ вычисляется диагональная эмпирическая матрица информации Фишера на калибровочном наборе:
$$F_i = \frac{1}{|D_{\text{cal}}|} \sum_{x \in D_{\text{cal}}} \left( \frac{\partial \mathcal{L}(x)}{\partial \theta_i} \right)^2, \qquad \hat{F}_i = \frac{F_i}{\text{median}(F) + \varepsilon}$$
Поправка демпфируется обратно пропорционально важности веса и обнуляется для top-$p$ критических параметров (с автоматическим исключением для ремонта нейронов-выбросов):
$$\Delta W_{\text{safe}} = M_{\text{freeze}} \odot \frac{\Delta W}{1 + \alpha \hat{F}}$$

### 6.2. Двухуровневая песочница (`src/weightx/execution/sandbox.py`)

Если LLM использует инструмент `l3_sandbox_exec` для аналитического вычисления тензорной поправки на Python:
1. **Статический AST-фильтр** блокирует `import`, `open`, `exec`, `eval`, `compile`, `__subclasses__`, `__globals__`, `os`, `sys`, `subprocess`, `socket`.
2. **Изолированный субпроцесс** запускает `python -I -S` (на Linux поддерживается обертка в `bwrap --unshare-all --unshare-net`) с очищенным окружением, отключенными сокетами, жестким лимитом памяти и таймаутом. Обмен данными происходит только через сериализованный массив `safetensors` во временной директории.

### 6.3. Верификация и автоматический откат Vespasian (`src/weightx/execution/vespasian.py`)

После каждой правки движок `VespasianVerifier` оценивает модель на скрытом наборе `held_out_split`:
- Проверяется отсутствие `NaN`/`Inf` во всех тензорах.
- Вычисляется комбинированная метрика здоровья $S = 0.7 \cdot \text{Accuracy}_{\text{held-out}} + 0.3 \cdot \exp(-0.25 \cdot \text{NLL}_{\text{held-out}})$.
- Вычисляется $\text{KL}(P_{\text{base}} \parallel P_{\text{candidate}})$ на референсном наборе для предотвращения деградации общих языковых способностей.
- Если $S \ge S_{\text{best}} + \tau$ и $\text{KL} \le \text{KL}_{\max}$, чекпоинт сохраняется как новый `vespasian_best`. В противном случае выполняется **мгновенный автоматический откат** весов к `vespasian_best`.
