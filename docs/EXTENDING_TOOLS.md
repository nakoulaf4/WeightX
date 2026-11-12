# Руководство разработчика: расширение инструментов, YAML-операций и провайдеров WeightX

Архитектура **WeightX** спроектирована так, чтобы исследователи могли добавлять собственные диагностические метрики, хирургические операции над тензорами и адаптеры LLM без изменения ядра харнесса.

---

## 1. Добавление нового инструмента в Tool Layer (`@weightx_tool`)

Все инструменты, доступные большой языковой модели, декорируются через `@weightx_tool` и автоматически получают JSON-схему и сериализацию вывода в формат **TOON**.

### Пример создания собственного инструмента спектральной диагностики

```python
from typing import Any
import torch
from weightx.tools.registry import ToolRegistry, weightx_tool
from weightx.l3_weights.workspace import WeightWorkspace


def register_custom_spectral_tool(registry: ToolRegistry, workspace: WeightWorkspace) -> None:
    @weightx_tool(
        "l2_spectral_entropy_gap",
        "Вычисляет спектральный зазор (отношение sigma_1 / sigma_2) для выбранного матричного тензора.",
        tensor_name={"type": "string", "description": "Точное имя 2D-тензора весов."},
        required=["tensor_name"],
    )
    def l2_spectral_entropy_gap(tensor_name: str) -> dict[str, Any]:
        t = workspace.get_tensor(tensor_name).detach().float()
        if t.ndim != 2:
            return {"error": f"Tensor '{tensor_name}' must be 2D, got {list(t.shape)}"}
        s = torch.linalg.svdvals(t)
        gap = float((s[0] / (s[1] + 1e-12)).item()) if s.numel() > 1 else 1.0
        return {
            "tensor_name": tensor_name,
            "sigma_1": round(float(s[0].item()), 6),
            "sigma_2": round(float(s[1].item()), 6) if s.numel() > 1 else 0.0,
            "spectral_gap_ratio": round(gap, 4),
        }

    registry.register(l2_spectral_entropy_gap)
```

После вызова `register_custom_spectral_tool(harness.registry, harness.workspace)` инструмент сразу появляется в списке JSON-схем для всех API и CLI провайдеров, а его результат автоматически кодируется в **TOON**.

---

## 2. Добавление новой декларативной операции в BrainSurgery-WX

Декларативный YAML-движок находится в `src/weightx/declarative/plan_engine.py`. Чтобы добавить новую трансформацию (например, спектральное клиппирование сингулярных чисел `spectral_clip_`):

1. Добавьте имя операции в множество `_ALLOWED_TRANSFORMS` в `src/weightx/declarative/plan_engine.py`.
2. Реализуйте обработчик внутри `_apply_transform`:

```python
if op_name == "spectral_clip_":
    max_sigma = float(spec["max_sigma"])
    for name in targets:
        mat = working_tensors[name].float()
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        s_clipped = torch.clamp(s, max=max_sigma)
        working_tensors[name] = (u @ torch.diag(s_clipped) @ vh).to(working_tensors[name].dtype)
    return len(targets)
```

Теперь LLM сможет безопасно использовать `spectral_clip_` в YAML-планах вместе с инвариантами `assert`:

```yaml
transforms:
  - assert:
      shape: { of: "model.layers.2.mlp.down_proj.weight", is: [64, 172] }
  - spectral_clip_:
      target: "model.layers.2.mlp.down_proj.weight"
      max_sigma: 12.0
  - assert:
      finite: { of: "model.layers.2.mlp.down_proj.weight" }
```

---

## 3. Подключение собственного LLM-провайдера (`LLMProvider`)

Вы можете передать любую функцию или класс в `WeightXHarness`:

### Вариант А: Через `CustomCallableProvider` (быстрый способ)

```python
from weightx.providers import CustomCallableProvider, LLMMessage, ToolDefinition

def my_llm_bridge(messages: list[LLMMessage], tools: list[ToolDefinition]) -> dict:
    # Вызов вашего внутреннего корпоративного SDK или локального движка
    return {
        "content": "Выполняю разведку архитектуры.",
        "tool_calls": [
            {"id": "call_1", "name": "l1_autoarch_inspect", "arguments": {"resolve_with_calibration": True}}
        ],
    }

provider = CustomCallableProvider(my_llm_bridge, name="corp-internal-llm")
```

### Вариант Б: Наследование от базового класса `LLMProvider`

```python
from weightx.providers.base import LLMProvider, LLMMessage, LLMResponse, ToolDefinition

class MyCustomProvider(LLMProvider):
    provider_name = "my_custom_provider"

    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...
```
