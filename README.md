# CycleGAN · Apple ↔ Orange

Streamlit-демо для CycleGAN, обученного на датасете apple2orange.

## Структура репозитория

```
├── app.py            # Streamlit-приложение
├── model.py          # Архитектура Generator + load_generators()
├── generators.pt     # Веса двух генераторов (~87 MB)
└── requirements.txt
```

## Запуск локально

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Архитектура модели

- **Генераторы**: ResNet-9 (9 residual blocks, InstanceNorm, ReflectionPad)
- **Дискриминаторы**: PatchGAN 70×70
- **Датасет**: apple2orange (unpaired)
- **Обучение**: 200 эпох, Adam β₁=0.5, lr=2e-4 → линейный спад с эпохи 100
- **Loss**: LSGAN + Cycle Consistency (λ=10)
