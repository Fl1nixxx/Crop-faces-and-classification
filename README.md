# Crop-faces-and-classification

### Веб-приложение на Streamlit для обнаружения лиц на изображениях и их последующей классификации.

Поддерживает три режима работы:
- Автоматический режим
- Ручная разметка
- Клик по участку фото

## Модель

Для классификации используется простая сеть Resnet101 

Для автоматического поиска лиц используется MTCNN

## Структура проекта

```text
Crop-faces-and-classification/
├── app.py
├── requirements.txt
├── runtime.txt
│
└── src/
    ├── __init__.py
    ├── model.py
    ├── inference.py
    ├── face_crop.py
    └── ui.py
