# Инструкция по установке и запуску Document Validator

## Требования

### Система

- macOS (arm64)
- Python 3.8+
- Qt 5
- CMake 3.14+

## Установка

### 1. Установка системных зависимостей

```bash
# Homebrew пакеты
brew install qt@5 cmake nlohmann-json

# Добавить Qt в PATH
export PATH="/opt/homebrew/opt/qt@5/bin:$PATH"
```

### 2. Установка Python зависимостей

```bash
# Создать виртуальное окружение (рекомендуется)
python3 -m venv venv
source venv/bin/activate

# Установить пакеты
pip install -r requirements.txt
```

### 3. Конвертация моделей

Если у вас есть `.pth` файлы, но нет `.pt` файлов для моделей реалистичности:

```bash
python3 convert_models.py
```

Это создаст:

- `realism_model_1.pt` из `validator_visual_1.pth` (если существует `.pt`)
- `realism_model_2.pt` из `validator_visual_2.pth`
- `realism_model_3.pt` из `validator_visual_3.pth`

**Важно:** OCR модель (`ocr_validator_model.pth`) остается в формате `.pth` и используется напрямую Python скриптом.

### 4. Сборка приложения

```bash
# Очистка и пересборка
rm -rf build
mkdir build
cd build

# Конфигурация CMake
cmake ..

# Сборка
make -j4
```

### 5. Запуск

```bash
# Из директории build
./maboy.app/Contents/MacOS/maboy

# Или через open (macOS)
open maboy.app
```

## Структура моделей

Проект требует следующие файлы моделей в корне:

### Модели реалистичности (ConvNeXt Tiny)

- `realism_model_1.pt` (TorchScript) или `validator_visual_1.pth` (state_dict)
- `realism_model_2.pt` или `validator_visual_2.pth`
- `realism_model_3.pt` или `validator_visual_3.pth`

### OCR модель (MLP + TF-IDF)

- `ocr_validator_model.pth` (PyTorch state_dict)
- `tfidf_vectorizer.pkl` (опционально, если обучен отдельно)

### EAST детектор (опционально для doc_orc.py)

- `frozen_east_text_detection.pb`

## Как работает валидация

1. **C++ GUI** (`maboy.app`) принимает файл от пользователя
2. **DocumentValidator** (C++) вызывает `validator.py` через subprocess
3. **validator.py** (Python):
   - Загружает 3 ConvNeXt модели реалистичности
   - Загружает MLP модель для OCR проверки
   - Запускает EasyOCR для извлечения текста
   - Векторизует текст через TF-IDF
   - Прогоняет через все модели
   - Возвращает JSON результат
4. **DocumentValidator** парсит JSON и отображает результат в GUI

## Устранение неполадок

### nlohmann/json не найдена

```bash
brew install nlohmann-json
# Или укажите путь в CMakeLists.txt
```

### Python не найден

Убедитесь что `python3` доступен:

```bash
which python3
```

### EasyOCR медленно загружается

При первом запуске EasyOCR скачивает модели (~100MB). Это нормально.

### Модели не найдены

Проверьте что все `.pt` и `.pth` файлы находятся в корне проекта:

```bash
ls -lh *.pt *.pth
```

## Разработка

### Тестирование Python валидатора отдельно

```bash
python3 validator.py /path/to/document.jpg --project-root .
```

### Просмотр логов

Python скрипт выводит логи в stderr, финальный JSON в stdout. В C++ коде все логи сохраняются.

### Изменение моделей

Если вы хотите использовать другие модели:

1. Обновите `validator.py` (архитектуру и загрузку)
2. Обновите пути в `loadModels()`
3. Пересоберите проект
