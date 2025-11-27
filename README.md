# Document Validator

Professional document validation system using deep learning. Combines C++ GUI (Qt5) with Python ML backend.

## Features

- 3 ConvNeXt realism models with majority voting
- OCR field validation using MLP classifier with TF-IDF vectorization
- Modern Qt5 GUI with drag & drop support
- Hybrid architecture: C++ interface, Python ML backend

## Architecture

```
┌─────────────────┐
│   Qt5 GUI       │  C++ (macOS app)
│   (maboy.app)   │
└────────┬────────┘
         │ subprocess
         ▼
┌─────────────────┐
│  validator.py   │  Python (ML backend)
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
 ConvNeXt  ConvNeXt  ConvNeXt    MLP+TF-IDF
 Model 1   Model 2   Model 3    OCR Model
```

## System Requirements

- macOS (arm64)
- Python 3.8+
- Homebrew

## Installation

### 1. Install Dependencies

```bash
brew install qt@5 cmake nlohmann-json

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Models

Ensure these model files are in the project root:

**Realism models:**

- `validator_visual_1.pth` or `realism_model_1.pt`
- `validator_visual_2.pth` or `realism_model_2.pt`
- `validator_visual_3.pth` or `realism_model_3.pt`

**OCR model:**

- `ocr_validator_model.pth`
- `ocr_vectorizer.pkl`

### 3. Build Application

```bash
rm -rf build
mkdir build
cd build
cmake ..
make -j4
```

### 4. Run

```bash
./maboy.app/Contents/MacOS/maboy
```

## Usage

1. Launch `maboy.app`
2. Drag and drop document image or click "Select file"
3. Wait for validation results
4. Review detailed analysis
   - ✅ **Документ валидный** - если прошли обе проверки
   - ❌ **Документ невалидный** - если не прошла хотя бы одна проверка

### Критерии валидации

**Итоговый вердикт = (Реалистичность ✓) AND (Поля корректны ✓)**

- **Реалистичность**: Мажоритарное голосование (≥2 из 3 моделей должны сказать "реальный")
- **Поля**: OCR + MLP классификатор должен подтвердить корректность извлеченного текста

## 🔧 Разработка

### Тестирование Python валидатора

```bash
# Активировать venv

## Project Structure

```

maboy/
├── CMakeLists.txt
├── validator.py
├── requirements.txt
├── include/
│ └── DocumentValidator.h
├── src/
│ └── DocumentValidator.cpp
├── gui/
│ ├── include/
│ ├── src/
│ └── resources/
├── scripts/
│ ├── convert_models.py
│ ├── doc_orc.py
│ ├── test_model_comparison.py
│ └── original_validator.py
├── docs/
│ └── SETUP.md
├── models/
│ └── (model files: .pt, .pth, .pkl)
└── build/

````

## Python CLI

```bash
source venv/bin/activate
python3 validator.py /path/to/document.jpg --project-root .
````

JSON output:

```json
{
  "final_verdict": true,
  "realism_majority": true,
  "ocr_valid": true,
  "detailed_message": "...",
  "realism_results": [...],
  "ocr_result": {...}
}
```

## Troubleshooting

### Python not found

```bash
which python3
ln -s /opt/homebrew/bin/python3 /usr/local/bin/python3
```

### Models not loading

Check terminal logs for detailed error messages.

### EasyOCR slow startup

First run downloads models (~100MB). This happens once.

### Build error: nlohmann/json.hpp not found

```bash
brew install nlohmann-json
rm -rf build && mkdir build && cd build && cmake .. && make -j4
```

## License

MIT License

## Technologies

- Qt 5
- PyTorch
- EasyOCR
- scikit-learn
- nlohmann/json
