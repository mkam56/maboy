#!/bin/bash

# Скрипт для первого запуска Document Validator
# Проверяет все зависимости и помогает настроить окружение

echo "=== Document Validator - Первый запуск ==="
echo ""

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Проверка 1: Python
echo -n "Проверка Python 3... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Найден Python $PYTHON_VERSION${NC}"
    PYTHON_CMD="python3"
else
    echo -e "${RED}✗ Python 3 не найден${NC}"
    echo "  Установите: brew install python3"
    exit 1
fi

# Проверка 2: Homebrew пакеты
echo -n "Проверка Qt 5... "
if brew list qt@5 &> /dev/null; then
    echo -e "${GREEN}✓ Установлена${NC}"
else
    echo -e "${YELLOW}⚠ Не установлена${NC}"
    echo "  Установите: brew install qt@5"
fi

echo -n "Проверка nlohmann-json... "
if brew list nlohmann-json &> /dev/null; then
    echo -e "${GREEN}✓ Установлена${NC}"
else
    echo -e "${YELLOW}⚠ Не установлена${NC}"
    echo "  Установите: brew install nlohmann-json"
fi

# Проверка 3: Python зависимости
echo ""
echo "Проверка Python пакетов..."

check_python_package() {
    if $PYTHON_CMD -c "import $1" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1"
        return 1
    fi
}

MISSING_PACKAGES=0
check_python_package "torch" || ((MISSING_PACKAGES++))
check_python_package "torchvision" || ((MISSING_PACKAGES++))
check_python_package "cv2" || ((MISSING_PACKAGES++))  # opencv-python
check_python_package "sklearn" || ((MISSING_PACKAGES++))  # scikit-learn
check_python_package "PIL" || ((MISSING_PACKAGES++))  # pillow
check_python_package "easyocr" || ((MISSING_PACKAGES++))

if [ $MISSING_PACKAGES -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Не хватает $MISSING_PACKAGES пакетов${NC}"
    echo "Установить все зависимости? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Установка зависимостей..."
        $PYTHON_CMD -m pip install -r requirements.txt
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Зависимости установлены${NC}"
        else
            echo -e "${RED}✗ Ошибка установки${NC}"
            exit 1
        fi
    fi
fi

# Проверка 4: Файлы моделей
echo ""
echo "Проверка моделей..."

check_model_file() {
    if [ -f "$1" ]; then
        SIZE=$(du -h "$1" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $1 ($SIZE)"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 - НЕ НАЙДЕН"
        return 1
    fi
}

MISSING_MODELS=0

# Модели реалистичности (нужен хотя бы один вариант)
echo "Модели реалистичности:"
if [ -f "realism_model_1.pt" ] || [ -f "validator_visual_1.pth" ]; then
    check_model_file "realism_model_1.pt" || check_model_file "validator_visual_1.pth"
else
    echo -e "  ${RED}✗${NC} Модель 1 не найдена (ни .pt, ни .pth)"
    ((MISSING_MODELS++))
fi

check_model_file "realism_model_2.pt" || ((MISSING_MODELS++))
check_model_file "realism_model_3.pt" || ((MISSING_MODELS++))

# OCR модель
echo "OCR модель:"
check_model_file "ocr_validator_model.pth" || ((MISSING_MODELS++))

if [ -f "tfidf_vectorizer.pkl" ]; then
    check_model_file "tfidf_vectorizer.pkl"
else
    echo -e "  ${YELLOW}⚠${NC} tfidf_vectorizer.pkl не найден (будет создан автоматически)"
fi

if [ $MISSING_MODELS -gt 0 ]; then
    echo ""
    echo -e "${RED}✗ Не хватает $MISSING_MODELS обязательных моделей${NC}"
    echo "Убедитесь что все .pt и .pth файлы находятся в корне проекта"
    exit 1
fi

# Проверка 5: validator.py
echo ""
echo -n "Проверка validator.py... "
if [ -f "validator.py" ]; then
    echo -e "${GREEN}✓ Найден${NC}"
    chmod +x validator.py
else
    echo -e "${RED}✗ Не найден${NC}"
    exit 1
fi

# Проверка 6: Сборка проекта
echo ""
echo -n "Проверка сборки... "
if [ -f "build/maboy.app/Contents/MacOS/maboy" ]; then
    echo -e "${GREEN}✓ Приложение собрано${NC}"
    NEED_BUILD=0
else
    echo -e "${YELLOW}⚠ Приложение не собрано${NC}"
    NEED_BUILD=1
fi

if [ $NEED_BUILD -eq 1 ]; then
    echo ""
    echo "Собрать приложение? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Сборка приложения..."
        rm -rf build
        mkdir build
        cd build
        cmake .. && make -j4
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Сборка успешна${NC}"
            cd ..
        else
            echo -e "${RED}✗ Ошибка сборки${NC}"
            cd ..
            exit 1
        fi
    fi
fi

# Итог
echo ""
echo "========================================="
echo -e "${GREEN}✓ Все проверки пройдены!${NC}"
echo "========================================="
echo ""
echo "Запустить приложение:"
echo "  cd build"
echo "  ./maboy.app/Contents/MacOS/maboy"
echo ""
echo "Или:"
echo "  open build/maboy.app"
echo ""
echo "Тестировать Python валидатор:"
echo "  python3 validator.py /path/to/image.jpg --project-root ."
echo ""
