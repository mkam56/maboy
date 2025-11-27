import os
import re
import time
import joblib
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import easyocr

# ========== 0. Настройки ==========
DATA_DIR = "./datasets/MIDV2020"
MODEL_PATH = "ocr_validator_model.pth"
VECT_PATH = "ocr_vectorizer.pkl"
OCR_CACHE = "ocr_cache.pkl"
DEVICE = "cpu"  # "cuda" если есть GPU и torch.cuda.is_available()
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32

# ========== 1. Инициализация EasyOCR ==========
print("Инициализация EasyOCR... (это может занять время при первом запуске)")
t0 = time.time()
# явно укажем gpu=False, если у тебя нет CUDA, чтобы избежать лишних предупреждений и попыток
reader = easyocr.Reader(['en', 'ru'], gpu=False)
print(f"EasyOCR инициализирован за {time.time()-t0:.1f}s")

# ========== 2. Сбор путей к изображениям ==========
image_paths = []
for mode in ["photo", "scan_upright"]:
    base_dir = os.path.join(DATA_DIR, mode, "images")
    if not os.path.exists(base_dir):
        continue
    for subdir, _, files in os.walk(base_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(subdir, fname))

print(f"Найдено изображений: {len(image_paths)}")
if len(image_paths) == 0:
    print("Внимание: не найдено изображений. Проверь DATA_DIR.")
    # можно exit или продолжать с пустым набором
    # exit(1)

# ========== 3. OCR + кэширование результатов ==========
if os.path.exists(OCR_CACHE):
    print(f"Загрузка OCR кэша из {OCR_CACHE}")
    texts = joblib.load(OCR_CACHE)
else:
    texts = []
    # Проход с прогрессбаром. detail=0 — только текст.
    print("Запуск OCR по изображениям (показываю прогресс)...")
    for path in tqdm(image_paths, desc="OCR images"):
        try:
            result = reader.readtext(path, detail=0)
            text = " ".join(result)
        except Exception as e:
            # лог ошибки, но продолжаем
            text = ""
        texts.append(text)
    # Кэшируем, чтобы не выполнять OCR при каждом запуске
    joblib.dump(texts, OCR_CACHE)
    print(f"OCR результаты сохранены в {OCR_CACHE}")

# Метки: сначала все валидные
labels = [0] * len(texts)

# ========== 4. Синтетические "фейки" ==========
def generate_fake_text(text):
    # чуть более консервативно, чтобы не портить слишком сильно
    text = re.sub(r"\b(19|20)\d{2}\b", "3000", text)  # неверные годы
    text = re.sub(r"\b[A-ZА-Я]{2,}\b", "XXX", text)  # грубая замена ЗАГЛАВНЫХ слов
    # допустим, еще вставим цифру в слово фамилии как пример
    text = re.sub(r"\b([A-Za-zА-Яа-я]{3,})\b", lambda m: (m.group(1) + ("1" if np.random.rand() < 0.02 else "")), text)
    return text

print("Генерация синтетических фейков...")
fake_texts = [generate_fake_text(t) for t in texts]
texts_all = texts + fake_texts
labels_all = labels + [1] * len(fake_texts)
print(f"Всего образцов: {len(texts_all)} (валид: {len(labels)}, фейк: {len(fake_texts)})")

# ========== 5. Векторизация ==========
print("Векторизация текстов (CountVectorizer). Это может занять время...")
t0 = time.time()
vectorizer = CountVectorizer(max_features=3000, ngram_range=(1, 2))
# Если данных очень много — можно сначала sample или использовать hashing trick
X = vectorizer.fit_transform(texts_all)
joblib.dump(vectorizer, VECT_PATH)
print(f"Vectorizer сохранён в {VECT_PATH}. Время: {time.time()-t0:.1f}s")
y = np.array(labels_all)

# ========== 6. Трен/валид сплит ==========
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

# Предупреждение про toarray:
print("Преобразование sparse->dense для передачи в PyTorch. Если данных много, это займёт память.")
t0 = time.time()
X_train_arr = X_train.toarray().astype(np.float32)
X_val_arr = X_val.toarray().astype(np.float32)
print(f"toarray выполнен за {time.time()-t0:.1f}s; размер X_train: {X_train_arr.nbytes/1e6:.1f} MB")

# ========== 7. Модель ==========
class OCRValidator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.fc(x)

model = OCRValidator(X_train_arr.shape[1]).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Подготовка dataloader
train_dataset = TensorDataset(torch.from_numpy(X_train_arr), torch.from_numpy(y_train.astype(np.int64)))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== 8. Обучение с прогрессбаром ==========
print("Начинаем обучение:")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    # tqdm для батчей, покажет прогресс эпохи
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    for xb, yb in pbar:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # обновляем описание прогрессбара текущим loss
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} finished. Avg Loss: {avg_loss:.4f}")

# ========== 9. Валидация ==========
print("Валидация...")
model.eval()
with torch.no_grad():
    X_val_t = torch.from_numpy(X_val_arr).to(DEVICE)
    logits = model(X_val_t)
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = torch.argmax(logits, dim=1).cpu().numpy()

print(classification_report(y_val, preds, target_names=["valid", "fake"]))
try:
    auc = roc_auc_score(y_val, probs)
    print("AUC:", auc)
except Exception as e:
    print("Не удалось вычислить AUC:", e)

# ========== 10. Сохранение модели ==========
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ OCR-модель сохранена в {MODEL_PATH}")
