# ocr_pipeline.py
import os
import json
import re
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import easyocr

# ========== Конфигурация ==========
DATA_DIR = "./datasets/MIDV2020"
OUT_DIR = "./ocr_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Путь к EAST модели (можно поменять, если у вас другой файл)
# Обычно имя: frozen_east_text_detection.pb
EAST_PATH = "frozen_east_text_detection.pb"  # если нет — поставьте None, код будет пытаться без EAST
MIN_CONFIDENCE = 0.5
EAST_WIDTH = 320   # размеры для EAST; кратны 32
EAST_HEIGHT = 320

# EasyOCR reader (поддержка языков: 'ru','en' — можно добавить)
READER = easyocr.Reader(['ru', 'en'], gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)


# ========== Утилиты ==========
def resize_keep_aspect(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image, 1.0, 1.0
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized, float(dim[0]) / w, float(dim[1]) / h  # scale_x, scale_y


# EAST text detector
def east_detect(image, net, input_size=(EAST_WIDTH, EAST_HEIGHT), min_confidence=MIN_CONFIDENCE):
    # returns list of boxes in original image coordinates: [(startX, startY, endX, endY, score), ...]
    orig_h, orig_w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, input_size, (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    # two outputs: geometry and scores
    (scores, geometry) = net.forward(net.getUnconnectedOutLayersNames())
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue
            # geometry
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))
    # apply nms
    boxes = []
    if len(rects) > 0:
        boxes_np = np.array(rects)
        picks = cv2.dnn.NMSBoxes(rects, confidences, min_confidence, 0.4)
        if isinstance(picks, (list, tuple, np.ndarray)):
            pick_iter = picks
        else:
            pick_iter = picks.flatten()
        for i in pick_iter:
            (sX, sY, eX, eY) = rects[int(i)]
            boxes.append((max(0, sX), max(0, sY), min(orig_w, eX), min(orig_h, eY), confidences[int(i)]))
    return boxes


# Fallback: детект текста по контурным методам (если нет EAST)
def simple_text_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # adaptive thresh + dilate to merge letters
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    dil = cv2.dilate(th, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = image.shape[:2]
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww < 30 or hh < 10:
            continue
        boxes.append((x, y, x + ww, y + hh, 1.0))
    # sort top-to-bottom
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


# ========== Парсинг полей (базовые эвристики) ==========
# Здесь можно расширять правила под конкретную страну/шаблон документа.
RE_DATE = re.compile(r'\b(19|20|18|21|22|23)\d{2}\b')  # 4 цифры года (пример)
RE_FULLDATE = re.compile(r'\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b')
RE_NUMBER = re.compile(r'\b[0-9A-Z\-]{4,}\b')  # простой шаблон номера
RE_NAME = re.compile(r'^[A-ZА-ЯЁ][\w\-\s]{1,60}$', re.I)

def parse_fields(text):
    """
    Примитивная функция, которая пытается извлечь даты, возможные имена и номера из текста.
    Возвращает dict с ключами: names, dates, numbers, raw_text
    """
    res = {'names': [], 'dates': [], 'numbers': [], 'raw_text': text}
    text = text.replace('\n', ' ')
    # Найти форматы дат d/m/yyyy или просто yyyy
    for m in RE_FULLDATE.finditer(text):
        res['dates'].append(m.group(1))
    for m in RE_DATE.finditer(text):
        y = m.group(0)
        if y not in res['dates']:
            res['dates'].append(y)
    # Номера (серии, паспорта)
    for m in RE_NUMBER.finditer(text):
        token = m.group(0)
        # отбросим «слова» которые скорее всего текст
        if len(token) >= 4 and sum(c.isdigit() for c in token) >= 2:
            res['numbers'].append(token)
    # Имена — простая эвристика: слова с буквами и заглавной первой
    tokens = re.split(r'[\s,;:/]+', text)
    for t in tokens:
        t = t.strip()
        if len(t) >= 2 and t.isalpha() and RE_NAME.match(t):
            # отбросим общие слова (можно расширить стоп-листом)
            if len(t) > 2:
                res['names'].append(t)
    # уберём дубликаты
    for k in ['names', 'dates', 'numbers']:
        res[k] = list(dict.fromkeys(res[k]))
    return res


# ========== Основная логика обработки одного изображения ==========
def process_image(path, east_net=None):
    img = cv2.imread(path)
    if img is None:
        return None
    orig = img.copy()
    h, w = img.shape[:2]

    # Детект области текста
    boxes = []
    if east_net is not None:
        try:
            # востановим в нужный размер для fast processing (EAST требует кратности 32)
            # но тут мы детектим в оригинале — ok
            boxes = east_detect(img, east_net, input_size=(EAST_WIDTH, EAST_HEIGHT), min_confidence=MIN_CONFIDENCE)
        except Exception as e:
            print("EAST error:", e)
            boxes = simple_text_regions(img)
    else:
        boxes = simple_text_regions(img)

    # Если ничего не найдено, попробуем whole-image OCR
    results = []
    if len(boxes) == 0:
        ocr_res = READER.readtext(img, detail=0)
        txt = " ".join(ocr_res)
        parsed = parse_fields(txt)
        return {'image': os.path.basename(path), 'text': txt, 'fields': parsed, 'boxes': []}

    # Иначе для каждого бокса делаем OCR
    aggregated_text = []
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))  # top-left order
    for (sX, sY, eX, eY, score) in boxes_sorted:
        # pad
        pad = 3
        sXp = max(0, sX - pad)
        sYp = max(0, sY - pad)
        eXp = min(w, eX + pad)
        eYp = min(h, eY + pad)
        crop = orig[sYp:eYp, sXp:eXp]
        try:
            # EasyOCR возвращает список строк или (bbox, text, prob) в зависимости от флагов
            ocr_out = READER.readtext(crop, detail=0)
            text_block = " ".join(ocr_out).strip()
            if text_block:
                aggregated_text.append({'box': (sXp, sYp, eXp, eYp), 'text': text_block, 'score': score})
        except Exception as e:
            # fallback: pytesseract (если установлен)
            try:
                import pytesseract
                text_block = pytesseract.image_to_string(crop, lang='rus+eng')
                if text_block.strip():
                    aggregated_text.append({'box': (sXp, sYp, eXp, eYp), 'text': text_block.strip(), 'score': score})
            except Exception:
                continue

    full_text = " ".join([b['text'] for b in aggregated_text])
    parsed = parse_fields(full_text)
    return {'image': os.path.basename(path), 'text': full_text, 'fields': parsed, 'boxes': aggregated_text}


# ========== Проход по директории ==========
def run_on_dataset(data_dir=DATA_DIR, out_dir=OUT_DIR, east_path=EAST_PATH):
    # Попробуем загрузить EAST если есть
    east_net = None
    if east_path and os.path.exists(east_path):
        try:
            east_net = cv2.dnn.readNet(east_path)
            print("EAST loaded.")
        except Exception as e:
            print("Не удалось загрузить EAST:", e)
            east_net = None
    else:
        print("EAST not provided or not found — using simple contour detector.")

    # собираем все изображения
    img_paths = []
    for mode in ["photo", "scan_upright"]:
        base_dir = os.path.join(data_dir, mode, "images")
        if not os.path.exists(base_dir):
            continue
        for subdir, _, files in os.walk(base_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_paths.append(os.path.join(subdir, fname))

    if len(img_paths) == 0:
        print("No images found. Проверь DATA_DIR.")
        return

    for p in tqdm(img_paths):
        out = process_image(p, east_net=east_net)
        if out is None:
            continue
        # сохранить json
        base = Path(p).stem
        with open(os.path.join(out_dir, base + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_on_dataset()

