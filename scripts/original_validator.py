#!/usr/bin/env python3
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
import joblib
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import easyocr
import warnings

warnings.filterwarnings("ignore")

# ============================================
#                CONFIG
# ============================================
IMG_SIZE = 224
VISUAL_MODELS = [
    "validator_visual_1.pth",
    "validator_visual_2.pth",
    "validator_visual_3.pth"
]

OCR_MODEL_PATH = "ocr_validator_model.pth"
VECTORIZER_PATH = "ocr_vectorizer.pkl"

DEVICE = "cpu"  # Принудительно CPU для теста
# DEVICE = (
#     "mps" if torch.backends.mps.is_available()
#     else "cuda" if torch.cuda.is_available()
#     else "cpu"
# )

# ============================================
#              TRANSFORM
# ============================================
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# ============================================
#           VISUAL MODEL LOADER
# ============================================
def load_visual_model(path):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)

    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(image=img)["image"].unsqueeze(0)
    return tensor.to(DEVICE)


def predict_visual(model, tensor):
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        return pred, probs


# ============================================
#              OCR MODEL
# ============================================
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


def load_ocr_model():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = OCRValidator(len(vectorizer.get_feature_names_out()))

    state = torch.load(OCR_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)

    model = model.to(DEVICE)
    model.eval()
    return model, vectorizer


# ============================================
#               OCR READING
# ============================================
reader = easyocr.Reader(["en", "ru"], gpu=False)

def extract_text(path):
    try:
        result = reader.readtext(path, detail=0)
        return " ".join(result)
    except Exception:
        return ""


def predict_ocr(model, vectorizer, text):
    X = vectorizer.transform([text]).toarray().astype(np.float32)
    X = torch.tensor(X).to(DEVICE)

    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
        return pred, probs


# ============================================
#              MAIN VALIDATOR
# ============================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_document.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]

    print("\n===============================")
    print("   DOCUMENT VALIDATION START")
    print("===============================\n")
    print(f"📄 File: {img_path}")
    print(f"⚙️  Device: {DEVICE}\n")

    # ------------------------------
    # 1) VISUAL VALIDATION
    # ------------------------------
    try:
        img_tensor = preprocess_image(img_path)
    except Exception as e:
        print(f"❌ Image error: {e}")
        sys.exit(1)

    print("🔍 Running visual validators:")
    visual_votes = []

    for idx, path in enumerate(VISUAL_MODELS, 1):
        try:
            model = load_visual_model(path)
        except Exception as e:
            print(f"❌ Failed to load model {path}: {e}")
            visual_votes.append(1)
            continue

        pred, probs = predict_visual(model, img_tensor)
        visual_votes.append(pred)

        status = "VALID ✅" if pred == 0 else "FAKE ❌"
        print(f"  • Model {idx}: {path}")
        print(f"      P(valid)={probs[0]:.4f}  P(fake)={probs[1]:.4f} → {status}")

    # visual decision
    valid_votes = visual_votes.count(0)
    visual_final = 0 if valid_votes >= 2 else 1

    print(f"\n🎯 Visual decision: {valid_votes}/3 → {'VALID' if visual_final==0 else 'FAKE'}\n")

    # ------------------------------
    # 2) OCR VALIDATION
    # ------------------------------
    print("🔍 Running OCR validator:")

    text = extract_text(img_path)
    model_ocr, vectorizer = load_ocr_model()
    pred_ocr, probs_ocr = predict_ocr(model_ocr, vectorizer, text)

    ocr_status = "VALID ✅" if pred_ocr == 0 else "FAKE ❌"
    print(f"  • OCR model: {OCR_MODEL_PATH}")
    print(f"      P(valid)={probs_ocr[0]:.4f}  P(fake)={probs_ocr[1]:.4f} → {ocr_status}\n")

    # ------------------------------
    # FINAL DECISION
    # ------------------------------
    final = "VALID ✅" if (visual_final == 0 and pred_ocr == 0) else "FAKE ❌"

    print("===============================")
    print(f" FINAL RESULT: {final}")
    print("===============================\n")
