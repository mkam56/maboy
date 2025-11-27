#!/usr/bin/env python3
"""
Сравнение выходов .pth и .pt моделей на одном и том же изображении
"""

import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

IMG_PATH = "/Users/mehkam/Downloads/test3.jpeg"
IMG_SIZE = 224

# Трансформация
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_pth_model(path):
    """Загрузка .pth модели"""
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_pt_model(path):
    """Загрузка .pt TorchScript модели"""
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model

def preprocess_image(img_path):
    """Препроцессинг изображения"""
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    return tensor

def test_model(model, tensor, model_name):
    """Тест модели"""
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        
        print(f"\n{model_name}:")
        print(f"  Raw output: {output[0].tolist()}")
        print(f"  Probabilities: P(valid)={probs[0]:.6f}, P(fake)={probs[1]:.6f}")
        print(f"  Prediction: {'VALID' if pred == 0 else 'FAKE'}")
        
        return pred, probs

if __name__ == "__main__":
    print("="*60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ .pth vs .pt")
    print("="*60)
    print(f"Изображение: {IMG_PATH}")
    
    # Препроцессинг
    tensor = preprocess_image(IMG_PATH)
    print(f"Размер тензора: {tensor.shape}")
    
    # Тест моделей 2 и 3
    for i in [2, 3]:
        print(f"\n{'='*60}")
        print(f"MODEL {i}")
        print('='*60)
        
        pth_path = f"validator_visual_{i}.pth"
        pt_path = f"realism_model_{i}.pt"
        
        try:
            # .pth модель
            model_pth = load_pth_model(pth_path)
            pred_pth, probs_pth = test_model(model_pth, tensor, f".pth (validator_visual_{i}.pth)")
            
            # .pt модель
            model_pt = load_pt_model(pt_path)
            pred_pt, probs_pt = test_model(model_pt, tensor, f".pt (realism_model_{i}.pt)")
            
            # Сравнение
            print(f"\n{'='*60}")
            if pred_pth == pred_pt and np.allclose(probs_pth, probs_pt, atol=1e-4):
                print(f"✅ Модель {i}: ИДЕНТИЧНЫЕ РЕЗУЛЬТАТЫ")
            else:
                print(f"❌ Модель {i}: РАЗЛИЧНЫЕ РЕЗУЛЬТАТЫ!")
                print(f"   .pth prediction: {pred_pth} (P={probs_pth[pred_pth]:.4f})")
                print(f"   .pt  prediction: {pred_pt} (P={probs_pt[pred_pt]:.4f})")
                print(f"   Разница в вероятностях: {np.abs(probs_pth - probs_pt)}")
        
        except Exception as e:
            print(f"❌ Ошибка при тестировании модели {i}: {e}")
    
    print("\n" + "="*60)
