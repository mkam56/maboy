#!/usr/bin/env python3
"""
Конвертация моделей из .pth (state_dict) в .pt (TorchScript)
для использования с LibTorch в C++
"""

import torch
import torch.nn as nn
from torchvision import models
import os

# Конфигурация
IMG_SIZE = 224
DEVICE = "cpu"  # Для совместимости с LibTorch используем CPU

def convert_realism_model(pth_path, pt_path, model_name):
    """Конвертация модели реалистичности"""
    print(f"\n{'='*50}")
    print(f"Конвертация: {model_name}")
    print(f"Из: {pth_path}")
    print(f"В:  {pt_path}")
    print(f"{'='*50}")
    
    # Создаем архитектуру модели (ConvNeXt Tiny)
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    
    # Загружаем веса
    print("Загрузка весов...")
    state_dict = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Создаем пример входных данных
    # Формат: [batch_size, channels, height, width]
    example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    
    # Конвертация в TorchScript через tracing
    print("Конвертация в TorchScript...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    # Сохранение
    traced_model.save(pt_path)
    print(f"✅ Модель сохранена: {pt_path}")
    
    # Проверка
    print("\nПроверка загрузки...")
    loaded_model = torch.jit.load(pt_path)
    loaded_model.eval()
    
    with torch.no_grad():
        output = loaded_model(example_input)
        probs = torch.softmax(output, dim=1)
        print(f"Выходной размер: {output.shape}")
        print(f"Вероятности: {probs[0].tolist()}")
    
    print("✅ Проверка пройдена!")
    return True


def convert_ocr_model(pth_path, pt_path, model_name):
    """Конвертация OCR модели"""
    print(f"\n{'='*50}")
    print(f"Конвертация: {model_name}")
    print(f"Из: {pth_path}")
    print(f"В:  {pt_path}")
    print(f"{'='*50}")
    
    # OCR модель может использовать больший размер
    OCR_SIZE = 448
    
    # Создаем архитектуру модели
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    
    # Загружаем веса
    print("Загрузка весов...")
    state_dict = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Пример входных данных для OCR (больший размер)
    example_input = torch.randn(1, 3, OCR_SIZE, OCR_SIZE)
    
    # Конвертация в TorchScript
    print("Конвертация в TorchScript...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    # Сохранение
    traced_model.save(pt_path)
    print(f"✅ Модель сохранена: {pt_path}")
    
    # Проверка
    print("\nПроверка загрузки...")
    loaded_model = torch.jit.load(pt_path)
    loaded_model.eval()
    
    with torch.no_grad():
        output = loaded_model(example_input)
        probs = torch.softmax(output, dim=1)
        print(f"Выходной размер: {output.shape}")
        print(f"Вероятности: {probs[0].tolist()}")
    
    print("✅ Проверка пройдена!")
    return True


def main():
    # Пути к файлам
    base_dir = "/Users/mehkam/CLionProjects/maboy"
    
    models_to_convert = [
        {
            "input": os.path.join(base_dir, "validator_visual_1.pth"),
            "output": os.path.join(base_dir, "realism_model_1.pt"),
            "name": "Realism Model 1",
            "type": "realism"
        },
        {
            "input": os.path.join(base_dir, "validator_visual_2.pth"),
            "output": os.path.join(base_dir, "realism_model_2.pt"),
            "name": "Realism Model 2",
            "type": "realism"
        },
        {
            "input": os.path.join(base_dir, "validator_visual_3.pth"),
            "output": os.path.join(base_dir, "realism_model_3.pt"),
            "name": "Realism Model 3",
            "type": "realism"
        },
        {
            "input": os.path.join(base_dir, "ocr_validator_model.pth"),
            "output": os.path.join(base_dir, "ocr_model.pt"),
            "name": "OCR Model",
            "type": "ocr"
        }
    ]
    
    print("\n🚀 Начало конвертации моделей...")
    print(f"Всего моделей: {len(models_to_convert)}\n")
    
    success_count = 0
    failed_models = []
    
    for model_info in models_to_convert:
        try:
            # Проверка существования входного файла
            if not os.path.exists(model_info["input"]):
                print(f"\n❌ Файл не найден: {model_info['input']}")
                failed_models.append(model_info["name"])
                continue
            
            # Конвертация в зависимости от типа
            if model_info["type"] == "realism":
                success = convert_realism_model(
                    model_info["input"],
                    model_info["output"],
                    model_info["name"]
                )
            else:  # ocr
                success = convert_ocr_model(
                    model_info["input"],
                    model_info["output"],
                    model_info["name"]
                )
            
            if success:
                success_count += 1
                
        except Exception as e:
            print(f"\n❌ Ошибка при конвертации {model_info['name']}: {e}")
            failed_models.append(model_info["name"])
    
    # Итоги
    print("\n" + "="*50)
    print("ИТОГИ КОНВЕРТАЦИИ")
    print("="*50)
    print(f"✅ Успешно: {success_count}/{len(models_to_convert)}")
    
    if failed_models:
        print(f"❌ Ошибки: {len(failed_models)}")
        for model_name in failed_models:
            print(f"  - {model_name}")
    
    if success_count == len(models_to_convert):
        print("\n🎉 ВСЕ МОДЕЛИ УСПЕШНО КОНВЕРТИРОВАНЫ!")
        print("\n📝 Конвертированные файлы:")
        for model_info in models_to_convert:
            if os.path.exists(model_info["output"]):
                size_mb = os.path.getsize(model_info["output"]) / (1024 * 1024)
                print(f"  ✓ {os.path.basename(model_info['output'])} ({size_mb:.1f} MB)")
        
        print("\n🚀 Теперь можно запустить приложение:")
        print("   cd /Users/mehkam/CLionProjects/maboy/build")
        print("   ./maboy.app/Contents/MacOS/maboy")
    else:
        print("\n⚠️  Не все модели были конвертированы")
    
    print()


if __name__ == "__main__":
    main()
