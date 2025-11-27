#!/usr/bin/env python3

import sys
import json
import argparse
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import joblib
import easyocr
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")


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


class DocumentValidator:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() 
            else 'cuda' if torch.cuda.is_available() 
            else 'cpu'
        )
        
        self.realism_models = []
        self.realism_names = ["Realism Model 1", "Realism Model 2", "Realism Model 3"]
        
        self.ocr_model = None
        self.tfidf_vectorizer = None
        self.ocr_reader = None
        
        self.realism_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
    
    def load_models(self):
        models_dir = self.project_root / "models"
        
        realism_paths = [
            models_dir / "validator_visual_1.pth",
            models_dir / "validator_visual_2.pth",
            models_dir / "validator_visual_3.pth"
        ]
        
        loaded_count = 0
        for i, (path, name) in enumerate(zip(realism_paths, self.realism_names)):
            try:
                if not path.exists():
                    pt_path = models_dir / f"realism_model_{i+1}.pt"
                    if not pt_path.exists():
                        pt_path = self.project_root / f"realism_model_{i+1}.pt"
                    if pt_path.exists():
                        model = torch.jit.load(str(pt_path), map_location=self.device)
                        model.eval()
                        self.realism_models.append(model)
                        loaded_count += 1
                    continue
                
                model = models.convnext_tiny(weights=None)
                in_features = model.classifier[2].in_features
                model.classifier[2] = nn.Linear(in_features, 2)
                
                state_dict = torch.load(str(path), map_location=self.device)
                model.load_state_dict(state_dict)
                model = model.to(self.device)
                model.eval()
                
                self.realism_models.append(model)
                loaded_count += 1
            except Exception:
                pass
        
        if loaded_count < 2:
            return False
        
        ocr_path = models_dir / "ocr_validator_model.pth"
        if not ocr_path.exists():
            ocr_path = self.project_root / "ocr_validator_model.pth"
            
        vectorizer_path = models_dir / "ocr_vectorizer.pkl"
        if not vectorizer_path.exists():
            vectorizer_path = self.project_root / "ocr_vectorizer.pkl"
        
        try:
            if vectorizer_path.exists():
                self.tfidf_vectorizer = joblib.load(str(vectorizer_path))
            else:
                return False
            
            if ocr_path.exists():
                input_dim = len(self.tfidf_vectorizer.get_feature_names_out())
                self.ocr_model = OCRValidator(input_dim=input_dim)
                
                state_dict = torch.load(ocr_path, map_location=self.device)
                self.ocr_model.load_state_dict(state_dict)
                
                self.ocr_model = self.ocr_model.to(self.device)
                self.ocr_model.eval()
            else:
                return False
            
        except Exception:
            return False
        
        try:
            self.ocr_reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())
        except Exception:
            return False
        
        return True
    
    def run_realism_model(self, image_path, model, model_name):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = self.realism_transform(image=img)["image"].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                is_valid = (pred == 0)
                confidence = float(probs[pred])
            
            return {
                'is_valid': is_valid,
                'confidence': confidence,
                'model_name': model_name
            }
        except Exception:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'model_name': model_name
            }
    
    def extract_text_with_ocr(self, image_path):
        try:
            result = self.ocr_reader.readtext(str(image_path), detail=0)
            text = " ".join(result)
            return text
        except Exception:
            return ""
    
    def validate_ocr_fields(self, text):
        try:
            if not text or not text.strip():
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'model_name': 'OCR Model'
                }
            
            tfidf_vector = self.tfidf_vectorizer.transform([text]).toarray().astype(np.float32)
            input_tensor = torch.tensor(tfidf_vector).to(self.device)
            
            with torch.no_grad():
                output = self.ocr_model(input_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                pred = int(np.argmax(probs))
                valid_prob = float(probs[0])
                invalid_prob = float(probs[1])
            
            is_valid = (pred == 0)
            confidence = max(valid_prob, invalid_prob)
            
            return {
                'is_valid': is_valid,
                'confidence': confidence,
                'model_name': 'OCR Model'
            }
        except Exception:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'model_name': 'OCR Model'
            }
    
    def validate(self, document_path):
        result = {
            'final_verdict': False,
            'realism_results': [],
            'realism_majority': False,
            'ocr_result': {},
            'ocr_valid': False,
            'extracted_text': '',
            'detailed_message': ''
        }
        
        try:
            realism_results = []
            for model, name in zip(self.realism_models, self.realism_names):
                res = self.run_realism_model(document_path, model, name)
                realism_results.append(res)
            
            result['realism_results'] = realism_results
            
            valid_count = sum(1 for r in realism_results if r['is_valid'])
            result['realism_majority'] = valid_count >= 2
            
            extracted_text = self.extract_text_with_ocr(document_path)
            result['extracted_text'] = extracted_text
            
            ocr_result = self.validate_ocr_fields(extracted_text)
            result['ocr_result'] = ocr_result
            result['ocr_valid'] = ocr_result['is_valid']
            
            result['final_verdict'] = result['realism_majority'] and result['ocr_valid']
            result['detailed_message'] = self.create_detailed_message(result)
            
            return result
            
        except Exception as e:
            result['detailed_message'] = f"Validation error: {str(e)}"
            return result
    
    def create_detailed_message(self, result):
        msg = "=========================================\n"
        msg += "   VALIDATION RESULT                   \n"
        msg += "=========================================\n\n"
        
        msg += "REALISM CHECK (3 models):\n"
        for i, r in enumerate(result['realism_results'], 1):
            msg += f"  {i}.  {r['model_name']}: "
            msg += f"{'VALID' if r['is_valid'] else 'INVALID'} "
            msg += f"({r['confidence']*100:.1f}%)\n"
        
        msg += f"\n  Majority decision: "
        msg += f"{'REAL' if result['realism_majority'] else 'FAKE'}\n"
        
        msg += "\nFIELD CHECK (OCR + neural network):\n"
        ocr = result['ocr_result']
        msg += f"  {ocr['model_name']}: "
        msg += f"{'CORRECT' if result['ocr_valid'] else 'INCORRECT'} "
        msg += f"({ocr['confidence']*100:.1f}%)\n"
        
        msg += "\n-----------------------------------------\n"
        msg += "FINAL VERDICT:\n"
        msg += f"   Realism: {'OK' if result['realism_majority'] else 'FAIL'}\n"
        msg += f"   Fields: {'OK' if result['ocr_valid'] else 'FAIL'}\n"
        msg += "-----------------------------------------\n"
        
        if result['final_verdict']:
            msg += "   DOCUMENT VALID\n"
        else:
            msg += "   DOCUMENT INVALID\n"
            msg += "\n   Reason: "
            if not result['realism_majority'] and not result['ocr_valid']:
                msg += "Fake + Invalid fields\n"
            elif not result['realism_majority']:
                msg += "Document is fake\n"
            else:
                msg += "Fields are invalid\n"
        
        all_confidences = [r['confidence'] for r in result['realism_results']]
        all_confidences.append(result['ocr_result']['confidence'])
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        msg += f"\nAverage confidence: {avg_confidence*100:.1f}%\n"
        
        return msg


def main():
    parser = argparse.ArgumentParser(description='Document Validator')
    parser.add_argument('image_path', type=str)
    parser.add_argument('--project-root', type=str, default='.')
    args = parser.parse_args()
    
    validator = DocumentValidator(args.project_root)
    
    if not validator.load_models():
        result = {
            'final_verdict': False,
            'error': 'Failed to load models'
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    result = validator.validate(args.image_path)
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result['final_verdict'] else 1)


if __name__ == '__main__':
    main()
