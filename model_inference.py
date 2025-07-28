import os
from numpy import ceil
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_data(self, path: str, sep: str = ';'):
    """
    Читает CSV или XLS/XLSX в зависимости от расширения файла.
    Возвращает списки текстов и меток.
    """
    
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    elif ext in {".csv"}:
        df = pd.read_csv(path, engine="python", sep=sep)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")

    if self.with_labels:
        expected = {'Текст', 'Блок'}
        if not expected.issubset(df.columns):
            raise ValueError(f"Ожидаемые колонки {expected}, но найдены {set(df.columns)}")
        texts = df['Текст'].astype(str).tolist()
        labels = df["Блок"].tolist()
        return texts, labels
    else:
        if 'Текст' not in df.columns:
            raise ValueError(f"Ожидаем колонку 'Текст', но найдена {set(df.columns)}")
        
        texts = df["Текст"].astype(str).tolist()
        return texts
    

def predict(self, texts, model, tokenizer, device, batch_size=32):
    model.eval()
    
    total_texts = len(texts)
    
    all_probs, all_preds = [], []
    max_len = 512
    with torch.no_grad():
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            probs = softmax(out.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().tolist()
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds)
            
            percent = int(i / total_texts * 100)
            self.progress.emit(percent)
            
    return all_preds, all_probs

def load_model_and_tokenizer(model_dir, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    
    id2label = {i: lab for i, lab in enumerate(class_names)}
    label2id = {lab: i for i, lab in enumerate(class_names)}
    
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    return model, tokenizer, device, id2label