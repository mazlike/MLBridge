import os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]
    
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
    

def predict(self, texts, model, tokenizer, device, batch_size=16, max_len=512):
    """
    Предсказывает классы и вероятности для списка текстов.
    Отдаёт список меток и список вероятностей для каждого текста.
    Использует DataLoader для батчей и ваш прогресс-бар.
    """
    model.to(device).eval()
    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_batches = len(loader)

    all_probs, all_preds = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # токенизация и перенос на устройство
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            # прямой проход
            outputs = model(**enc)

            # вероятности и предсказания
            probs = F.softmax(outputs.logits, dim=1)
            preds = probs.argmax(dim=1).cpu().tolist()
            
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds)
            
            percent = int(batch_idx / total_batches * 100)
            self.progress.emit(percent)
            
    return all_preds, all_probs

def load_model_and_tokenizer(model_dir, class_names):
    if model_dir is None:
        raise ValueError("model_dir не должен быть None. Укажите путь к вашей папке с чекпоинтами.")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Не могу найти папку по пути: {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True      # <-- не будет пытаться скачать из интернета
        )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True      # <-- не будет пытаться скачать из интернета
        ).to(device)
    
    id2label = {i: lab for i, lab in enumerate(class_names)}
    label2id = {lab: i for i, lab in enumerate(class_names)}
    
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    return model, tokenizer, device, id2label

def load_model_and_tokenizer_pth(model_dir, class_names):
    if model_dir is None:
        raise ValueError("model_dir не должен быть None. Укажите путь к вашей папке с чекпоинтами.")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Не могу найти папку по пути: {model_dir}")
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.BertTokenizer.from_pretrained(
        model_dir,
        
        )
    model = transformers.BertForSequenceClassification.from_pretrained(
        model_dir,

        ).to(device)
    
    id2label = {i: lab for i, lab in enumerate(class_names)}
    label2id = {lab: i for i, lab in enumerate(class_names)}
    
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    return model, tokenizer, device, id2label