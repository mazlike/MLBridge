import os
import re
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
    
# --- Прекомпилированные паттерны (делаем один раз при импортe модуля) ---
HTML_TAG_RE = re.compile(r'<[^>]+>')
URL_RE = re.compile(r'http[s]?://\S+|www\.\S+')
SQUARE_BRACKETS_RE = re.compile(r'\[.*?\]')
EMOJI_RE = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)
PUNCT_RE = re.compile(r'[^\w\s]')
MULTI_WS_RE = re.compile(r'\s+')

# --- Функция очистки (использует прекомпилированные паттерны) ---
def remove_html_tags(text: str) -> str:
    """Чистит один текст: убирает html, ссылки, скобочные вставки, эмодзи, спец.символы, лишние пробелы."""
    if text is None:
        return ""
    text = str(text)

    # удалить HTML-теги (включая <br/>, <br>, и т.п.)
    text = HTML_TAG_RE.sub(' ', text)

    # удалить URL
    text = URL_RE.sub('', text)

    # удалить содержимое в квадратных скобках [..]
    text = SQUARE_BRACKETS_RE.sub('', text)

    # убрать эмодзи (компактный паттерн)
    text = EMOJI_RE.sub('', text)

    # удалить пунктуацию, оставить буквы/цифры/пробелы/подчёрки
    text = PUNCT_RE.sub(' ', text)

    # заменить переносы строк на пробелы, сжать многократные пробелы
    text = text.replace('\n', ' ')
    text = MULTI_WS_RE.sub(' ', text).strip()

    return text.lower()


def preprocess_texts(texts):
    """
    Приводит texts (Series/list/np.array/str) к списку очищенных строк.
    Использует vectorized apply для pandas.Series.
    """
    if isinstance(texts, pd.Series):
        return texts.astype(str).apply(remove_html_tags).tolist()
    elif isinstance(texts, list) or (hasattr(texts, '__iter__') and not isinstance(texts, (str, bytes))):
        return [remove_html_tags(t) for t in texts]
    else:
        # единичный объект
        return [remove_html_tags(texts)]

    
def predict(self, texts, model, tokenizer, device, batch_size=16, max_len=512):
    """
    Предсказывает классы и вероятности для списка текстов.
    texts может быть pd.Series, list или одиночной строкой.
    Возвращает (all_preds, all_probs).
    """
    model.to(device)
    model.eval()

    texts_clean = preprocess_texts(texts)
    dataset = TextDataset(texts_clean)   # TextDataset должен возвращать строку на getitem
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_batches = len(loader) if len(loader) > 0 else 1

    all_probs, all_preds = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # batch ожидается как list[str] — токенизируем список
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            # учесть разные типы возвращаемого объекта
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().tolist()

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds)

            # чтобы в конце достичь 100%
            percent = int((batch_idx + 1) / total_batches * 100)
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
        local_files_only=True
        )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True
        )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True
    ).to(device)
    id2label = {i: lab for i, lab in enumerate(class_names)}
    label2id = {lab: i for i, lab in enumerate(class_names)}
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    return model, tokenizer, device, id2label