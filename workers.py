import logging
from pathlib import Path
from PySide6.QtCore import QObject, Signal
from model_inference import load_data, load_model_and_tokenizer, predict
from sklearn.metrics import accuracy_score 
from config import CLASS_NAMES

class InferenceWorker(QObject):
    
    # Сигналы, которые будут эмитироваться в GUI‑поток
    finished = Signal()                     # когда всё завершилось
    error    = Signal(str)                  # при исключении
    progress = Signal(int)                  # для отчёта о прогрессе (0–100)
    result   = Signal(list, list, list, list, object)
    
    def __init__(self, model_path, data_path, batch_size, with_labels=False):
        super().__init__()
        self.model_path = model_path
        self.data_path  = data_path
        self.batch_size = batch_size
        self.with_labels  = with_labels

    @staticmethod
    def has_pth_files(folder_path: str) -> bool:
        p = Path(folder_path)
        return any(child.is_file() and child.suffix.lower() == ".pth" for child in p.iterdir())

    def run(self):
        """Метод, куда вынесена вся тяжёлая работа."""
        try:            
            if self.has_pth_files(self.model_path):
                # try:
                #    model, tokenizer, device, id2label = load_model_and_tokenizer_pth(
                #         self.model_path, class_names
                #     )
                #    logging.info("Модель и токенизатор загружены.") 
                # except Exception as e:
                #     logging.error(f"Ошибка загрузки модели .pth {e}")
                #     return
                logging.info("Найдены .pth файлы, но функционала пока что нет")
                return
            else:
                try:
                    # Загрузка модели и токенизатора + device
                    model, tokenizer, device, id2label = load_model_and_tokenizer(
                        self.model_path, CLASS_NAMES
                    )
                    logging.info("Модель и токенизатор загружены.")
                except Exception as e:
                    logging.error("Ошибка загрузки модели" + str(e))
                    return
                
            try:
                if self.with_labels:
                    texts, raw_labels = load_data(self, self.data_path, sep=';')
                    filtered = [(t, l) for t, l in zip(texts, raw_labels) 
                                if isinstance(l, str) and l in model.config.label2id]
                    if not filtered:
                        raise ValueError("Нет валидных строк с метками из label2id")
                    logging.info(f"Загружено {len(texts)} примеров, после фильтрации осталось {len(texts)}.")
                else:
                    texts = load_data(self, self.data_path, sep=';')    
                # Предсказание
                logging.info("Запуск предсказания...")
                preds, probs = predict(self, texts, model, tokenizer, device,
                                    batch_size=self.batch_size)
                logging.info("Предсказание завершено.")
                
                self.progress.emit(100)
            except Exception as e:
                logging.error("Ошибка предсказания" + str(e))
                return
            
            # Отображаем метрики в консоли Qt
            if self.with_labels:
                texts, true_labels = zip(*filtered)
                true_labels = [model.config.label2id[l] for l in true_labels]
                acc = accuracy_score(true_labels, preds)
                logging.info(f"Accuracy: {acc:.4f}")
                self.result.emit(texts, preds, probs, true_labels, id2label)
            else:
                self.result.emit(texts, preds, probs, [], id2label)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()