import logging
from pathlib import Path
from typing import List, Tuple
from PySide6.QtCore import QObject, Signal

import torch

from model_inference import load_data, load_model_and_tokenizer, predict  # предполагается существование
from sklearn.metrics import accuracy_score
from config import CLASS_NAMES, BATCH_SIZE


class InferenceWorker(QObject):
    finished = Signal()
    error = Signal(str)
    progress = Signal(int)                 # 0..100
    result = Signal(list, list, list, list, list, object)
    eta = Signal(str)
    
    def __init__(self, model_path: str, data_path: str, batch_size: int, with_labels: bool = False):
        super().__init__()
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.batch_size = int(batch_size)
        self.with_labels = with_labels

        
        # <-- флаг отмены
        self._stop_requested = False
    def request_stop(self):
        """Вызвать извне (MainWindow) для безопасной остановки."""
        logging.info("Requesting stop for InferenceWorker")
        self._stop_requested = True

    def _is_stop_requested(self) -> bool:
        return getattr(self, "_stop_requested", False)
    
    @staticmethod
    def has_pth_files(folder_path: Path) -> bool:
        try:
            p = Path(folder_path)
            if not p.exists() or not p.is_dir():
                return False
            return any(child.is_file() and child.suffix.lower() == ".pth" for child in p.iterdir())
        except Exception:
            logging.exception("Ошибка при проверке .pth файлов")
            return False

    def _emit_error(self, message: str):
        logging.error(message)
        try:
            self.error.emit(message)
        except Exception:
            logging.exception("Не удалось эмитить сигнал ошибки")

    def _validate_inputs(self) -> bool:
        if not self.model_path.exists():
            self._emit_error(f"Путь к модели не найден: {self.model_path}")
            return False
        if not self.data_path.exists():
            self._emit_error(f"Путь к данным не найден: {self.data_path}")
            return False
        if self.batch_size <= 0:
            self._emit_error("batch_size должен быть > 0")
            return False
        return True

    def run(self):
        """Главный метод — загружает модель, данные и делает предсказание."""
        try:
            self.progress.emit(0)

            if not self._validate_inputs():
                return  # _emit_error уже вызван

            if self._is_stop_requested():
                logging.info("Остановлено до начала работы.")
                return
            
            # Проверка наличия .pth (временная заглушка)
            if self.has_pth_files(self.model_path):
                # можно реализовать отдельную загрузку .pth
                msg = "Найдены .pth файлы, функционал загрузки .pth не реализован."
                self._emit_error(msg)
                return

            # Загрузка модели / токенайзера
            try:
                model, tokenizer, device, id2label = load_model_and_tokenizer(str(self.model_path), CLASS_NAMES)
                logging.info("Модель и токенизатор загружены.")
            except Exception as e:
                logging.exception("Ошибка загрузки модели/токенизатора")
                self._emit_error(f"Ошибка загрузки модели: {e}")
                return
            
            if self._is_stop_requested():
                logging.info("Остановлено после загрузки модели.")
                return

            # Загрузка данных
            try:
                if self.with_labels:
                    texts, raw_labels = load_data(self, str(self.data_path), sep=';')  # ваш load_data может отличаться
                # фильтрация по валидным меткам
                    filtered: List[Tuple[str, str]] = [
                        (t, l) for t, l in zip(texts, raw_labels)
                        if isinstance(l, str) and l in model.config.label2id
                    ]
                    if not filtered:
                        raise ValueError("Нет валидных строк с метками из label2id")
                    logging.info(f"Загружено {len(texts)} примеров, после фильтрации осталось {len(filtered)}.")
                    texts, true_labels_str = zip(*filtered)
                    true_labels = [model.config.label2id[l] for l in true_labels_str]
                else:
                    texts = load_data(self, str(self.data_path), sep=';')
                    true_labels = []
                    logging.info(f"Загружено {len(texts)} примеров (без меток).")
            except Exception as e:
                logging.exception("Ошибка загрузки/обработки данных")
                self._emit_error(f"Ошибка загрузки данных: {e}")
                return
            
            if self._is_stop_requested():
                logging.info("Остановлено после загрузки данных.")
                return
            
            # Предсказание с прогрессом
            try:
                model.eval()
                preds, probs, manual_flags = predict(self, texts, model, tokenizer, device, batch_size=self.batch_size)
                # Предполагаем, что predict уже эмитит прогресс через self.progress (как у вас было задумано).
                self.progress.emit(100)
                logging.info("Предсказание завершено.")
            except Exception as e:
                logging.exception("Ошибка при предсказании")
                self._emit_error(f"Ошибка предсказания: {e}")
                return

            # Метрики и отправка результата
            try:
                if self.with_labels:
                    if len(preds) < len(true_labels):
                        true_labels = true_labels[:len(preds)]

                    if len(preds) > 0:
                        acc = accuracy_score(true_labels, preds)
                        logging.info(f"Accuracy на предсказанных данных (в долях): {acc:.4f}")
                    else:
                        logging.warning("Нет предсказаний для вычисления метрик")
                    self.result.emit(list(texts), list(manual_flags), list(preds), list(probs), list(true_labels), id2label)
                else:
                    self.result.emit(list(texts), list(manual_flags), list(preds), list(probs), [], id2label)
            except Exception as e:
                logging.exception("Ошибка при отправке результата")
                self._emit_error(f"Ошибка отправки результата: {e}")
                return

        except Exception as e:
            logging.exception("Необработанная ошибка в run()")
            self._emit_error(str(e))
        finally:
            # Очистка VRAM (если использовался CUDA)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logging.exception("Ошибка при очистке CUDA cache")
            # гарантируем эмит finished
            try:
                self.finished.emit()
            except Exception:
                logging.exception("Не удалось эмитнуть finished")
            