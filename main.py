from datetime import datetime
import sys
import os
from pathlib import Path

import logging
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu
from PySide6.QtWidgets import QFileDialog, QListWidgetItem
from PySide6.QtCore import Qt, QObject, Signal, QThread
import pandas as pd
from ui_mainwindow import Ui_MainWindow

from model_inference import load_data, predict, load_model_and_tokenizer
from sklearn.metrics import accuracy_score, classification_report
from PySide6.QtCore import QObject, Signal, QSettings

from dotenv import load_dotenv

load_dotenv()

class_names = os.getenv("CLASS_NAMES").split(",")


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

    def run(self):
        """Метод, куда вынесена вся тяжёлая работа."""
        try:            
            if has_pth_files(self.model_path):
                # model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pth')
                # tokenizer_path = os.path.join(MODELS_DIR, f'{model_name}_tokenizer')

                # if (not os.path.exists(model_path) or
                # not os.path.exists(tokenizer_path)):
                #     raise FileNotFoundError(f'Модель {model_name} не найдена')

                # checkpoint = torch.load(model_path, map_location=self.device)
                # model_name_from_checkpoint = checkpoint['model_name']
                # num_classes = checkpoint['num_classes']

                # self.model = BERTClassifier(model_name_from_checkpoint, num_classes)
                # self.model.load_state_dict(checkpoint['model_state_dict'])
                # self.model.to(self.device)
                # self.model.eval()

                # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                # self.current_model_name = model_name
                ...
            else:
                try:
                    # Загрузка модели и токенизатора + device
                    model, tokenizer, device, id2label = load_model_and_tokenizer(
                        self.model_path, class_names
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

class EmittingStream(QObject):
    """
    Класс-обёртка для перенаправления sys.stdout/sys.stderr
    в сигнал textWritten.
    """
    textWritten = Signal(str)
    
    def write(self, text):
        # Вызывается, когда print() пишет сюда
        if text.strip():
            self.textWritten.emit(text)
            
    def flush(self):
        pass

def has_pth_files(folder_path: str) -> bool:
        """
        Возвращает True, если в папке есть хотя бы один файл с суффиксом .pth.
        """
        p = Path(folder_path)
        # Итерируемся по всем элементам папки и проверяем суффикс
        for child in p.iterdir():
            if child.is_file() and child.suffix.lower() == ".pth":
                return True
        return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("DSubbota", "MLBridge")
        self.default_model_dir = self.settings.value("default/model_dir", "")
        self.default_data_dir = self.settings.value("default/data_dir", "")
        # Инициализация UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Сюда будем складывать полные пути
        self.model_path = None
        self.data_path = None
        
        # Подключаем экшены меню
        self.ui.actionLoadFolder.triggered.connect(self.load_model_folder)
        self.ui.actionLoadZip.triggered.connect(self.load_model_zip)
        self.ui.actionLoadData.triggered.connect(self.load_data)
        self.ui.saveResults.clicked.connect(self.save_results)

        # Подключаем кнопку скачивания (если есть) и блокируем её
        self.ui.saveResults.setEnabled(False)
        self.ui.startInference.setEnabled(False)
        
        self.ui.listWidgetData.itemClicked.connect(self.on_item_selected)
        self.ui.listWidgetModel.itemClicked.connect(self.on_item_selected)
        
        self.ui.startInference.clicked.connect(self.inference)
        
        # ——————————————————————————————
        # Перенаправляем stdout/stderr в plainTextEdit
        self.log_widget = self.ui.inferenceLogs
        
        self.stream = EmittingStream()
        self.stream.textWritten.connect(self.log_widget.appendPlainText)
        
        sys.stdout = self.stream
        sys.stderr = self.stream
        
        # ——————————————————————————————
        # Настраиваем модуль logging
        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        
        # Загрузить сохранённые ранее пути
        self.load_lists()
        
        self._init_context_menus()
     
    def closeEvent(self, event):
        self.save_lists()
        super().closeEvent(event)
        
    def load_lists(self):
        """При старте читаем из настроек списки путей и заполняем виджеты."""
        # Модели
        model_paths = self.settings.value("lists/models", [], type=list)
        for path in model_paths:
            filename = os.path.basename(path)
            item = QListWidgetItem(f"Model: {filename}")
            item.setData(Qt.UserRole, path)
            self.ui.listWidgetModel.addItem(item)

        # Данные
        data_paths = self.settings.value("lists/data", [], type=list)
        for path in data_paths:
            filename = os.path.basename(path)
            item = QListWidgetItem(f"Data: {filename}")
            item.setData(Qt.UserRole, path)
            self.ui.listWidgetData.addItem(item)
            
    def save_lists(self):
        """Сохраняем из виджетов только списки путей."""
        model_paths = [
            self.ui.listWidgetModel.item(i).data(Qt.UserRole)
            for i in range(self.ui.listWidgetModel.count())
        ]
        self.settings.setValue("lists/models", model_paths)

        data_paths = [
            self.ui.listWidgetData.item(i).data(Qt.UserRole)
            for i in range(self.ui.listWidgetData.count())
        ]
        self.settings.setValue("lists/data", data_paths)    
    
    def open_context_menu(self, widget, pos):
        menu = QMenu()
        remove_act = menu.addAction("Удалить")
        action = menu.exec(widget.mapToGlobal(pos))
        if action == remove_act:
            for item in widget.selectedItems():
                widget.takeItem(widget.row(item))
            self.save_lists()  # сохраним сразу
                
    def _update_buttons_state(self):
        """
        Включаем startInference, когда заданы и модель, и данные.
        Сбрасываем saveResults, пока не будет нового инференса.
        """
        ready = bool(getattr(self, 'model_path', None) and getattr(self, 'data_path', None))
        self.ui.startInference.setEnabled(ready)
        self.ui.saveResults.setEnabled(False)

    def _init_context_menus(self):
        """Привязываем правый клик для удаления."""
        for widget in (self.ui.listWidgetModel, self.ui.listWidgetData):
            widget.setContextMenuPolicy(Qt.CustomContextMenu)
            widget.customContextMenuRequested.connect(
                lambda pos, w=widget: self.open_context_menu(w, pos)
            )
    
    def on_item_selected(self, item: QListWidgetItem):
        """
        Универсальный слот: обрабатывает клики по виджетам.
        self.sender() - это QListWidget, который эмитировал сигнал.
        """
        # Получаем полный путь
        path = item.data(Qt.UserRole)
        filename = os.path.basename(path)
        
        # Определяем тип: модель или данные
        sender = self.sender()
        if sender is self.ui.listWidgetModel:
            kind = "Модель"
            self.model_path = path
            # Обновляем метку
            self.ui.chosedModelInference.setText(
                f"{kind}: {filename}"
            )
        else:
            kind = "Данные"
            self.data_path = path
            self.ui.chosedDataInference.setText(
                f"{kind}: {filename}"
            )
        # каждый раз после выбора пересчитываем, можно ли начать инференс
        self._update_buttons_state()
       
            
    def inference(self):
        self.with_labels = self.ui.infWithLabels.isChecked()
        
        # Проверяем, что модель и данные выбраны
        if not getattr(self, 'model_path', None):
            logging.warning("Сначала выберите модель!")
            return

        if not getattr(self, 'data_path', None):
            logging.warning("Сначала выберите данные!")
            return
        
        self.thread = QThread()
        self.worker = InferenceWorker(
            model_path=self.model_path,
            data_path=self.data_path,
            batch_size=16,
            with_labels=self.with_labels
        )
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.ui.progressBar.setValue)
        self.worker.result.connect(self.on_inference_result)
        self.worker.error.connect(lambda msg: logging.critical("Ошибка! ", msg))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def on_inference_result(self, texts, preds, probs, true_labels, id2label):
        """
        Слот, вызываемый после завершения инференса.
        Сформирует self.results_df и разблокирует кнопку сохранения.
        """
        self.results_df = pd.DataFrame(
            {
                "text":     texts,
                "pred_label": [id2label[p] for p in preds]
            }
        )
        
        if self.with_labels:
            report_dict = classification_report(
                true_labels,
                preds,
                target_names=[id2label[i] for i in sorted(id2label)],
                output_dict=True
            )
            # Конвертируем в DataFrame и транспонируем для удобства
            self.df_report = pd.DataFrame(report_dict).transpose()
            logging.info("Сгенерирован classification report.")
        else:
            self.df_report = None
            logging.info("classification report пропущен (infWithLabels=False).")
            
        logging.info(f"Инференс завершён: предсказано {len(preds)} записей.")
        
        self.ui.saveResults.setEnabled(True)
        
    def save_results(self):
        """
        Слот для кнопки 'Сохранить результаты' - сохраняет self.results_df в Excel.
        """
        if not hasattr(self, 'results_df'):
            logging.warning("Нет данных. Сначала выполните инференс.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результаты в Excel",
            f"predictions_{datetime.today().strftime('%Y_%m_%d')}.xlsx",
            "Excel files (*.xlsx);;All files (*)"
        )
        if not path:
            return
        
        try:
            # Сохраняем
            with pd.ExcelWriter(path) as writer:
                self.results_df.to_excel(writer, sheet_name='Результаты', index=False)
                if self.df_report is not None:
                    self.df_report.to_excel(writer, sheet_name='Classification Report')
            self.ui.statusBar.showMessage(f"Таблица сохранена: {path}",  5000)
        except Exception as e:
            logging.critical(f"Ошибка сохранения! {e}")
        
    def load_model_zip(self):
        path = QFileDialog.getOpenFileName(
            self,
            "Выберите архив с моделью",
            "",
            "Archive files (*.zip);;All Files (*)"
        )
        if not path:
            return

        filename = os.path.basename(path)
        # Сохраняем полный путь
        self.model_path = path

        # Добавляем в QListWidget
        item = QListWidgetItem(f"Model: {filename}")
        item.setData(Qt.UserRole, path)
        self.ui.listWidgetModel.addItem(item)

        self.ui.statusBar.showMessage(f"Модель загружена: {filename}")

    def load_model_folder(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с моделью",
            "",
            QFileDialog.ShowDirsOnly
        )
        if not path:
            return

        filename = os.path.basename(path)
        # Сохраняем полный путь
        self.model_path = path

        # Добавляем в QListWidget
        item = QListWidgetItem(f"Model: {filename}")
        item.setData(Qt.UserRole, path)
        self.ui.listWidgetModel.addItem(item)

        self.ui.statusBar.showMessage(f"Модель загружена: {filename}")    
    
    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите данные",
            "",
            "Data files (*.csv *.xlsx);;All Files (*)"
        )
        if not path:
            return

        filename = os.path.basename(path)
        # Сохраняем полный путь
        self.data_path = path

        # Добавляем в QListWidget
        item = QListWidgetItem(f"Data: {filename}")
        item.setData(Qt.UserRole, path)
        self.ui.listWidgetData.addItem(item)

        self.ui.statusBar.showMessage(f"Данные загружены: {filename}")

            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
