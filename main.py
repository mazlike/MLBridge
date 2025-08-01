import sys, os, shutil
from datetime import datetime
from pathlib import Path
from send2trash import send2trash
import pandas as pd
from ui_mainwindow import Ui_MainWindow
import logging

from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QMessageBox, QFileDialog, QListWidgetItem
from PySide6.QtCore import Qt, QObject, Signal, QThread, QStandardPaths, QUrl 
from PySide6.QtGui import QIcon, QDesktopServices
from model_inference import load_data, load_model_and_tokenizer_pth, predict, load_model_and_tokenizer
from sklearn.metrics import accuracy_score, classification_report
import json

# Определяем базовый каталог приложения
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(__file__)

# Дефолтные папки внутри BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
DOWNLOADS_DIR = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
RESULTS_DIR = os.path.join(BASE_DIR, "results") 

# Путь к _internal/config.json
config_path = os.path.join(BASE_DIR, "config.json")

# Чтение и загрузка
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Используем переменные
class_names = config.get("class_names", [])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("icon256.ico"))
                
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(DATA_DIR,   exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Инициализация UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Сюда будем складывать полные пути
        self.model_path = None
        self.data_path = None
        
        # Подключаем экшены меню
        self.ui.actionLoadModel.triggered.connect(self.load_model_folder)
        self.ui.actionLoadData.triggered.connect(self.load_data)
        self.ui.saveResults.clicked.connect(self.save_results)
        self.ui.actionAbout.triggered.connect(self.show_about)
        self.ui.actionOpenAppFolder.triggered.connect(self.openAppFolder)
        self.ui.actionOpenResults.triggered.connect(self.openResultsFolder)
        
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
        self.ui.actionRefresh.clicked.connect(self.load_lists)
        
        self._init_context_menus()
    
    def show_about(self):
        QMessageBox.about(
            self,
            "О программе MLBridge",
            "<h2>MLBridge v1.0</h2>"
            "<p>Настольное приложение для инференса моделей.</p>"
            "<p>Автор: D.Subbota</p>"
            "<p>© 2025 Все права защищены.</p>"
        )
    
    def openAppFolder(self):
        """Открывает внешний файловый менеджер в папке с программой."""
        path = BASE_DIR  # константа, куда установлена/распакована программа
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def openResultsFolder(self):
        """Открывает внешний файловый менеджер в папке с результатами."""
        path = RESULTS_DIR  # константа для результатов
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    
    def load_lists(self):
        """Сканируем папки models/ и data/ и заполняем QListWidget’ы,
           сохраняя при этом выделение по именам."""
        # 1. Сохраняем названия выделенных элементов
        sel_models = {item.text() for item in self.ui.listWidgetModel.selectedItems()}
        sel_data   = {item.text() for item in self.ui.listWidgetData.selectedItems()}

        # 2. Блокируем обновления и сигналы, чтобы при добавлении не прыгал курсор
        for lw in (self.ui.listWidgetModel, self.ui.listWidgetData):
            lw.blockSignals(True)
            lw.setUpdatesEnabled(False)
            # дополнительно сбрасываем текущее выделение/строку,
            # чтобы Qt не выбирал автоматически какой-то элемент
            lw.clearSelection()
            lw.setCurrentRow(-1)

        # 3. Очищаем списки
        self.ui.listWidgetModel.clear()
        self.ui.listWidgetData.clear()

        # 4. Заполняем модели
        for entry in os.scandir(MODELS_DIR):
            if entry.is_dir():
                item = QListWidgetItem(entry.name)
                item.setData(Qt.UserRole, entry.path)
                self.ui.listWidgetModel.addItem(item)

        # 5. Заполняем данные
        for entry in os.scandir(DATA_DIR):
            if entry.name.lower().endswith((".csv", ".xlsx")):
                item = QListWidgetItem(entry.name)
                item.setData(Qt.UserRole, entry.path)
                self.ui.listWidgetData.addItem(item)

        # 6. Восстанавливаем выделение по имени
        for lw, sel in (
            (self.ui.listWidgetModel, sel_models),
            (self.ui.listWidgetData, sel_data)
        ):
            for i in range(lw.count()):
                item = lw.item(i)
                if item.text() in sel:
                    item.setSelected(True)

        # 7. Включаем обновления и сигналы обратно
        for lw in (self.ui.listWidgetModel, self.ui.listWidgetData):
            lw.setUpdatesEnabled(True)
            lw.blockSignals(False)

        # 8. Обновляем состояние кнопок
        self._update_buttons_state()

  
    
    def open_context_menu(self, widget, pos):
        item = widget.itemAt(pos)
        if not item or not item.data(Qt.UserRole):
            return

        menu = QMenu()
        remove_act = menu.addAction("Удалить (в корзину)")
        action = menu.exec(widget.mapToGlobal(pos))

        # Если нажали именно на «Удалить…»
        if action == remove_act:
            path = item.data(Qt.UserRole)
            try:
                send2trash(path)
                logging.info(f"Отправлено в корзину: {path}")
                # Только **тут** убираем из виджета
                widget.takeItem(widget.row(item))
            except Exception as e:
                logging.error(f"Не удалось отправить в корзину {path}: {e}")

            
                
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
        # Собираем имя файла с датой
        default_name = f"predictions_{datetime.today().strftime('%Y_%m_%d')}.xlsx"
        # Формируем полный путь внутри RESULTS_DIR
        default_path = os.path.join(RESULTS_DIR, default_name)
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результаты в Excel",
            default_path,
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

    def load_model_folder(self):
        """Выбираем исходную папку с моделью, копируем её в ./models и обновляем список."""
        src_folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с моделью",
            DOWNLOADS_DIR,            # стартуем из «Загрузок»
            QFileDialog.ShowDirsOnly
        )
        if not src_folder:
            return

        name = os.path.basename(src_folder)
        dst_folder = os.path.join(MODELS_DIR, name)

        try:
            # если такая папка уже есть — перезаписать
            if os.path.exists(dst_folder):
                shutil.rmtree(dst_folder)
            shutil.copytree(src_folder, dst_folder)
            logging.info(f"Скопирована модель в {dst_folder}")
        except Exception as e:
            logging.error(f"Не удалось скопировать модель: {e}")
            return

        # обновляем QListWidget
        self.load_lists()   
    
    def load_data(self):
        """Выбираем CSV/XLSX, копируем в ./data и обновляем список."""
        src_file, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл с данными",
            DOWNLOADS_DIR,
            "Data files (*.csv, *.xlsx);;All files (*)")
        if not src_file:
            return
        
        name = os.path.basename(src_file)
        dst_file = os.path.join(DATA_DIR, name)
        
        try:
            # если есть - перезаписать
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy2(src_file, dst_file)
            logging.info(f"Скопирован файл данных в {dst_file}")
        except Exception as e:
            logging.error(f"Не удалось скопировать данных: {e}")
            return
        
        self.load_lists()
            
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
                # try:
                #    model, tokenizer, device, id2label = load_model_and_tokenizer_pth(
                #         self.model_path, class_names
                #     )
                #    logging.info("Модель и токенизатор загружены.") 
                # except Exception as e:
                #     logging.error(f"Ошибка загрузки модели .pth {e}")
                #     return
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

            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
