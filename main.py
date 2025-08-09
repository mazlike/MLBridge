import sys, os, shutil, logging
import pandas as pd

from ui_mainwindow import Ui_MainWindow
from datetime import datetime

from send2trash import send2trash

from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QMessageBox, QFileDialog, QListWidgetItem
from PySide6.QtCore import Qt, QObject, Signal, QThread, QUrl 
from PySide6.QtGui import QIcon, QDesktopServices
from sklearn.metrics import classification_report

from workers import InferenceWorker
from config import BASE_DIR, MODELS_DIR, DATA_DIR, DOWNLOADS_DIR, RESULTS_DIR

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("icon256.ico"))
        self._init_dirs()
        
        # Инициализация UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Сюда будем складывать полные пути
        self.model_path = None
        self.data_path = None
        
        self._init_connections()
        self._init_logging()
        
        # Загрузить сохранённые ранее пути
        self.load_lists()
        self.ui.actionRefresh.clicked.connect(self.load_lists)
        
        self._init_context_menus()
        
    def _init_dirs(self):
        for d in (MODELS_DIR, DATA_DIR, RESULTS_DIR):
            os.makedirs(d, exist_ok=True)
        self.last_model_files = set()
        self.last_data_files = set()
    
    def _init_connections(self):
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
        
    def _init_logging(self):
        self.log_widget = self.ui.inferenceLogs
        self.stream = EmittingStream()
        self.stream.textWritten.connect(self.log_widget.appendPlainText)
        sys.stdout = sys.stderr = self.stream

        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(
            logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
        )
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    
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
    
    def _load_data(self, path):
        """Попытаться прочитать CSV/XLSX и вернуть DataFrame или None."""
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.csv':
                try:
                    df = pd.read_csv(path, low_memory=False, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(path, low_memory=False, encoding='cp1251')
            elif ext in ('.xls', '.xlsx'):
                df = pd.read_excel(path, engine='openpyxl')
            else:
                logging.warning("Неизвестный тип файла: %s", path)
                return None
            return df
        except Exception as e:
            logging.error("Ошибка при чтении файла данных %s: %s", path, e)
            return None

    
    def show_about(self):
        QMessageBox.about(
            self,
            "О программе MLBridge",
            "<h2>MLBridge v1.1</h2>"
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
        current_model_files = {
            f.name: f.path for f in os.scandir(MODELS_DIR) if f.is_dir()
        }
        current_data_files = {
            f.name: f.path for f in os.scandir(DATA_DIR) if f.name.lower().endswith((".csv", ".xlsx"))
        }

        # Если ничего не изменилось — просто выходим
        if set(current_model_files) == self.last_model_files and set(current_data_files) == self.last_data_files:
            return

        self.last_model_files = set(current_model_files)
        self.last_data_files = set(current_data_files)

        # Сохраняем выделения
        sel_models = {item.text() for item in self.ui.listWidgetModel.selectedItems()}
        sel_data   = {item.text() for item in self.ui.listWidgetData.selectedItems()}

        # Очищаем
        self.ui.listWidgetModel.clear()
        self.ui.listWidgetData.clear()

        # Заполняем заново
        for name, path in current_model_files.items():
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, path)
            self.ui.listWidgetModel.addItem(item)

        for name, path in current_data_files.items():
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, path)
            self.ui.listWidgetData.addItem(item)

        # Восстанавливаем выделение
        for lw, sel in (
            (self.ui.listWidgetModel, sel_models),
            (self.ui.listWidgetData, sel_data)
        ):
            for i in range(lw.count()):
                item = lw.item(i)
                if item.text() in sel:
                    item.setSelected(True)

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
                "БлокML": [id2label[p] for p in preds]
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
        else:
            self.df_report = None
            logging.info("classification report пропущен (без меток).")
            
        logging.info(f"Инференс завершён: предсказано {len(preds)} записей.")
        
        self.ui.saveResults.setEnabled(True)
        
    def save_results(self):
        """
        Слот для кнопки 'Сохранить результаты' - сохраняет self.results_df в Excel.
        Логика объединения одинаковая для случаев с меткой и без: если есть self.data,
        пробуем объединить её с self.results_df по колонкам (если длины совпадают),
        иначе сохраняем в разные листы.
        """
        if not hasattr(self, 'results_df'):
            logging.warning("Нет данных. Сначала выполните инференс.")
            return

        default_name = f"predictions_{datetime.today().strftime('%Y_%m_%d')}.xlsx"
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
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                data_df = self._load_data(self.data_path)
                
                if data_df is not None:
                    # Есть исходные данные — пробуем объединить
                    try:
                        combined = pd.concat(
                            [data_df.reset_index(drop=True), self.results_df.reset_index(drop=True)],
                            axis=1
                        )
                        # --- добавляем колонку "Ручная обработка" ---
                        if 'Текст' in combined.columns:
                            combined['Ручная обработка'] = combined['Текст'].astype(str).apply(
                                lambda x: 'Нужна' if 'фото' in x.lower() else ''
                            )
                        else:
                            logging.warning("Колонка 'Текст' не найдена, пропуск пометки 'Ручная обработка'")
                        combined.to_excel(writer, sheet_name='Данные', index=False)
                    except Exception as ex:
                        logging.exception("Ошибка при объединении data и results_df: %s. Сохраняю отдельно.", ex)
                        data_df.to_excel(writer, sheet_name='Данные', index=False)
                        # --- добавляем колонку "Ручная обработка" ---
                        if 'Текст' in data_df.columns:
                            data_df['Ручная обработка'] = data_df['Текст'].astype(str).apply(
                                lambda x: 'Нужна' if 'фото' in x.lower() else ''
                            )
                        else:
                            logging.warning("Колонка 'Текст' не найдена, пропуск пометки 'Ручная обработка'")
                        self.results_df.to_excel(writer, sheet_name='Результаты', index=False)
                else:
                    # Нет исходных данных — поведение как раньше
                    self.results_df.to_excel(writer, sheet_name='Результаты', index=False)

                # Сохраняем отчет если есть
                if self.df_report is not None:
                    self.df_report.to_excel(writer, sheet_name='Classification Report')

            self.ui.statusBar.showMessage(f"Таблица сохранена: {path}", 5000)
        except Exception as e:
            logging.critical(f"Ошибка сохранения! {e}", exc_info=True)


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
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
