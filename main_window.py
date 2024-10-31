import sys
import os
import pandas as pd
from datetime import date
import numpy as np
import h5py
from scipy import signal, stats
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QToolBar, QListView, QFileDialog, QPushButton, QMessageBox,
                             QLineEdit, QLabel, QTreeView, QSplitter, QStatusBar, QProgressBar, QStyleFactory)
from PyQt6.QtGui import QIcon, QFileSystemModel, QStandardItemModel, QStandardItem, QFont, QPalette, QColor
from PyQt6.QtCore import Qt, QSize, QDir, pyqtSignal, QObject
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from nptdms import TdmsFile
from preprocess_window import PreprocessWindow
from peak_detector_window import PeakDetectorWindow
from h5_reader_window import H5ReaderWindow
import logging
import traceback

logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def exception_hook(exctype, value, tb):
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    print(f"An error occurred: {error_msg}")
    QMessageBox.critical(None, "Error", f"An unexpected error occurred:\n\n{error_msg}")
    sys.exit(1)

sys.excepthook = exception_hook

class FileUpdateSignal(QObject):
    updated = pyqtSignal(list)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TDMS Analysis App")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setup_ui()
        self.initialize_attributes()
        self.set_default_directory()

    def setup_ui(self):
        self.setStyle(QStyleFactory.create("Fusion"))
        self.setPalette(self.create_white_palette())

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_widget = self.setup_left_panel()
        self.plot_widget = self.setup_center_panel()
        right_widget = self.setup_right_panel()

        splitter.addWidget(left_widget)
        splitter.addWidget(self.plot_widget)
        splitter.addWidget(right_widget)

        splitter.setSizes([200, 600, 200])

        self.create_toolbar()
        self.setup_status_bar()

    def setup_left_panel(self):
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        nav_layout = QHBoxLayout()
        self.setup_navigation_buttons(nav_layout)
        left_layout.addLayout(nav_layout)

        self.setup_file_system_view(left_layout)

        return left_widget

    def setup_navigation_buttons(self, layout):
        self.back_button = QPushButton("←")
        self.forward_button = QPushButton("→")
        
        button_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 4px;
                font-size: 16px;
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BBDEFB;
            }
        """
        self.back_button.setStyleSheet(button_style)
        self.forward_button.setStyleSheet(button_style)
        
        self.back_button.clicked.connect(self.go_back)
        self.forward_button.clicked.connect(self.go_forward)
        
        self.back_button.setToolTip("Go back")
        self.forward_button.setToolTip("Go forward")
        
        layout.addWidget(self.back_button)
        layout.addWidget(self.forward_button)
        
        self.current_dir_label = QLabel()
        self.current_dir_label.setStyleSheet("font-weight: bold; color: #333; margin-left: 10px;")
        layout.addWidget(self.current_dir_label)
        
        layout.addStretch()

    def setup_file_system_view(self, layout):
        self.file_system_model = QFileSystemModel()
        self.file_system_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self.file_system_model.setNameFilters(["*.tdms", "*.h5"])
        self.file_system_model.setNameFilterDisables(False)

        self.left_file_list = QListView()
        self.left_file_list.setModel(self.file_system_model)
        self.left_file_list.doubleClicked.connect(self.on_item_double_clicked)
        self.left_file_list.setStyleSheet("""
            QListView {
                background-color: white;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
            }
            QListView::item:selected {
                background-color: #E3F2FD;
                color: black;
            }
            QListView::item:hover {
                background-color: #F5F5F5;
            }
            QListView::item:selected:active {
                background-color: #2196F3;
                color: white;
            }
        """)
        layout.addWidget(self.left_file_list)

    def setup_center_panel(self):
        plot_widget = QWidget()
        self.plot_layout = QVBoxLayout()
        plot_widget.setLayout(self.plot_layout)

        control_panel = self.setup_control_panel()
        self.plot_layout.addWidget(control_panel)

        self.setup_plot()

        return plot_widget

    def setup_control_panel(self):
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedHeight(40)  # 增加高度到40像素

        # Downsample factor
        control_layout.addWidget(QLabel("Downsample:"))
        self.downsample_input = QLineEdit("10")
        self.downsample_input.setFixedWidth(50)  # 稍微增加宽度
        self.downsample_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px;
                font-size: 12px;
            }
        """)
        self.downsample_input.textChanged.connect(self.update_plot)
        control_layout.addWidget(self.downsample_input)

        control_layout.addSpacing(20)  # 增加间距

        # 60Hz Filter
        self.filter_button = QPushButton("60Hz Filter: ON")
        self.filter_button.setCheckable(True)
        self.filter_button.setChecked(True)
        self.filter_button.clicked.connect(self.toggle_filter)
        self.style_toggle_button(self.filter_button)
        control_layout.addWidget(self.filter_button)

        control_layout.addSpacing(20)  # 增加间距

        # Remove Outliers
        self.outlier_button = QPushButton("Remove Outliers: ON")
        self.outlier_button.setCheckable(True)
        self.outlier_button.setChecked(True)
        self.outlier_button.clicked.connect(self.toggle_outlier_removal)
        self.style_toggle_button(self.outlier_button)
        control_layout.addWidget(self.outlier_button)

        control_layout.addStretch()
        return control_panel

    def style_toggle_button(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:checked {
                background-color: #F44336;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
        """)

    def setup_plot(self):
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'Arial',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 1.5,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'axes.edgecolor': '#888888',
            'axes.linewidth': 1,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
        })

        self.figure = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1, 3])  # 使用2行，比例为1:3
        self.ax_voltage = self.figure.add_subplot(gs[0, 0])
        self.ax_current = self.figure.add_subplot(gs[1, 0], sharex=self.ax_voltage)

        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout.addWidget(self.toolbar)

    def setup_right_panel(self):
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        self.add_to_list_button = QPushButton("Add Current File to List")
        self.add_to_list_button.clicked.connect(self.add_current_file_to_list)
        self.add_to_list_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        right_layout.addWidget(self.add_to_list_button)

        self.right_file_list = QTreeView()
        self.right_file_model = QStandardItemModel()
        self.right_file_list.setModel(self.right_file_model)
        self.right_file_model.setHorizontalHeaderLabels(['Files'])
        self.right_file_list.setStyleSheet("""
            QTreeView {
                background-color: white;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
            }
            QTreeView::item:selected {
                background-color: #E3F2FD;
                color: black;
            }
            QTreeView::item:hover {
                background-color: #F5F5F5;
            }
        """)
        right_layout.addWidget(self.right_file_list)

        file_info_widget = self.setup_file_info_panel()
        right_layout.addWidget(file_info_widget)

        return right_widget

    def setup_file_info_panel(self):
        file_info_widget = QWidget()
        file_info_layout = QVBoxLayout()
        file_info_widget.setLayout(file_info_layout)

        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("""
            QLabel {
                background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        file_info_layout.addWidget(self.file_info_label)

        button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.add_button = QPushButton("Add")
        self.delete_button = QPushButton("Delete")

        for button in [self.prev_button, self.next_button, self.add_button, self.delete_button]:
            button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 5px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)

        self.prev_button.clicked.connect(self.load_previous_file)
        self.next_button.clicked.connect(self.load_next_file)
        self.add_button.clicked.connect(self.add_current_file_to_list)
        self.delete_button.clicked.connect(self.delete_current_file_from_list)

        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.delete_button)

        file_info_layout.addLayout(button_layout)

        return file_info_widget

    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #F5F5F5;
                color: #333333;
            }
        """)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                width: 10px;
                margin: 0.5px;
            }
        """)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()

    def create_toolbar(self):
        self.toolbar = QToolBar()
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: white;
                border-bottom: 1px solid #E0E0E0;
                spacing: 5px;
            }
            QToolButton {
                background-color: white;
                border: none;
                padding: 5px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background-color: #F5F5F5;
            }
        """)

        import_action = self.toolbar.addAction(QIcon("icons/import.png"), "Import")
        import_action.triggered.connect(self.import_files)

        preprocess_action = self.toolbar.addAction(QIcon("icons/preprocess.png"), "Preprocess")
        preprocess_action.triggered.connect(self.open_preprocess_window)

        peak_detector_action = self.toolbar.addAction(QIcon("icons/peak_detector.png"), "Peak Detector")
        peak_detector_action.triggered.connect(self.open_peak_detector_window)

        h5_reader_action = self.toolbar.addAction(QIcon("icons/h5_reader.png"), "H5 Reader")
        h5_reader_action.triggered.connect(self.open_h5_reader_window)

    def initialize_attributes(self):
        self.current_file_index = None
        self.history = []
        self.current_index = -1
        self.current_file = None
        self.current_time_voltage = None
        self.current_voltage = None
        self.current_time_current = None
        self.current_current = None
        self.filter_on = True
        self.remove_outliers_on = True
        self.file_update_signal = FileUpdateSignal()
        self.preprocess_window = None
    
    def create_white_palette(self):
        palette = QPalette()
        white = QColor(255, 255, 255)
        palette.setColor(QPalette.ColorRole.Window, white)
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Base, white)
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ColorRole.ToolTipBase, white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Button, white)
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        return palette

########################################左侧列表#########################################################

    def set_default_directory(self):
        default_path = r"H:\Experiment"
        if os.path.exists(default_path) and os.path.isdir(default_path):
            self.set_root_directory(default_path)
        else:
            QMessageBox.warning(self, "Warning",
                                f"Default directory '{default_path}' not found. Please select a directory manually.")
            self.import_files()

    def import_files(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        directory = file_dialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.set_root_directory(directory)

    def set_root_directory(self, directory):
        self.file_system_model.setRootPath(directory)
        self.left_file_list.setRootIndex(self.file_system_model.index(directory))
        self.add_to_history(directory)
        self.status_bar.showMessage(f"Current directory: {directory}", 5000)

    def on_item_double_clicked(self, index):
        path = os.path.abspath(self.file_system_model.filePath(index))
        logging.info(f"Double-clicked item: {path}")
        if os.path.isdir(path):
            self.left_file_list.setRootIndex(index)
            self.add_to_history(path)
        elif path.lower().endswith('.tdms'):
            logging.info(f"Attempting to read TDMS file: {path}")
            self.read_tdms_file(path)
        elif path.lower().endswith('.h5'):
            logging.info(f"Attempting to read H5 file: {path}")
            self.read_h5_file(path)
        else:
            logging.warning(f"Unsupported file type: {path}")
    
    def highlight_current_file(self):
        if self.current_file:
            index = self.file_system_model.index(self.current_file)
            self.left_file_list.setCurrentIndex(index)
            self.left_file_list.scrollTo(index)

    def add_to_history(self, path):
        self.current_index += 1
        self.history = self.history[:self.current_index]
        self.history.append(path)
        self.update_navigation_buttons()

    def go_back(self):
        if self.current_index > 0:
            self.current_index -= 1
            path = self.history[self.current_index]
            self.left_file_list.setRootIndex(self.file_system_model.index(path))
            self.update_navigation_buttons()

    def go_forward(self):
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            path = self.history[self.current_index]
            self.left_file_list.setRootIndex(self.file_system_model.index(path))
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        self.back_button.setEnabled(self.current_index > 0)
        self.forward_button.setEnabled(self.current_index < len(self.history) - 1)
        
        # Update current directory label
        if self.history:
            self.current_dir_label.setText(os.path.basename(self.history[self.current_index]))

    def read_tdms_file(self, file_path):
        try:
            # 确保文件路径是绝对路径
            file_path = os.path.abspath(file_path)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
            
            logging.info(f"Attempting to read TDMS file: {file_path}")
            
            with TdmsFile.open(file_path) as tdms_file:
                voltage = None
                current = None
                time = None

                for group in tdms_file.groups():
                    for channel in group.channels():
                        if channel.name == 'Dev2/ai0':
                            voltage = channel[:]
                            time = np.arange(len(voltage)) * channel.properties.get('wf_increment', 1)
                        elif channel.name == 'Dev2/ai1':
                            current = channel[:]

                if voltage is None or current is None:
                    raise ValueError("Dev2/ai0 or Dev2/ai1 channel not found in the TDMS file")

                if time is None:
                    time = np.arange(len(voltage))

                self.current_file = file_path
                self.current_time_voltage = time
                self.current_voltage = voltage
                self.current_time_current = time
                self.current_current = current

                files_in_directory = self.get_data_files_in_directory()
                try:
                    self.current_file_index = files_in_directory.index(file_path)
                except ValueError:
                    logging.warning(f"File {file_path} not found in the current directory list. Setting index to 0.")
                    self.current_file_index = 0

                self.update_plot()
                self.update_file_info()
                self.highlight_current_file()
                
                file_name = os.path.basename(file_path)
                self.status_bar.showMessage(f"Loaded file: {file_name}", 5000)
                logging.info(f"Successfully loaded TDMS file: {file_name}")

        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            logging.error(error_msg)
        except ValueError as e:
            error_msg = f"Invalid TDMS file format: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            logging.error(error_msg)
        except Exception as e:
            error_msg = f"Failed to read TDMS file: {str(e)}\n\nFile path: {file_path}\n\nFull traceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
            logging.error(error_msg)
        finally:
            # 确保在出现错误时清除任何部分加载的数据
            if not hasattr(self, 'current_file') or self.current_file != file_path:
                self.current_file = None
                self.current_time_voltage = None
                self.current_voltage = None
                self.current_time_current = None
                self.current_current = None
                self.current_file_index = None
        
    def update_file_info(self):
        if self.current_file:
            file_name = os.path.basename(self.current_file)
            file_size = os.path.getsize(self.current_file) / (1024 * 1024)  # Size in MB
            creation_time = os.path.getctime(self.current_file)
            creation_date = date.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            
            info_text = f"<b>File:</b> {file_name}<br><b>Size:</b> {file_size:.2f} MB<br><b>Created:</b> {creation_date}"
            self.file_info_label.setText(info_text)
            self.file_info_label.setTextFormat(Qt.TextFormat.RichText)
        else:
            self.file_info_label.setText("No file selected")

    def load_previous_file(self):
        if self.current_file_index is not None and self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_file_at_index(self.current_file_index)

    def load_next_file(self):
        files_in_directory = self.get_data_files_in_directory()
        if self.current_file_index is not None and self.current_file_index < len(files_in_directory) - 1:
            self.current_file_index += 1
            self.load_file_at_index(self.current_file_index)

    def load_file_at_index(self, index):
        files_in_directory = self.get_data_files_in_directory()
        if 0 <= index < len(files_in_directory):
            file_path = files_in_directory[index]
            self.read_tdms_file(file_path)

    def get_data_files_in_directory(self):
        if self.current_file:
            directory = os.path.dirname(os.path.abspath(self.current_file))
        else:
            directory = os.path.abspath(self.file_system_model.filePath(self.left_file_list.rootIndex()))
        
        data_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.tdms', '.h5')):
                    data_files.append(os.path.abspath(os.path.join(root, file)))
        
        logging.info(f"Data files in directory {directory}: {data_files}")
        return data_files

    def delete_current_file_from_list(self):
        if self.current_file:
            file_name = os.path.basename(self.current_file)
            folder_name = os.path.basename(os.path.dirname(self.current_file))
            experiment_name = os.path.basename(os.path.dirname(os.path.dirname(self.current_file)))

            # Find and remove the file from the right file list
            experiment_item = self.find_item_by_text(self.right_file_model, experiment_name)
            if experiment_item:
                folder_item = self.find_item_by_text(experiment_item, folder_name)
                if folder_item:
                    for row in range(folder_item.rowCount()):
                        file_item = folder_item.child(row)
                        if file_item.text() == file_name:
                            folder_item.removeRow(row)
                            break

            # Update CSV files
            self.update_csv_files(experiment_name, folder_name)

            # Update preprocess window
            self.update_preprocess_window()

    def read_h5_file(self, file_path):
        try:
            with h5py.File(file_path, 'r') as h5_file:
                voltage = None
                current = None
                time = None

                # 假设电压、电流和时间数据存储在特定的数据集中
                # 您可能需要根据实际的H5文件结构调整这些键
                if 'voltage' in h5_file:
                    voltage = h5_file['voltage'][:]
                if 'current' in h5_file:
                    current = h5_file['current'][:]
                if 'time' in h5_file:
                    time = h5_file['time'][:]

                if voltage is None or current is None:
                    raise ValueError("Voltage or current data not found in the H5 file")

                if time is None:
                    time = np.arange(len(voltage))

                self.current_file = file_path
                self.current_time_voltage = time
                self.current_voltage = voltage
                self.current_time_current = time
                self.current_current = current

                self.update_plot()
                self.update_file_info()
                self.highlight_current_file()
                
                file_name = os.path.basename(file_path)
                self.status_bar.showMessage(f"Loaded H5 file: {file_name}", 5000)
                logging.info(f"Successfully loaded H5 file: {file_name}")

        except Exception as e:
            error_msg = f"Failed to read H5 file: {str(e)}\n\nFile path: {file_path}\n\nFull traceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
            logging.error(error_msg)
            self.clear_current_data()
    
    def clear_current_data(self):
        self.current_file = None
        self.current_time_voltage = None
        self.current_voltage = None
        self.current_time_current = None
        self.current_current = None
        self.current_file_index = None

    def find_item_by_text(self, parent_item, text):
        if isinstance(parent_item, QStandardItemModel):
            for row in range(parent_item.rowCount()):
                item = parent_item.item(row)
                if item.text() == text:
                    return item
        else:
            for row in range(parent_item.rowCount()):
                item = parent_item.child(row)
                if item.text() == text:
                    return item
        return None

###############################画图###################################################
    def toggle_filter(self):
        self.filter_on = self.filter_button.isChecked()
        self.filter_button.setText(f"60Hz Filter: {'ON' if self.filter_on else 'OFF'}")
        self.update_plot()

    def apply_60hz_filter(self, data, fs):
        # Design a notch filter
        f0 = 60.0  # Frequency to be removed
        Q = 30.0  # Quality factor
        b, a = signal.iirnotch(f0, Q, fs)

        # Apply the filter
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data

    def toggle_outlier_removal(self):
        self.remove_outliers_on = self.outlier_button.isChecked()
        self.outlier_button.setText(f"Remove Outliers: {'ON' if self.remove_outliers_on else 'OFF'}")
        self.update_plot()

    def remove_outliers(self, data, time, threshold=3.5):
        # Use Z-score for outlier detection
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores < threshold

        # Remove outliers
        cleaned_data = data[outlier_mask]
        cleaned_time = time[outlier_mask]

        # Debug information
        print(f"Original data points: {len(data)}")
        print(f"Cleaned data points: {len(cleaned_data)}")
        print(f"Removed {len(data) - len(cleaned_data)} outliers")

        return cleaned_time, cleaned_data

    def update_plot(self):
        if self.current_voltage is None or self.current_current is None:
            return

        try:
            downsample = int(self.downsample_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer for downsampling.")
            return

        self.progress_bar.show()
        self.progress_bar.setValue(0)

        time_voltage = self.current_time_voltage[::downsample]
        voltage = self.current_voltage[::downsample]
        time_current = self.current_time_current[::downsample]
        current = self.current_current[::downsample]

        self.progress_bar.setValue(25)

        if self.remove_outliers_on:
            time_voltage, voltage = self.remove_outliers(voltage, time_voltage)
            time_current, current = self.remove_outliers(current, time_current)

        self.progress_bar.setValue(50)

        if self.filter_on:
            fs = 1.0 / (time_current[1] - time_current[0])
            current = self.apply_60hz_filter(current, fs)

        self.progress_bar.setValue(75)

        self.plot_data(time_voltage, voltage, time_current, current)

        self.progress_bar.setValue(100)
        self.progress_bar.hide()

    def plot_data(self, time_voltage, voltage, time_current, current):
        self.ax_voltage.clear()
        self.ax_current.clear()

        self.ax_voltage.plot(time_voltage, voltage, color='#1E90FF', linewidth=1.5)
        self.ax_voltage.set_ylabel('Voltage (V)', fontweight='bold')
        self.ax_voltage.tick_params(axis='both', which='major', labelsize=8)
        self.ax_voltage.label_outer()

        self.ax_current.plot(time_current, current, color='#FF6347', linewidth=1.5)
        self.ax_current.set_xlabel('Time (s)', fontweight='bold')
        self.ax_current.set_ylabel('Current (nA)', fontweight='bold')
        self.ax_current.tick_params(axis='both', which='major', labelsize=8)

        if self.ax_voltage.get_legend():
            self.ax_voltage.get_legend().remove()
        if self.ax_current.get_legend():
            self.ax_current.get_legend().remove()

        x_min = min(time_voltage.min(), time_current.min())
        x_max = max(time_voltage.max(), time_current.max())
        self.ax_voltage.set_xlim(x_min, x_max)
        self.ax_current.set_xlim(x_min, x_max)

        for ax in [self.ax_voltage, self.ax_current]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='both', length=0)
            ax.grid(True, linestyle=':', alpha=0.6)

        self.figure.tight_layout()
        self.canvas.draw()

#########################右列表文件#############################

    def add_current_file_to_list(self):
        if self.current_file:
            file_name = os.path.basename(self.current_file)
            folder_name = os.path.basename(os.path.dirname(self.current_file))
            experiment_name = os.path.basename(os.path.dirname(os.path.dirname(self.current_file)))

            # Check if the experiment already exists in the tree
            experiment_item = None
            for i in range(self.right_file_model.rowCount()):
                item = self.right_file_model.item(i)
                if item.text() == experiment_name:
                    experiment_item = item
                    break

            # If experiment doesn't exist, create it
            if experiment_item is None:
                experiment_item = QStandardItem(experiment_name)
                self.right_file_model.appendRow(experiment_item)

            # Check if the folder exists under the experiment
            folder_item = None
            for i in range(experiment_item.rowCount()):
                item = experiment_item.child(i)
                if item.text() == folder_name:
                    folder_item = item
                    break

            # If folder doesn't exist, create it
            if folder_item is None:
                folder_item = QStandardItem(folder_name)
                experiment_item.appendRow(folder_item)

            # Check if the file already exists in the folder
            for i in range(folder_item.rowCount()):
                if folder_item.child(i).text() == file_name:
                    logging.info(
                        f"File '{file_name}' already exists in folder '{folder_name}' under experiment '{experiment_name}'. Not adding duplicate.")
                    QMessageBox.information(self, "Duplicate File",
                                            f"The file '{file_name}' is already in the folder '{folder_name}' under experiment '{experiment_name}'.")
                    return

            if self.preprocess_window and self.preprocess_window.isVisible():
                self.update_preprocess_window()


            # Add the file to the folder
            file_item = QStandardItem(file_name)
            file_item.setToolTip(self.current_file)
            folder_item.appendRow(file_item)

            logging.info(f"Added file to list: {self.current_file}")

            # Update CSV files
            self.update_csv_files(experiment_name, folder_name)

            # Update preprocess window
            self.update_preprocess_window()
        else:
            logging.warning("Attempted to add file to list, but no current file selected")

##################################EXCELFILE##################################################################

    def update_csv_files(self, experiment_name, folder_name):
        try:
            if not self.current_file:
                return

            logging.info(f"Updating CSV files for {self.current_file}")

            # Get the directory of the current file
            current_dir = os.path.dirname(self.current_file)
            parent_dir = os.path.dirname(os.path.dirname(current_dir))

            # Create ProcessingData folder if it doesn't exist
            processing_data_dir = os.path.join(current_dir, "ProcessingData")
            os.makedirs(processing_data_dir, exist_ok=True)

            # Update PreProcess.xlsx for the specific folder
            preprocess_file = os.path.join(processing_data_dir, "PreProcess.xlsx")
            self.update_preprocess_csv(preprocess_file, experiment_name, folder_name)

            # Update Processing List.xlsx in the parent directory
            processing_list_file = os.path.join(parent_dir, "Processing List.xlsx")
            self.update_processing_list_csv(processing_list_file)

            logging.info("CSV files updated successfully")
        except Exception as e:
            logging.error(f"Error in update_csv_files: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred while updating CSV files: {str(e)}")

    def update_preprocess_csv(self, file_path, experiment_name, folder_name):
        try:
            today = date.today().strftime("%Y-%m-%d")

            # Read existing data or create new DataFrame
            try:
                with pd.ExcelFile(file_path) as xls:
                    existing_df = pd.read_excel(xls, sheet_name=None)
            except FileNotFoundError:
                existing_df = {today: pd.DataFrame(columns=['序号', '项目名称', '添加日期', '备注'])}

            if today not in existing_df:
                existing_df[today] = pd.DataFrame(columns=['序号', '项目名称', '添加日期', '备注'])

            df = existing_df[today]

            # Get files for the specific folder under the specific experiment
            experiment_item = None
            for i in range(self.right_file_model.rowCount()):
                item = self.right_file_model.item(i)
                if item.text() == experiment_name:
                    experiment_item = item
                    break

            if experiment_item:
                folder_item = None
                for i in range(experiment_item.rowCount()):
                    item = experiment_item.child(i)
                    if item.text() == folder_name:
                        folder_item = item
                        break

                if folder_item:
                    files = [folder_item.child(j).text() for j in range(folder_item.rowCount())]

                    # Prepare new data
                    new_data = []
                    for file_name in files:
                        if file_name not in df['项目名称'].values:
                            new_data.append({
                                '序号': len(df) + len(new_data) + 1,
                                '项目名称': file_name,
                                '添加日期': today,
                                '备注': ''
                            })

                    # Add new data if any
                    if new_data:
                        new_df = pd.DataFrame(new_data)
                        df = pd.concat([df, new_df], ignore_index=True)
                        df['序号'] = range(1, len(df) + 1)  # Reorder 序号

                        # Update the existing DataFrame
                        existing_df[today] = df

                        # Write to Excel file, preserving other sheets
                        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                            for sheet_name, sheet_df in existing_df.items():
                                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

                    logging.info(
                        f"PreProcess.xlsx updated for folder {folder_name} under experiment {experiment_name} with {len(new_data)} new entries")
                else:
                    logging.warning(f"Folder {folder_name} not found in experiment {experiment_name}")
            else:
                logging.warning(f"Experiment {experiment_name} not found in the file list")

        except Exception as e:
            logging.error(f"Error updating PreProcess CSV: {str(e)}", exc_info=True)

    def update_processing_list_csv(self, file_path):
        try:
            today = date.today().strftime("%Y-%m-%d")

            # Read existing data or create new DataFrame
            try:
                with pd.ExcelFile(file_path) as xls:
                    existing_df = pd.read_excel(xls, sheet_name=None)
            except FileNotFoundError:
                existing_df = {
                    today: pd.DataFrame(columns=['序号', '实验编号', '文件夹名称', '文件名', '添加日期', '备注'])}

            if today not in existing_df:
                existing_df[today] = pd.DataFrame(
                    columns=['序号', '实验编号', '文件夹名称', '文件名', '添加日期', '备注'])

            df = existing_df[today]

            # Prepare new data
            updated_data = []
            for i in range(self.right_file_model.rowCount()):
                experiment_item = self.right_file_model.item(i)
                experiment_name = experiment_item.text()
                for j in range(experiment_item.rowCount()):
                    folder_item = experiment_item.child(j)
                    folder_name = folder_item.text()
                    for k in range(folder_item.rowCount()):
                        file_name = folder_item.child(k).text()

                        # Check if this file already exists
                        existing_row = df[(df['实验编号'] == experiment_name) &
                                          (df['文件夹名称'] == folder_name) &
                                          (df['文件名'] == file_name)]
                        if existing_row.empty:
                            # Add new row
                            updated_data.append({
                                '序号': len(df) + len(updated_data) + 1,
                                '实验编号': experiment_name,
                                '文件夹名称': folder_name,
                                '文件名': file_name,
                                '添加日期': today,
                                '备注': ''
                            })
                        else:
                            # Update existing row
                            df.loc[(df['实验编号'] == experiment_name) &
                                   (df['文件夹名称'] == folder_name) &
                                   (df['文件名'] == file_name), '添加日期'] = today

            # Add new data if any
            if updated_data:
                new_df = pd.DataFrame(updated_data)
                df = pd.concat([df, new_df], ignore_index=True)

            df['序号'] = range(1, len(df) + 1)  # Reorder 序号

            # Sort the DataFrame
            df = df.sort_values(by=['实验编号', '文件夹名称', '文件名'])

            # Update the existing DataFrame
            existing_df[today] = df

            # Write to Excel file, preserving other sheets
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name, sheet_df in existing_df.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

            logging.info(
                f"Processing List.xlsx updated with {len(updated_data)} new entries and existing entries updated")
        except Exception as e:
            logging.error(f"Error updating Processing List CSV: {str(e)}", exc_info=True)


    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.XButton1:
            self.go_back()
        elif event.button() == Qt.MouseButton.XButton2:
            self.go_forward()
        else:
            super().mousePressEvent(event)

#################################Preprocess_window############################################################

    def open_preprocess_window(self):
        try:
            logging.info("Attempting to open preprocess window")
            logging.debug(f"self: {self}")

            if self.preprocess_window is None or not self.preprocess_window.isVisible():
                logging.info("Creating new PreprocessWindow")
                file_list = self.get_file_list()
                logging.debug(f"File list: {file_list}")

                logging.debug("Initializing PreprocessWindow")
                self.preprocess_window = PreprocessWindow(self, file_list)
                logging.info("PreprocessWindow initialized successfully")

                logging.debug("Connecting signals")
                self.preprocess_window.closed.connect(self.on_preprocess_window_closed)
                self.file_update_signal.updated.connect(self.preprocess_window.update_file_list)
                logging.info("Signals connected successfully")

            logging.debug("Showing PreprocessWindow")
            self.preprocess_window.show()
            logging.info("Preprocess window opened successfully")
        except Exception as e:
            logging.error(f"Error opening preprocess window: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open preprocess window: {str(e)}")

    def get_file_list(self):
        file_list = []
        for i in range(self.right_file_model.rowCount()):
            experiment_item = self.right_file_model.item(i)
            experiment_name = experiment_item.text()
            for j in range(experiment_item.rowCount()):
                folder_item = experiment_item.child(j)
                folder_name = folder_item.text()
                for k in range(folder_item.rowCount()):
                    file_item = folder_item.child(k)
                    file_name = file_item.text()
                    full_path = file_item.toolTip()
                    file_list.append((experiment_name, folder_name, file_name, full_path))
        return file_list

    def update_preprocess_window(self):
        try:
            file_list = self.get_file_list()
            self.file_update_signal.updated.emit(file_list)
            logging.info(f"Sent update to preprocess window with {len(file_list)} files")
        except Exception as e:
            logging.error(f"Error updating preprocess window: {str(e)}", exc_info=True)

    def on_preprocess_window_closed(self):
        logging.info("Preprocess window closed")
        self.preprocess_window = None

#####################################Peak Detector Window##############################################

    def open_peak_detector_window(self):
        try:
            logging.info("Attempting to open peak detector window")
            if not hasattr(self, 'peak_detector_window') or not self.peak_detector_window.isVisible():
                logging.info("Creating new PeakDetectorWindow")
                current_path = self.file_system_model.filePath(self.left_file_list.rootIndex())
                logging.debug(f"Current path: {current_path}")

                logging.debug("Initializing PeakDetectorWindow")
                self.peak_detector_window = PeakDetectorWindow(None, self.file_system_model,
                                                               current_path)  # Set parent to None
                logging.info("PeakDetectorWindow initialized successfully")

            logging.debug("Showing PeakDetectorWindow")
            self.peak_detector_window.show()
            self.peak_detector_window.activateWindow()  # Ensure the window is brought to the front
            logging.info("Peak detector window opened successfully")
        except Exception as e:
            logging.error(f"Error opening peak detector window: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open peak detector window: {str(e)}")

    def get_current_file_list(self):
        file_list = []
        current_index = self.left_file_list.rootIndex()
        for i in range(self.file_system_model.rowCount(current_index)):
            index = self.file_system_model.index(i, 0, current_index)
            file_path = self.file_system_model.filePath(index)
            if file_path.endswith('.tdms'):
                file_name = os.path.basename(file_path)
                folder_name = os.path.basename(os.path.dirname(file_path))
                experiment_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                file_list.append((experiment_name, folder_name, file_name, file_path))
        return file_list

###############################H5 Reader###############################################################################

    def open_h5_reader_window(self):
        try:
            logging.info("Attempting to open H5 Reader window")
            if not hasattr(self, 'h5_reader_window') or not self.h5_reader_window.isVisible():
                logging.info("Creating new H5ReaderWindow")
                self.h5_reader_window = H5ReaderWindow()
                logging.info("H5ReaderWindow initialized successfully")

            logging.debug("Showing H5ReaderWindow")
            self.h5_reader_window.show()
            self.h5_reader_window.activateWindow()  # Ensure the window is brought to the front
            logging.info("H5 Reader window opened successfully")
        except Exception as e:
            logging.error(f"Error opening H5 Reader window: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open H5 Reader window: {str(e)}")


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.critical(f"Unhandled exception in main: {str(e)}", exc_info=True)
        QMessageBox.critical(None, "Critical Error", f"An unhandled error occurred: {str(e)}")
        sys.exit(1)
