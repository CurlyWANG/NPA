import os
import h5py
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgba
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTreeView, QFileDialog, QMessageBox, QLabel,
                             QCheckBox, QTreeWidget, QTreeWidgetItem, QLineEdit, QGroupBox, QStatusBar, QComboBox,
                             QDialog, QTabWidget)
from PyQt6.QtGui import QDoubleValidator, QIntValidator, QBrush, QColor
from PyQt6.QtCore import Qt, QDir, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from scipy.signal import find_peaks, medfilt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pywt
from scipy.optimize import OptimizeWarning
import traceback
import logging
from scipy import stats
import warnings

class PeakDetectorWindow(QMainWindow):
    def __init__(self, parent, file_system_model, initial_path):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Peak Detector")
        self.setGeometry(150, 150, 1400, 800)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowSystemMenuHint | Qt.WindowType.WindowMinMaxButtonsHint)

        self.file_system_model = file_system_model
        self.initial_path = initial_path
        self.h5_file = None

        # Initialize QLineEdit objects with validators and default values
        self.slider1_size = 5
        self.slider2_size = 1
        self.slider1_size_input = QLineEdit(str(self.slider1_size))
        self.slider2_size_input = QLineEdit(str(self.slider2_size))
        validator = QDoubleValidator(0.0, 1000.0, 3)
        self.slider1_size_input.setValidator(validator)
        self.slider2_size_input.setValidator(validator)

        # Initialize spike detection parameters
        self.spike_threshold = 0.5
        self.spike_distance = 10
        self.spike_amplitude = 0.1
        self.show_spike_info_on_plot = False
        self.is_inverted = False

        # Add storage for accepted spikes
        self.accepted_spikes = []

        # Initialize baseline methods
        self.baseline_methods = {
            "Gaussian Fit": self.calculate_baseline,
            "Sliding Window": self.calculate_sliding_window_baseline,
            "Median Filter": self.median_filter_baseline,
            "Asymmetric Least Squares": self.als_baseline,
            "Wavelet Transform": self.wavelet_baseline
        }
        self.current_baseline_method = "Gaussian Fit"

        self.baseline_params = {
            "gaussian": {"bins": 50},
            "sliding_window": {"window_size": 1000, "step_size": 100},
            "median_filter": {"kernel_size": 501},
            "als": {"lam": 1e5, "p": 0.001},
            "wavelet": {"wavelet": 'sym8', "level": 5}
        }

        self.create_widgets()

        self.history = [self.initial_path]
        self.current_index = 0

        self.time = None
        self.voltage = None
        self.current = None

        self.span_selector1 = None
        self.span_selector2 = None

        self.slider1_start = 0
        self.slider2_start = 0

        # Add status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Timer for clearing status bar messages
        self.statusTimer = QTimer()
        self.statusTimer.setSingleShot(True)
        self.statusTimer.timeout.connect(self.clearStatusMessage)

        # 更新颜色方案
        self.colors = {
            'background': '#ffffff',
            'current': '#1e3a8a',
            'fit': '#ff6b6b',
            'baseline': '#2ecc71',
            'peak': '#e74c3c',
            'span': '#f39c12',
            'accepted': '#9b59b6',
            'slider1': '#4ecdc4',
            'slider2': '#ff9ff3'
        }

        # 设置 Seaborn 的样式
        sns.set_theme(style="whitegrid", font="Arial")
        sns.set_palette([self.colors['current'], self.colors['fit'], self.colors['baseline'],
                         self.colors['peak'], self.colors['span'], self.colors['accepted']])

        # 更新字体大小
        self.font_sizes = {
            'title': 16,
            'label': 14,
            'tick': 12,
            'legend': 12,
            'annotation': 10
        }

    def create_widgets(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 左侧：文件列表和导航
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        # 添加文件夹按钮
        self.add_folder_button = QPushButton("Add Folder")
        self.add_folder_button.clicked.connect(self.add_folder)
        left_layout.addWidget(self.add_folder_button)

        # 导航按钮
        nav_layout = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.forward_button = QPushButton("Forward")
        self.back_button.clicked.connect(self.go_back)
        self.forward_button.clicked.connect(self.go_forward)
        nav_layout.addWidget(self.back_button)
        nav_layout.addWidget(self.forward_button)
        left_layout.addLayout(nav_layout)

        # 文件系统模型和视图
        self.file_system_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self.file_system_model.setNameFilters(["*.tdms", "*.h5"])
        self.file_system_model.setNameFilterDisables(False)

        self.file_tree_view = QTreeView()
        self.file_tree_view.setModel(self.file_system_model)
        self.file_tree_view.setRootIndex(self.file_system_model.index(self.initial_path))
        self.file_tree_view.setColumnWidth(0, 200)
        self.file_tree_view.doubleClicked.connect(self.on_item_double_clicked)
        left_layout.addWidget(self.file_tree_view)

        # 添加滑块控制区域到左侧布局
        slider_control_group = QGroupBox("Slider Controls")
        slider_control_layout = QVBoxLayout()
        slider_control_group.setLayout(slider_control_layout)

        # 滑块1控制
        slider1_layout = QHBoxLayout()
        self.slider1_prev_button = QPushButton("◀")
        self.slider1_next_button = QPushButton("▶")
        slider1_layout.addWidget(QLabel("Slider 1 Size:"))
        slider1_layout.addWidget(self.slider1_size_input)
        slider1_layout.addWidget(self.slider1_prev_button)
        slider1_layout.addWidget(self.slider1_next_button)
        slider_control_layout.addLayout(slider1_layout)

        # 滑块2控制
        slider2_layout = QHBoxLayout()
        self.slider2_prev_button = QPushButton("◀")
        self.slider2_next_button = QPushButton("▶")
        slider2_layout.addWidget(QLabel("Slider 2 Size:"))
        slider2_layout.addWidget(self.slider2_size_input)
        slider2_layout.addWidget(self.slider2_prev_button)
        slider2_layout.addWidget(self.slider2_next_button)
        slider_control_layout.addLayout(slider2_layout)

        # 应用按钮
        self.apply_slider_size_button = QPushButton("Apply Slider Sizes")
        slider_control_layout.addWidget(self.apply_slider_size_button)

        left_layout.addWidget(slider_control_group)

        # 连接按钮信号
        self.apply_slider_size_button.clicked.connect(self.apply_slider_sizes)
        self.slider1_prev_button.clicked.connect(lambda: self.move_slider(1, -1))
        self.slider1_next_button.clicked.connect(lambda: self.move_slider(1, 1))
        self.slider2_prev_button.clicked.connect(lambda: self.move_slider(2, -1))
        self.slider2_next_button.clicked.connect(lambda: self.move_slider(2, 1))

        # Add Spike Detection Control
        spike_control_group = QGroupBox("Spike Detection Control")
        spike_control_layout = QVBoxLayout()
        spike_control_group.setLayout(spike_control_layout)

        # Threshold control
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_input = QLineEdit(str(self.spike_threshold))
        self.threshold_input.setValidator(QDoubleValidator(0.0, 100.0, 2))
        threshold_layout.addWidget(self.threshold_input)
        spike_control_layout.addLayout(threshold_layout)

        # Distance control
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Min Distance:"))
        self.distance_input = QLineEdit(str(self.spike_distance))
        self.distance_input.setValidator(QIntValidator(1, 1000))
        distance_layout.addWidget(self.distance_input)
        spike_control_layout.addLayout(distance_layout)

        # Amplitude control
        amplitude_layout = QHBoxLayout()
        amplitude_layout.addWidget(QLabel("Min Amplitude:"))
        self.amplitude_input = QLineEdit(str(self.spike_amplitude))
        self.amplitude_input.setValidator(QDoubleValidator(0.0, 100.0, 2))
        amplitude_layout.addWidget(self.amplitude_input)
        spike_control_layout.addLayout(amplitude_layout)

        # Add invert button
        self.invert_button = QPushButton("Invert")
        self.invert_button.clicked.connect(self.toggle_invert)
        spike_control_layout.addWidget(self.invert_button)

        # Create a horizontal layout for the button and checkbox
        button_checkbox_layout = QHBoxLayout()

        # Apply button
        self.apply_spike_params_button = QPushButton("Apply Spike Parameters")
        self.apply_spike_params_button.clicked.connect(self.apply_spike_params)
        spike_control_layout.addWidget(self.apply_spike_params_button)

        # Add checkbox for displaying spike info on plot
        self.show_spike_info_checkbox = QCheckBox("Show Spike Info on Plot")
        self.show_spike_info_checkbox.setChecked(False)
        self.show_spike_info_checkbox.stateChanged.connect(self.toggle_spike_info_display)
        button_checkbox_layout.addWidget(self.show_spike_info_checkbox)

        spike_control_layout.addLayout(button_checkbox_layout)

        left_layout.addWidget(spike_control_group)

        # 添加基线算法选择下拉菜单
        baseline_selection_layout = QHBoxLayout()
        baseline_selection_layout.addWidget(QLabel("Baseline Method:"))
        self.baseline_method_combo = QComboBox()
        self.baseline_method_combo.addItems(list(self.baseline_methods.keys()))
        self.baseline_method_combo.currentTextChanged.connect(self.on_baseline_method_changed)
        baseline_selection_layout.addWidget(self.baseline_method_combo)

        # 将下拉菜单添加到左侧布局中，位置可以根据需要调整
        left_layout.addLayout(baseline_selection_layout)

        # 添加基线参数按钮
        self.params_button = QPushButton("Baseline Parameters")
        self.params_button.clicked.connect(self.open_params_dialog)
        left_layout.addWidget(self.params_button)

        # Spike Information
        spike_info_group = QGroupBox("Spike Information")
        spike_info_layout = QVBoxLayout()
        spike_info_group.setLayout(spike_info_layout)

        self.spike_info_tree = QTreeWidget()
        self.spike_info_tree.setHeaderLabels(["Spike", "Time", "Height", "Duration"])
        self.spike_info_tree.itemDoubleClicked.connect(self.on_spike_item_double_clicked)
        spike_info_layout.addWidget(self.spike_info_tree)

        # Buttons
        button_layout = QHBoxLayout()
        self.accept_button = QPushButton("Accept")
        self.undo_button = QPushButton("Undo")
        self.accept_all_button = QPushButton("Accept All")
        self.save_button = QPushButton("Save")

        self.accept_button.clicked.connect(self.on_accept_clicked)
        self.undo_button.clicked.connect(self.on_undo_clicked)
        self.accept_all_button.clicked.connect(self.on_accept_all_clicked)
        self.save_button.clicked.connect(self.on_save_clicked)

        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.undo_button)
        button_layout.addWidget(self.accept_all_button)
        button_layout.addWidget(self.save_button)

        spike_info_layout.addLayout(button_layout)

        left_layout.addWidget(spike_info_group)

        # 中间：Figure 1 和 Figure 2
        middle_widget = QWidget()
        middle_layout = QVBoxLayout()
        middle_widget.setLayout(middle_layout)

        # Figure 1 (Voltage and Current)
        self.figure1 = Figure(figsize=(8, 4))
        self.canvas1 = FigureCanvas(self.figure1)
        middle_layout.addWidget(self.canvas1, 1)

        gs1 = self.figure1.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0)
        self.plot1 = self.figure1.add_subplot(gs1[0])  # Voltage
        self.plot2 = self.figure1.add_subplot(gs1[1], sharex=self.plot1)  # Current

        # Figure 2 (Selected regions)
        self.figure2 = Figure(figsize=(8, 8))
        self.canvas2 = FigureCanvas(self.figure2)
        middle_layout.addWidget(self.canvas2, 2)

        gs2 = self.figure2.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0)
        self.plot3 = self.figure2.add_subplot(gs2[0])  # Selected region from plot2
        self.plot4 = self.figure2.add_subplot(gs2[1])  # Selected region from plot3

        # Add NavigationToolbar for figure2
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        middle_layout.addWidget(self.toolbar2)

        # Right side: Figure 3 (Summary plots)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        self.figure3 = Figure(figsize=(6, 10))
        self.canvas3 = FigureCanvas(self.figure3)
        right_layout.addWidget(self.canvas3)

        gs3 = self.figure3.add_gridspec(2, 1, height_ratios=[1, 1])
        self.plot5 = self.figure3.add_subplot(gs3[0])  # Selected spike plot
        self.plot6 = self.figure3.add_subplot(gs3[1])  # Scatter plot


        # 将所有部件添加到主布局
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(middle_widget, 4)
        main_layout.addWidget(right_widget, 2)

#################################文件列表#######################################################

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.navigate_to(folder)

    def go_back(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_view(self.history[self.current_index])

    def go_forward(self):
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.update_view(self.history[self.current_index])

    def update_view(self, path):
        self.file_tree_view.setRootIndex(self.file_system_model.index(path))
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        self.back_button.setEnabled(self.current_index > 0)
        self.forward_button.setEnabled(self.current_index < len(self.history) - 1)

    def on_item_double_clicked(self, index):
        path = self.file_system_model.filePath(index)
        if self.file_system_model.isDir(index):
            self.navigate_to(path)
        else:
            if path.lower().endswith('.h5'):
                self.open_h5_file(path)
            elif path.lower().endswith('.tdms'):
                self.read_tdms_file(path)
            else:
                print(f"Unsupported file type: {path}")

    def navigate_to(self, path):
        if path != self.history[self.current_index]:
            self.history = self.history[:self.current_index + 1]
            self.history.append(path)
            self.current_index = len(self.history) - 1
        self.update_view(path)

####################################文件读取##################################################

    def open_h5_file(self, file_path):
        try:
            self.h5_file = h5py.File(file_path, 'r')

            # 检查必要的数据集是否存在
            required_datasets = ['time', 'voltage', 'current']
            for dataset in required_datasets:
                if dataset not in self.h5_file:
                    raise KeyError(f"Required dataset '{dataset}' not found in the H5 file.")

            # 检查数据集是否为空
            if any(len(self.h5_file[dataset]) == 0 for dataset in required_datasets):
                raise ValueError("One or more datasets are empty.")

            # 加载数据
            self.time = self.h5_file['time'][()]
            self.voltage = self.h5_file['voltage'][()]
            self.current = self.h5_file['current'][()]

            # 显示文件信息
            self.show_file_info(file_path)

            # 绘制数据
            self.plot_data()

            # 应用滑块大小
            self.apply_slider_sizes()

            self.setStatusMessage(f"File opened: {os.path.basename(file_path)}")
        except Exception as e:
            error_msg = f"Error opening H5 file {file_path}: {str(e)}"
            print(error_msg)
            self.setStatusMessage(f"Error: {error_msg}", error=True)
            # 重置数据
            self.time = None
            self.voltage = None
            self.current = None

    def load_file_structure(self):
        self.datasets = []
        self.h5_tree.clear()

        def add_to_tree(name, item):
            parts = name.split('/')
            parent = self.h5_tree.invisibleRootItem()
            for i, part in enumerate(parts):
                path = '/'.join(parts[:i+1])
                if i < len(parts) - 1:
                    # 这是一个组
                    child = self.find_or_create_item(parent, part)
                else:
                    # 这是一个数据集
                    child = QTreeWidgetItem(parent)
                    child.setText(0, part)
                    if isinstance(item, h5py.Dataset):
                        shape_str = str(item.shape) if item.shape else "Scalar"
                        child.setText(1, "Dataset")
                        child.setText(2, shape_str)
                        self.datasets.append(name)
                parent = child

        self.h5_file.visititems(add_to_tree)
        self.update_dataset_combos()

    def find_or_create_item(self, parent, text):
        for i in range(parent.childCount()):
            if parent.child(i).text(0) == text:
                return parent.child(i)
        item = QTreeWidgetItem(parent)
        item.setText(0, text)
        return item

    def update_dataset_combos(self):
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        for dataset in self.datasets:
            self.x_axis_combo.addItem(dataset)
            self.y_axis_combo.addItem(dataset)

    def show_file_info(self, file_path):
        info = f"File: {os.path.basename(file_path)}\n"
        info += f"Path: {file_path}\n\n"
        info += "Available Datasets:\n"
        for key in ['time', 'voltage', 'current']:
            if key in self.h5_file:
                item = self.h5_file[key]
                info += f"{key}: Shape {item.shape}, Type {item.dtype}\n"
        self.setStatusMessage(info)

    def read_tdms_file(self, file_path):
        # Implement your TDMS file reading logic here
        self.data_presentation_label.setText(f"TDMS file selected: {file_path}")
        # You may want to implement actual TDMS reading logic similar to your MainWindow

    def setStatusMessage(self, message, duration=5000, error=False):
        if error:
            self.statusBar.setStyleSheet("QStatusBar{color:red;}")
        else:
            self.statusBar.setStyleSheet("")
        self.statusBar.showMessage(message)
        self.statusTimer.start(duration)  # Clear the message after 5 seconds

    def clearStatusMessage(self):
        self.statusBar.clearMessage()
        self.statusBar.setStyleSheet("")

##########################################绘图########################################################

    def init_plots(self):
        # 用空数据初始化图表
        self.plot1.set_title('Voltage')
        self.plot2.set_title('Current')
        self.plot3.set_title('Current')
        self.plot4.set_title('Peak Analysis')
        self.plot5.set_title('All Saved Peaks')
        self.plot6.set_title('Peak Duration')
        self.plot7.set_title('Peak Amplitude')
        self.plot8.set_title('Duration vs Amplitude')

        self.canvas1.draw()
        self.canvas2.draw()

    def update_plots_with_data(self):
        self.plot1.clear()
        self.plot2.clear()

        if 'time' in self.data and 'voltage' in self.data:
            self.plot1.plot(self.data['time'], self.data['voltage'],
                            color=self.display_settings.get('voltage_color', 'blue'),
                            label='Voltage')
            self.plot1.set_xlabel('Time')
            self.plot1.set_ylabel('Voltage')
            self.plot1.legend()

        if 'time' in self.data and 'current' in self.data:
            self.plot2.plot(self.data['time'], self.data['current'],
                            color=self.display_settings.get('current_color', 'red'),
                            label='Current')
            self.plot2.set_xlabel('Time')
            self.plot2.set_ylabel('Current')
            self.plot2.legend()

        # 应用显示设置
        font_size = self.display_settings.get('font_size', 10)
        font_weight = 'bold' if self.display_settings.get('font_bold', False) else 'normal'
        for plot in [self.plot1, self.plot2]:
            plot.tick_params(labelsize=font_size)
            plot.xaxis.label.set_fontsize(font_size)
            plot.xaxis.label.set_fontweight(font_weight)
            plot.yaxis.label.set_fontsize(font_size)
            plot.yaxis.label.set_fontweight(font_weight)

        if 'y_start' in self.display_settings and 'y_end' in self.display_settings:
            self.plot1.set_ylim(self.display_settings['y_start'], self.display_settings['y_end'])
            self.plot2.set_ylim(self.display_settings['y_start'], self.display_settings['y_end'])

        self.canvas1.draw()

    def plot_data(self):
        try:
            if self.time is None or self.voltage is None or self.current is None:
                raise ValueError("Data not loaded properly.")

            # Clear all plots
            for plot in [self.plot1, self.plot2, self.plot3, self.plot4]:
                plot.clear()

            # Plot voltage data
            self.plot1.plot(self.time, self.voltage)
            self.plot1.set_ylabel('Voltage')
            self.plot1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            # Plot current data
            self.plot2.plot(self.time, self.current)
            self.plot2.set_xlabel('Time')
            self.plot2.set_ylabel('Current')

            # Set x-axis limits to match data range exactly
            self.plot1.set_xlim(self.time[0], self.time[-1])
            self.plot2.set_xlim(self.time[0], self.time[-1])

            # Adjust subplot params instead of using tight_layout
            self.figure1.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0)
            self.figure2.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0)

            # Create SpanSelectors with new default values
            self.create_span_selectors()

            # Update all canvases
            self.canvas1.draw()
            self.canvas2.draw()

        except Exception as e:
            error_msg = f"Error loading or plotting data: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

        except Exception as e:
            error_msg = f"Error loading or plotting data: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def update_plot(self, plot, start_index, end_index, y_label):
        plot.clear()
        time_slice = self.time[start_index:end_index]
        current_slice = self.get_current_slice(start_index, end_index)
        sns.lineplot(x=time_slice, y=current_slice, ax=plot, color=self.colors['current'], linewidth=2)
        plot.set_xlabel('Time', fontweight='bold', fontsize=self.font_sizes['label'])
        plot.set_ylabel(y_label, fontweight='bold', fontsize=self.font_sizes['label'])
        plot.set_xlim(time_slice[0], time_slice[-1])
        plot.tick_params(axis='both', which='major', labelsize=self.font_sizes['tick'])
        sns.despine(ax=plot)

#######################################滑块显示###################################################

    def create_span_selectors(self):
        if self.time is None or len(self.time) == 0:
            print("Cannot create span selectors: No data available")
            return

        # Remove existing span selectors if they exist
        if self.span_selector1:
            self.span_selector1.disconnect_events()
        if self.span_selector2:
            self.span_selector2.disconnect_events()

        # Create new span selectors with updated colors
        self.span_selector1 = SpanSelector(
            self.plot2, self.onselect1, 'horizontal', useblit=True,
            props=dict(alpha=0.5, facecolor=self.colors['slider1']),
            interactive=True
        )

        self.span_selector2 = SpanSelector(
            self.plot3, self.onselect2, 'horizontal', useblit=True,
            props=dict(alpha=0.5, facecolor=self.colors['slider2']),
            interactive=True
        )

        # Set initial positions
        self.update_span_selector(self.span_selector1, self.time[0], self.slider1_size)
        self.update_span_selector(self.span_selector2, self.time[0], self.slider2_size)

    def update_span_selector(self, span_selector, start, size):
        if span_selector and self.time is not None:
            start_index = np.searchsorted(self.time, start)
            end_index = np.searchsorted(self.time, start + size)
            end_index = min(end_index, len(self.time) - 1)
            span_selector.extents = (self.time[start_index], self.time[end_index])
            span_selector.set_visible(True)
            span_selector.ax.figure.canvas.draw_idle()

            # Update the corresponding plot
            if span_selector == self.span_selector1:
                self.update_plot(self.plot3, start_index, end_index, 'Current')
                # Reset span_selector2 to the start of the new data range
                self.slider2_start = start_index
                self.update_span_selector(self.span_selector2, self.time[start_index], self.slider2_size)
            elif span_selector == self.span_selector2:
                self.update_plot(self.plot4, start_index, end_index, 'Peak Analysis')

    def onselect1(self, xmin, xmax):
        indmin, indmax = np.searchsorted(self.time, (xmin, xmax))
        indmax = min(len(self.time) - 1, indmax)
        self.update_plot(self.plot3, indmin, indmax, 'Current (Selected)')
        self.slider2_start = indmin
        self.update_span_selector(self.span_selector2, self.time[indmin], self.slider2_size)
        self.plot_spikes()
        self.canvas2.draw()

    def onselect2(self, xmin, xmax):
        indmin, indmax = np.searchsorted(self.time, (xmin, xmax))
        indmax = min(len(self.time) - 1, indmax)
        self.update_plot(self.plot4, indmin, indmax, 'Current (Analysis)')
        self.plot_spikes()
        self.canvas2.draw()

    def apply_slider_sizes(self):
        # 获取当前滑块的起始位置
        current_start1 = self.span_selector1.extents[0] if self.span_selector1 else self.time[0]
        current_start2 = self.span_selector2.extents[0] if self.span_selector2 else self.time[0]

        # 更新大小
        new_size1 = self.get_float_safely(self.slider1_size_input)
        new_size2 = self.get_float_safely(self.slider2_size_input)

        if new_size1 is not None:
            self.slider1_size = new_size1
        if new_size2 is not None:
            self.slider2_size = new_size2

        # 更新span selectors，保持当前位置
        self.update_span_selector(self.span_selector1, current_start1, self.slider1_size)
        self.update_span_selector(self.span_selector2, current_start2, self.slider2_size)

        # 重绘画布
        self.plot_spikes()
        self.canvas1.draw()
        self.canvas2.draw()

    def move_slider(self, slider_num, direction):
        if self.time is None:
            return

        if slider_num == 1:
            slider_size = self.slider1_size
            span_selector = self.span_selector1
        else:
            slider_size = self.slider2_size
            span_selector = self.span_selector2

        current_start = span_selector.extents[0]
        new_start = max(self.time[0], min(current_start + direction * slider_size, self.time[-1] - slider_size))

        self.update_span_selector(span_selector, new_start, slider_size)
        self.canvas1.draw()
        self.canvas2.draw()

    def get_float_safely(self, qline_edit):
        """Safely convert QLineEdit text to float."""
        text = qline_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
            if value <= 0:
                QMessageBox.warning(self, "Invalid Input", "Value must be positive.")
                return None
            return value
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")
            return None

#########################Peak detection#########################################################

    def apply_spike_params(self):
        try:
            logging.debug("Applying spike parameters")
            self.spike_threshold = float(self.threshold_input.text())
            self.spike_distance = int(self.distance_input.text())
            self.spike_amplitude = float(self.amplitude_input.text())
            logging.debug(
                f"Parameters: threshold={self.spike_threshold}, distance={self.spike_distance}, amplitude={self.spike_amplitude}")
            self.plot_spikes(update_plot3=False)
        except Exception as e:
            logging.error(f"Error in apply_spike_params: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def toggle_spike_info_display(self, state):
        self.show_spike_info_on_plot = state == Qt.CheckState.Checked.value
        self.plot_spikes()  # Redraw the plot with updated settings

    def toggle_invert(self):
        self.is_inverted = not self.is_inverted
        self.update_all_plots()
        self.invert_button.setText("Revert" if self.is_inverted else "Invert")
        self.setStatusMessage("Data inverted" if self.is_inverted else "Data reverted")

    def get_current_slice(self, start_index, end_index):
        current_slice = self.current[start_index:end_index]
        return -current_slice if self.is_inverted else current_slice

    def invert_data(self, data):
        return -data  # Invert the data around y=0

    def detect_spikes(self, data, baseline):
        try:
            logging.debug("Detecting spikes")

            # 归一化数据
            normalized_data = (data - baseline) / np.std(data)

            # 检测正峰和负峰
            pos_peaks, _ = find_peaks(normalized_data, height=self.spike_threshold, distance=self.spike_distance)
            neg_peaks, _ = find_peaks(-normalized_data, height=self.spike_threshold, distance=self.spike_distance)

            # 合并正峰和负峰
            all_peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))

            # 根据振幅过滤峰值
            filtered_peaks = []
            for peak in all_peaks:
                amplitude = abs(normalized_data[peak] * np.std(data))
                if amplitude > self.spike_amplitude:
                    filtered_peaks.append(peak)

            logging.debug(f"Detected {len(filtered_peaks)} spikes")
            return filtered_peaks
        except Exception as e:
            logging.error(f"Error in detect_spikes: {str(e)}")
            logging.error(traceback.format_exc())
            return []

    def on_baseline_method_changed(self, method):
        self.current_baseline_method = method
        self.update_all_plots()  # 使用新方法更新所有图表

    def open_params_dialog(self):
        dialog = BaselineParamsDialog(self)
        dialog.set_params(self.baseline_params)  # 设置当前参数
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.baseline_params = dialog.get_params()
            self.update_all_plots()  # 使用新参数更新所有图表

    def calculate_baseline(self, data):
        bins = self.baseline_params["gaussian"]["bins"]
        try:
            # 创建直方图
            hist, bin_edges = np.histogram(data, bins='auto')
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # 找到直方图的峰值
            max_index = np.argmax(hist)

            # 高斯函数用于拟合
            def gaussian(x, amp, mu, sigma):
                return amp * stats.norm.pdf(x, mu, sigma)

            # 使用峰值周围的数据子集进行拟合
            window = min(10, len(hist) // 4)  # 根据数据调整窗口大小
            left = max(0, max_index - window)
            right = min(len(hist), max_index + window)

            # 参数的初始猜测
            p0 = [hist[max_index], bin_centers[max_index], np.std(data)]

            # 抑制 OptimizeWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)

                # 曲线拟合
                popt, pcov = stats.curve_fit(gaussian, bin_centers[left:right], hist[left:right],
                                             p0=p0,
                                             bounds=([0, np.min(data), 0],
                                                     [np.inf, np.max(data), np.inf]),
                                             maxfev=10000)  # 增加最大迭代次数

            # 检查拟合质量
            if np.any(np.diag(pcov) < 0):
                raise ValueError("Fit resulted in negative variance")

            baseline_value = popt[1]

            # 添加一些诊断信息
            logging.info(f"Gaussian fit parameters: amplitude={popt[0]:.2f}, mean={popt[1]:.2f}, sigma={popt[2]:.2f}")
            logging.info(f"Estimated baseline value: {baseline_value:.2f}")

            return np.full_like(data, baseline_value)

        except Exception as e:
            logging.warning(f"Gaussian fit failed: {str(e)}. Using median as fallback.")
            # 如果高斯拟合失败，使用移动中位数作为备选方案
            window_size = min(1001, len(data) // 10)  # 使用数据长度的10%或1001，取较小值
            if window_size % 2 == 0:
                window_size += 1  # 确保窗口大小为奇数
            baseline = np.convolve(data, np.ones(window_size) / window_size, mode='same')
            return baseline

    def calculate_sliding_window_baseline(self, data, window_size=None, step_size=None):
        window_size = self.baseline_params["sliding_window"]["window_size"]
        step_size = self.baseline_params["sliding_window"]["step_size"]
        try:
            data_length = len(data)
            logging.debug(f"Input data length: {data_length}")

            # 动态调整窗口大小和步长
            if window_size is None:
                window_size = min(1000, max(100, data_length // 10))
            if step_size is None:
                step_size = max(1, window_size // 10)

            logging.debug(f"Using window_size: {window_size}, step_size: {step_size}")

            baseline = np.zeros_like(data)
            for i in range(0, data_length, step_size):
                start = i
                end = min(i + window_size, data_length)
                window_data = data[start:end]
                window_baseline = self.calculate_baseline(window_data)

                # 确保window_baseline是一个标量
                if isinstance(window_baseline, np.ndarray):
                    window_baseline = np.mean(window_baseline)

                fill_end = min(i + step_size, data_length)
                baseline[start:fill_end] = window_baseline

            # 处理开始和结束的边缘情况
            baseline[:window_size // 2] = baseline[window_size // 2]
            baseline[-(window_size // 2):] = baseline[-(window_size // 2) - 1]

            logging.debug(f"Output baseline length: {len(baseline)}")

            if len(baseline) != data_length:
                raise ValueError(f"Baseline length ({len(baseline)}) does not match data length ({data_length})")

            return baseline

        except Exception as e:
            logging.error(f"Error in calculate_sliding_window_baseline: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # 如果出错，返回一个与输入数据长度相同的数组，填充数据的中位数
            return np.full_like(data, np.median(data))

    def median_filter_baseline(self, data, kernel_size=501):
        kernel_size = self.baseline_params["median_filter"]["kernel_size"]
        return medfilt(data, kernel_size=kernel_size)

    def als_baseline(self, data, lam=10 ** 5, p=0.001, niter=10):
        lam = self.baseline_params["als"]["lam"]
        p = self.baseline_params["als"]["p"]
        L = len(data)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * data)
            w = p * (data > z) + (1 - p) * (data < z)
        return z

    def wavelet_baseline(self, data, wavelet='sym8', level=5):
        wavelet = self.baseline_params["wavelet"]["wavelet"]
        level = self.baseline_params["wavelet"]["level"]
        coeff = pywt.wavedec(data, wavelet, mode='per', level=level)
        coeff[1:] = [np.zeros_like(c) for c in coeff[1:]]
        baseline = pywt.waverec(coeff, wavelet, mode='per')
        return baseline

    def simplified_fit_spikes(self, data, time, filtered_peaks, baseline):
        try:
            logging.debug("Fitting spikes")
            y_fit = np.full_like(data, fill_value=np.nan)
            spike_info = []

            for peak in filtered_peaks:
                # 确定这是正峰还是负峰
                is_positive_spike = data[peak] > baseline[peak] if isinstance(baseline, np.ndarray) else baseline

                # 找到峰值的起始（向后）
                start = peak
                while start > 0 and ((data[start] > baseline[start]) if isinstance(baseline, np.ndarray) else (
                        data[start] > baseline)) == is_positive_spike:
                    start -= 1

                # 找到峰值的结束（向前）
                end = peak
                while end < len(data) - 1 and ((data[end] > baseline[end]) if isinstance(baseline, np.ndarray) else (
                        data[end] > baseline)) == is_positive_spike:
                    end += 1

                # 计算峰值属性
                height = abs(data[peak] - (baseline[peak] if isinstance(baseline, np.ndarray) else baseline))
                duration = time[end] - time[start]

                # 只包括持续时间 > 0.0002 的峰值
                if duration > 0.0002:
                    # 将拟合值设置为实际数据值
                    y_fit[start:end + 1] = data[start:end + 1]

                    spike_info.append({
                        'peak': peak,
                        'start': start,
                        'end': end,
                        'height': height,
                        'duration': duration,
                        'is_positive': is_positive_spike
                    })

            # 用基线值填充非峰值区域
            mask = np.isnan(y_fit)
            y_fit[mask] = baseline[mask] if isinstance(baseline, np.ndarray) else baseline

            logging.debug(f"Fitted {len(spike_info)} spikes")
            return y_fit, spike_info
        except Exception as e:
            logging.error(f"Error in simplified_fit_spikes: {str(e)}")
            logging.error(traceback.format_exc())
            return np.zeros_like(data), []


    def plot_spikes(self, update_plot3=True):
        try:
            logging.debug("Plotting spikes")
            if self.current is None or self.time is None or len(self.current) == 0 or len(self.time) == 0:
                logging.warning("No data available for plotting")
                QMessageBox.warning(self, "Warning", "No data available for plotting")
                return

            if not hasattr(self, 'span_selector2') or self.span_selector2 is None:
                logging.warning("Span selector is not initialized")
                QMessageBox.warning(self, "Warning", "Span selector is not initialized")
                return

            start, end = self.span_selector2.extents
            if start >= end or start < self.time[0] or end > self.time[-1]:
                logging.warning("Invalid span selector range")
                QMessageBox.warning(self, "Warning", "Invalid span selector range")
                return

            start_index = max(0, np.searchsorted(self.time, start) - 1)
            end_index = min(len(self.time) - 1, np.searchsorted(self.time, end) + 1)

            if start_index >= end_index:
                logging.warning("Invalid data range")
                QMessageBox.warning(self, "Warning", "Invalid data range")
                return

            time_slice = self.time[start_index:end_index]
            current_slice = self.get_current_slice(start_index, end_index)

            if len(time_slice) == 0 or len(current_slice) == 0:
                logging.warning("No data in selected range")
                QMessageBox.warning(self, "Warning", "No data in selected range")
                return

            # 使用选定的基线方法
            baseline_method = self.baseline_methods[self.current_baseline_method]
            baseline = baseline_method(current_slice)

            # 检测峰值
            filtered_peaks = self.detect_spikes(current_slice, baseline)
            y_fit, spike_info = self.simplified_fit_spikes(current_slice, time_slice, filtered_peaks, baseline)

            # 更新spike_info与绝对索引
            self.spike_info = [{**spike, 'start': spike['start'] + start_index, 'end': spike['end'] + start_index,
                                'peak': spike['peak'] + start_index} for spike in spike_info]

            if update_plot3:
                self.plot3.clear()
                sns.lineplot(x=time_slice, y=current_slice, ax=self.plot3, color=self.colors['current'], linewidth=2,
                             label='Current')

            self.plot4.clear()

            # 绘制电流
            sns.lineplot(x=time_slice, y=current_slice, ax=self.plot4, color=self.colors['current'], linewidth=2,
                         label='Current')

            # 绘制拟合和基线
            sns.lineplot(x=time_slice, y=y_fit, ax=self.plot4, color=self.colors['fit'], linewidth=1.5,
                         label='Simplified Fit')
            self.plot4.plot(time_slice, baseline, color=self.colors['baseline'], linestyle='--', label='Baseline',
                            linewidth=1.5)

            self.spike_info_tree.clear()
            for i, spike in enumerate(self.spike_info):
                peak_value = current_slice[spike['peak'] - start_index]
                marker_color = self.colors['peak'] if spike['is_positive'] else 'g'
                self.plot4.plot(time_slice[spike['peak'] - start_index], peak_value, "o", color=marker_color,
                                markersize=6)

                if self.show_spike_info_on_plot:
                    self.plot4.annotate(
                        f'Spike {i + 1}\nHeight: {spike["height"]:.2f}\nDuration: {spike["duration"]:.4f}s',
                        xy=(time_slice[spike['peak'] - start_index], peak_value),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=self.font_sizes['annotation'],
                        color=marker_color
                    )
                self.plot4.axvline(time_slice[spike['start'] - start_index], color=marker_color, linestyle=':',
                                   alpha=0.7, linewidth=1.5)
                self.plot4.axvline(time_slice[spike['end'] - start_index], color=marker_color, linestyle=':', alpha=0.7,
                                   linewidth=1.5)

                item = QTreeWidgetItem(self.spike_info_tree)
                item.setText(0, f"Spike {i + 1}")
                item.setText(1, f"{self.time[spike['peak']]:.4f}")
                item.setText(2, f"{spike['height']:.2f}")
                item.setText(3, f"{spike['duration']:.4f}")
                item.setText(4, "Positive" if spike['is_positive'] else "Negative")

                if spike['is_positive']:
                    item.setBackground(4, QBrush(QColor(200, 255, 200)))  # Light green
                else:
                    item.setBackground(4, QBrush(QColor(255, 200, 200)))  # Light red

            # 重新标记已接受的峰值
            for spike in self.accepted_spikes:
                self.mark_spike_in_plot4(spike)

            for plot in ([self.plot3, self.plot4] if update_plot3 else [self.plot4]):
                plot.set_xlabel('Time', fontweight='bold', fontsize=self.font_sizes['label'])
                plot.set_ylabel('Current', fontweight='bold', fontsize=self.font_sizes['label'])
                plot.tick_params(axis='both', which='major', labelsize=self.font_sizes['tick'])
                sns.despine(ax=plot)  # 移除顶部和右侧的轴线
                plot.set_xlim(time_slice[0], time_slice[-1])  # 设置x轴限制以精确匹配数据范围

            # 更新图例以包含基线方法信息
            self.plot4.legend(title=f"Baseline: {self.current_baseline_method}",
                              frameon=True, facecolor='none', edgecolor='none',
                              fontsize=self.font_sizes['legend'])

            # 更新反转按钮文本
            self.invert_button.setText("Revert" if self.is_inverted else "Invert")

            self.canvas2.draw()
            logging.debug("Finished plotting spikes")
        except Exception as e:
            logging.error(f"Error in plot_spikes: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"An error occurred while plotting: {str(e)}")

    def on_spike_item_double_clicked(self, item, column):
        if item.parent() is None:  # This is a top-level item (a spike)
            spike_index = self.spike_info_tree.indexOfTopLevelItem(item)
            if 0 <= spike_index < len(self.spike_info):
                spike = self.spike_info[spike_index]
                self.plot_selected_spike(spike, spike_index)
            else:
                logging.warning(f"Invalid spike index: {spike_index}")

    def plot_selected_spike(self, spike, spike_index):
        self.plot5.clear()
        start = spike['start']
        end = spike['end']
        peak = spike['peak']

        time_slice = self.time[start:end + 1]
        current_slice = self.get_current_slice(start, end + 1)

        # 使用当前选择的基线方法计算基线
        baseline_method = self.baseline_methods[self.current_baseline_method]
        baseline = baseline_method(current_slice)

        # 绘制电流数据
        self.plot5.plot(time_slice, current_slice, color=self.colors['current'], linewidth=2, label='Current')

        # 绘制基线
        self.plot5.plot(time_slice, baseline, color=self.colors['baseline'], linestyle='--', label='Baseline')

        # 标记峰值
        peak_color = self.colors['peak'] if spike['is_positive'] else 'g'
        self.plot5.plot(self.time[peak], current_slice[peak - start], 'o', color=peak_color, markersize=8, label='Peak')

        # 添加注释
        peak_value = current_slice[peak - start]
        baseline_value = baseline[peak - start]
        self.plot5.annotate(f'Peak: {peak_value:.2f}',
                            xy=(self.time[peak], peak_value),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=self.font_sizes['annotation'],
                            color=peak_color)

        self.plot5.annotate(f'Height: {spike["height"]:.2f}\nDuration: {spike["duration"]:.4f}s',
                            xy=(self.time[start], baseline_value),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=self.font_sizes['annotation'],
                            color=self.colors['current'])

        # 设置标签和标题
        self.plot5.set_title(f"Spike {spike_index + 1} ({'Positive' if spike['is_positive'] else 'Negative'})",
                             fontweight='bold', fontsize=self.font_sizes['title'])
        self.plot5.set_xlabel("Time", fontweight='bold', fontsize=self.font_sizes['label'])
        self.plot5.set_ylabel("Current", fontweight='bold', fontsize=self.font_sizes['label'])

        # 设置坐标轴范围
        self.plot5.set_xlim(time_slice[0], time_slice[-1])
        y_min = min(min(current_slice), min(baseline))
        y_max = max(max(current_slice), max(baseline))
        y_range = y_max - y_min
        self.plot5.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # 样式调整
        self.plot5.tick_params(axis='both', which='major', labelsize=self.font_sizes['tick'])
        self.plot5.legend(fontsize=self.font_sizes['legend'])

        # 添加当前使用的基线方法信息
        self.plot5.text(0.05, 0.95, f"Baseline: {self.current_baseline_method}",
                        transform=self.plot5.transAxes, fontsize=self.font_sizes['annotation'],
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.canvas3.draw()

    def update_all_plots(self):
        if self.span_selector1:
            start, end = self.span_selector1.extents
            start_index = max(0, np.searchsorted(self.time, start) - 1)
            end_index = min(len(self.time) - 1, np.searchsorted(self.time, end) + 1)
            self.update_plot(self.plot3, start_index, end_index, 'Current (Selected)')

        if self.span_selector2:
            start, end = self.span_selector2.extents
            start_index = max(0, np.searchsorted(self.time, start) - 1)
            end_index = min(len(self.time) - 1, np.searchsorted(self.time, end) + 1)
            self.update_plot(self.plot4, start_index, end_index, 'Current (Analysis)')

        # 只重新绘制 plot4 的尖峰
        self.plot_spikes(update_plot3=False)
        self.canvas1.draw()
        self.canvas2.draw()


    def on_accept_clicked(self):
        selected_items = self.spike_info_tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            if item.parent() is None:  # This is a top-level item (a spike)
                spike_index = self.spike_info_tree.indexOfTopLevelItem(item)
                spike = self.spike_info[spike_index]
                if spike not in self.accepted_spikes:
                    self.accepted_spikes.append(spike)
                    self.mark_spike_in_plot4(spike)
                    item.setBackground(0, QBrush(QColor(200, 255, 200)))  # Light green background
        self.update_scatter_plot()

    def on_undo_clicked(self):
        selected_items = self.spike_info_tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            if item.parent() is None:  # This is a top-level item (a spike)
                spike_index = self.spike_info_tree.indexOfTopLevelItem(item)
                spike = self.spike_info[spike_index]
                if spike in self.accepted_spikes:
                    self.accepted_spikes.remove(spike)
                    self.unmark_spike_in_plot4(spike)
                    item.setBackground(0, QBrush(QColor(255, 255, 255)))  # White background
        self.update_scatter_plot()

    def on_accept_all_clicked(self):
        for spike in self.spike_info:
            if spike not in self.accepted_spikes:
                self.accepted_spikes.append(spike)
                self.mark_spike_in_plot4(spike)

        for i in range(self.spike_info_tree.topLevelItemCount()):
            item = self.spike_info_tree.topLevelItem(i)
            item.setBackground(0, QBrush(QColor(200, 255, 200)))  # Light green background

        self.update_scatter_plot()

    def on_save_clicked(self):
        if not self.accepted_spikes:
            self.setStatusMessage("No spikes have been accepted.", error=True)
            return

        file_path = self.h5_file.filename
        directory = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]

        # Function to generate unique file name
        def get_unique_file_name(base_path, extension):
            counter = 1
            while True:
                new_path = f"{base_path}_{counter:03d}{extension}"
                if not os.path.exists(new_path):
                    return new_path
                counter += 1

        # Generate unique file names
        traces_file = get_unique_file_name(os.path.join(directory, f"{base_name}_spike_traces"), ".h5")
        properties_file = get_unique_file_name(os.path.join(directory, f"{base_name}_spike_properties"), ".h5")

        # Save spike traces
        with h5py.File(traces_file, 'w') as f:
            for i, spike in enumerate(self.accepted_spikes):
                group = f.create_group(f"spike_{i}")
                group.create_dataset("time", data=self.time[spike['start']:spike['end'] + 1])
                group.create_dataset("voltage", data=self.voltage[spike['start']:spike['end'] + 1])
                group.create_dataset("current", data=self.current[spike['start']:spike['end'] + 1])

        # Save spike properties
        with h5py.File(properties_file, 'w') as f:
            f.create_dataset("duration", data=[spike['duration'] for spike in self.accepted_spikes])
            f.create_dataset("amplitude", data=[spike['height'] for spike in self.accepted_spikes])

        self.setStatusMessage(f"Spike data saved to:\n{traces_file}\n{properties_file}")

    def mark_spike_in_plot4(self, spike):
        start_time = self.time[spike['start']]
        end_time = self.time[spike['end']]
        color = to_rgba(self.colors['accepted'])
        color_light = (*color[:3], 0.3)
        self.plot4.axvspan(start_time, end_time, color=color_light)
        self.canvas2.draw()

    def unmark_spike_in_plot4(self, spike):
        for collection in self.plot4.collections:
            if isinstance(collection, plt.matplotlib.collections.PolyCollection):
                xy = collection.get_paths()[0].vertices
                if xy[0][0] == self.time[spike['start']] and xy[2][0] == self.time[spike['end']]:
                    collection.remove()
        self.canvas2.draw()

    def update_scatter_plot(self):
        self.plot6.clear()
        durations = [spike['duration'] for spike in self.accepted_spikes]
        amplitudes = [spike['height'] for spike in self.accepted_spikes]
        sns.scatterplot(x=durations, y=amplitudes, ax=self.plot6, color=self.colors['peak'], alpha=0.7,
                        edgecolor=self.colors['current'])
        self.plot6.set_xlabel("Duration", fontweight='bold', fontsize=self.font_sizes['label'])
        self.plot6.set_ylabel("Amplitude", fontweight='bold', fontsize=self.font_sizes['label'])
        self.plot6.set_title("Spike Duration vs Amplitude", fontweight='bold', fontsize=self.font_sizes['title'])

        self.plot6.tick_params(axis='both', which='major', labelsize=self.font_sizes['tick'])
        sns.despine(ax=self.plot6)  # Remove top and right spines
        self.canvas3.draw()

class BaselineParamsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Baseline Calculation Parameters")
        self.setModal(True)
        self.resize(800, 400)

        layout = QVBoxLayout(self)

        # 创建选项卡窗口小部件
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # 高斯拟合参数
        gaussian_tab = QWidget()
        gaussian_layout = QVBoxLayout(gaussian_tab)
        self.gaussian_bins = QLineEdit(str(50))
        gaussian_layout.addWidget(QLabel("Number of bins:"))
        gaussian_layout.addWidget(self.gaussian_bins)
        tab_widget.addTab(gaussian_tab, "Gaussian Fit")

        # 滑动窗口参数
        sliding_tab = QWidget()
        sliding_layout = QVBoxLayout(sliding_tab)
        self.window_size = QLineEdit(str(1000))
        self.step_size = QLineEdit(str(100))
        sliding_layout.addWidget(QLabel("Window size:"))
        sliding_layout.addWidget(self.window_size)
        sliding_layout.addWidget(QLabel("Step size:"))
        sliding_layout.addWidget(self.step_size)
        tab_widget.addTab(sliding_tab, "Sliding Window")

        # 中值滤波参数
        median_tab = QWidget()
        median_layout = QVBoxLayout(median_tab)
        self.kernel_size = QLineEdit(str(501))
        median_layout.addWidget(QLabel("Kernel size:"))
        median_layout.addWidget(self.kernel_size)
        tab_widget.addTab(median_tab, "Median Filter")

        # ALS参数
        als_tab = QWidget()
        als_layout = QVBoxLayout(als_tab)
        self.als_lam = QLineEdit(str(1e5))
        self.als_p = QLineEdit(str(0.001))
        als_layout.addWidget(QLabel("Lambda:"))
        als_layout.addWidget(self.als_lam)
        als_layout.addWidget(QLabel("p:"))
        als_layout.addWidget(self.als_p)
        tab_widget.addTab(als_tab, "Asymmetric Least Squares")

        # 小波变换参数
        wavelet_tab = QWidget()
        wavelet_layout = QVBoxLayout(wavelet_tab)
        self.wavelet_type = QComboBox()
        self.wavelet_type.addItems(['sym8', 'db8', 'haar'])
        self.wavelet_level = QLineEdit(str(5))
        wavelet_layout.addWidget(QLabel("Wavelet type:"))
        wavelet_layout.addWidget(self.wavelet_type)
        wavelet_layout.addWidget(QLabel("Decomposition level:"))
        wavelet_layout.addWidget(self.wavelet_level)
        tab_widget.addTab(wavelet_tab, "Wavelet Transform")

        # OK和Cancel按钮
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

    def get_params(self):
        return {
            "gaussian": {"bins": int(self.gaussian_bins.text())},
            "sliding_window": {"window_size": int(self.window_size.text()),
                               "step_size": int(self.step_size.text())},
            "median_filter": {"kernel_size": int(self.kernel_size.text())},
            "als": {"lam": float(self.als_lam.text()),
                    "p": float(self.als_p.text())},
            "wavelet": {"wavelet": self.wavelet_type.currentText(),
                        "level": int(self.wavelet_level.text())}
        }

    def set_params(self, params):
        self.gaussian_bins.setText(str(params["gaussian"]["bins"]))
        self.window_size.setText(str(params["sliding_window"]["window_size"]))
        self.step_size.setText(str(params["sliding_window"]["step_size"]))
        self.kernel_size.setText(str(params["median_filter"]["kernel_size"]))
        self.als_lam.setText(str(params["als"]["lam"]))
        self.als_p.setText(str(params["als"]["p"]))
        self.wavelet_type.setCurrentText(params["wavelet"]["wavelet"])
        self.wavelet_level.setText(str(params["wavelet"]["level"]))


'''
        # Add baseline method selection to the spike control group
        baseline_layout = QHBoxLayout()
        baseline_layout.addWidget(QLabel("Baseline Method:"))
        self.baseline_method_combo = QComboBox()
        self.baseline_method_combo.addItems(list(self.baseline_methods.keys()))
        self.baseline_method_combo.currentTextChanged.connect(self.on_baseline_method_changed)

        baseline_layout.addWidget(self.baseline_method_combo)
        spike_control_layout.addLayout(baseline_layout)
'''