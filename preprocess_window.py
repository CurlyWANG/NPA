import os
import numpy as np
import h5py
import warnings
from typing import TYPE_CHECKING, List, Tuple
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QTreeWidget, QTreeWidgetItem,
                             QWidget, QMessageBox, QTabWidget, QSplitter, QLineEdit, QComboBox, QListWidgetItem,
                             QListWidget, QFormLayout, QLabel, QSpinBox, QCheckBox, QGroupBox, QGridLayout, QTextEdit,
                             QStyleFactory, QColorDialog, QFontComboBox, QDoubleSpinBox, QFileDialog, QRadioButton, QScrollArea)
from PyQt6.QtCore import pyqtSignal, Qt,  QTimer, QSize
from PyQt6.QtGui import QGuiApplication, QPalette, QColor, QFont
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.ticker as ticker
from scipy import signal, stats
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks, peak_widths
import traceback
import logging

if TYPE_CHECKING:
    from main_window import MainWindow


class PSDAnalysis:
    def __init__(self):
        """Initialize PSD analysis parameters"""
        self.default_params = {
            'window': 'hann',
            'nperseg': 1024,
            'noverlap': None,
            'detrend': 'constant',
            'scaling': 'density'
        }
        self.psd_data = []
        
    def calculate_psd(self, time_data, signal_data):
        """
        Calculate Power Spectral Density
        
        Parameters:
        -----------
        time_data : np.array
            Time points of the signal
        signal_data : np.array
            Signal values
            
        Returns:
        --------
        frequencies : np.array
            Frequency points (excluding DC)
        psd : np.array
            Power spectral density values (excluding DC)
        """
        try:
            # Calculate sampling frequency
            fs = 1 / np.mean(np.diff(time_data))
            
            # Remove any NaN or inf values
            mask = np.isfinite(signal_data)
            clean_signal = signal_data[mask]
            
            # Remove mean (DC component)
            clean_signal = clean_signal - np.mean(clean_signal)
            
            # Calculate optimal segment length
            nperseg = min(len(clean_signal), int(fs * 2))  # 2-second segments
            nperseg = 2 ** int(np.log2(nperseg))  # ensure power of 2 for efficiency
            
            # Calculate PSD using Welch's method
            frequencies, psd = signal.welch(clean_signal, 
                                          fs=fs,
                                          nperseg=nperseg,
                                          noverlap=nperseg//2,
                                          detrend='constant',
                                          scaling='density')
            
            # Remove DC component (first point)
            frequencies = frequencies[1:]
            psd = psd[1:]
            
            return frequencies, psd
            
        except Exception as e:
            logging.error(f"Error in PSD calculation: {str(e)}", exc_info=True)
            raise
            
    def apply_frequency_filter(self, frequencies, psd, f_min=None, f_max=None):
        """
        Apply frequency range filter to PSD data
        """
        try:
            mask = np.ones_like(frequencies, dtype=bool)
            
            if f_min is not None:
                mask &= (frequencies >= f_min)
            if f_max is not None:
                mask &= (frequencies <= f_max)
                
            return frequencies[mask], psd[mask]
            
        except Exception as e:
            logging.error(f"Error in frequency filtering: {str(e)}", exc_info=True)
            raise
            
    def calculate_psd_metrics(self, frequencies, psd):
        """
        Calculate various metrics from PSD data
        """
        try:
            metrics = {
                'total_power': np.trapz(psd, frequencies),
                'peak_frequency': frequencies[np.argmax(psd)],
                'mean_power': np.mean(psd),
                'median_power': np.median(psd),
                'std_power': np.std(psd)
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error in PSD metrics calculation: {str(e)}", exc_info=True)
            raise

class DisplaySettings:
    """Manage display settings for all plots"""
    def __init__(self):
        # Color settings
        self.colors = {
            'voltage': '#0C5DA5',    # Dark blue
            'current': '#00B945',    # Green
            'conductance': '#FF9500', # Orange
            'fit': '#FF2C00',        # Red
            'background': '#FFFFFF',  # White
            'grid': '#E6E6E6'        # Light gray
        }
        
        # Font settings
        self.fonts = {
            'family': 'Arial',
            'sizes': {
                'title': 16,          # 主标题
                'label': 14,          # 轴标签
                'tick': 12,           # 刻度标签
                'legend': 12,         # 图例
                'annotation': 12      # 注释
            },
            'weights': {
                'title': 'bold',      
                'label': 'bold',      
                'tick': 'bold',       
                'legend': 'bold'      
            }
        }
        
        # Line settings
        self.lines = {
            'widths': {
                'data': 1.0,          # 数据线
                'grid': 0.6,          # 网格线
                'axis': 1.0,          # 坐标轴
                'tick': 0.8,          # 刻度线
                'fit': 1.2            # 拟合线
            },
            'styles': {
                'data': '-',
                'grid': ':',
                'fit': '--'
            }
        }
        
        # Marker settings
        self.markers = {
            'size': 4,               # 标记大小
            'edge_width': 0.8        # 标记边缘
        }
        
        # Spacing settings
        self.spacing = {
            'label_pad': 6,          
            'tick_pad': 4,           
            'title_pad': 8           
        }
        
    def apply_rcparams(self):
        """Apply settings to matplotlib rcParams"""
        plt.rcParams.update({
            # Font settings
            'font.family': self.fonts['family'],
            'font.size': self.fonts['sizes']['tick'],
            'font.weight': self.fonts['weights']['tick'],
            
            # Figure settings
            'figure.dpi': 100,
            'figure.facecolor': self.colors['background'],
            'figure.edgecolor': self.colors['background'],
            'figure.titlesize': self.fonts['sizes']['title'],
            'figure.titleweight': self.fonts['weights']['title'],
            
            # Axes settings
            'axes.facecolor': self.colors['background'],
            'axes.edgecolor': 'black',
            'axes.labelsize': self.fonts['sizes']['label'],
            'axes.titlesize': self.fonts['sizes']['title'],
            'axes.linewidth': self.lines['widths']['axis'],
            'axes.grid': False,
            'axes.labelweight': self.fonts['weights']['label'],
            'axes.titleweight': self.fonts['weights']['title'],
            
            # Tick settings
            'xtick.labelsize': self.fonts['sizes']['tick'],
            'ytick.labelsize': self.fonts['sizes']['tick'],
            'xtick.major.width': self.lines['widths']['tick'],
            'ytick.major.width': self.lines['widths']['tick'],
            'xtick.minor.width': self.lines['widths']['tick'] * 0.8,
            'ytick.minor.width': self.lines['widths']['tick'] * 0.8,
            'xtick.major.size': 6,   # 主刻度线长度
            'ytick.major.size': 6,
            'xtick.minor.size': 4,   # 副刻度线长度
            'ytick.minor.size': 4,
            'xtick.major.pad': self.spacing['tick_pad'],
            'ytick.major.pad': self.spacing['tick_pad'],
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            
            # Legend settings
            'legend.fontsize': self.fonts['sizes']['legend'],
            'legend.frameon': False,
            'legend.borderpad': 0.4,
            'legend.borderaxespad': 0.5,
            'legend.handlelength': 1.5,
            'legend.handleheight': 0.8,
            'legend.handletextpad': 0.8,
            
            # Lines settings
            'lines.linewidth': self.lines['widths']['data'],
            'lines.markersize': self.markers['size'],
            'lines.markeredgewidth': self.markers['edge_width'],
            
            # Grid settings
            'grid.linewidth': self.lines['widths']['grid'],
            'grid.alpha': 0.3
        })

    def setup_axis(self, ax, xlabel=None, ylabel=None, title=None):
        """Setup common axis properties"""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.fonts['sizes']['label'],
                         fontweight=self.fonts['weights']['label'],
                         labelpad=self.spacing['label_pad'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.fonts['sizes']['label'],
                         fontweight=self.fonts['weights']['label'],
                         labelpad=self.spacing['label_pad'])
        if title:
            ax.set_title(title, fontsize=self.fonts['sizes']['title'],
                        fontweight=self.fonts['weights']['title'],
                        pad=self.spacing['title_pad'])

        # Configure ticks - 使用正确的参数
        ax.tick_params(which='major', direction='in', top=True, right=True,
                      labelsize=self.fonts['sizes']['tick'],
                      width=self.lines['widths']['tick'],
                      length=6)
                      
        ax.tick_params(which='minor', direction='in', top=True, right=True,
                      width=self.lines['widths']['tick'] * 0.8,
                      length=4)
        
        # Configure spines
        for spine in ax.spines.values():
            spine.set_linewidth(self.lines['widths']['axis'])

        # Add minor ticks
        ax.minorticks_on()
        
        # Set tick label weights
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight(self.fonts['weights']['tick'])
            
    def format_legend(self, ax, **kwargs):
        """Format legend with consistent style"""
        legend = ax.legend(**kwargs)
        if legend:
            # Set font properties for all legend text
            for text in legend.get_texts():
                text.set_fontsize(self.fonts['sizes']['legend'])
                text.set_fontweight(self.fonts['weights']['legend'])
            
            # Adjust legend line properties
            for handle in legend.legendHandles:
                handle.set_linewidth(self.lines['widths']['data'])

class PlotCustomizationWindow(QMainWindow):
    def __init__(self, parent=None, data=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle("Plot Customization")
        self.setGeometry(100, 100, 1200, 800)

        # Check if conductance data is available
        self.has_conductance = 'conductance' in self.data and self.data['conductance'] is not None

        # Initialize plot settings
        self.init_plot_settings()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: Controls
        left_widget = self.create_left_panel()
        main_layout.addWidget(left_widget)

        # Right side: Plot
        self.right_widget = self.create_right_panel()
        main_layout.addWidget(self.right_widget)

        # Set initial sizes
        main_layout.setStretchFactor(left_widget, 1)
        main_layout.setStretchFactor(self.right_widget, 3)

        # Use a timer to delay the initial plot update
        QTimer.singleShot(100, self.update_plot)

        # Show warning if conductance data is not available
        if not self.has_conductance:
            QMessageBox.warning(self, "Warning", "Conductance data is not available in the loaded file.")

    def init_plot_settings(self):
        self.plot_settings = {
            'voltage_color': "#0C5DA5",  # Dark blue
            'current_color': "#00B945",  # Green
            'conductance_color': "#FF9500",  # Orange
            'font_family': 'Arial',
            'font_size': 8,
            'title_font_size': 12,
            'label_font_size': 9,
            'tick_font_size': 8,
            'font_weight': 'bold',
            'line_width': 1.0,
            'marker_size': 4,
            'axis_line_width': 1.0,
            'tick_line_width': 1.0,
            'figure_size': (3.5, 2.625),
            'figure_width': 3.5,  # in inches
            'figure_height': 2.625,  # in inches
            'dpi': 300,
            'show_top_ticks': True,
            'show_bottom_ticks': True,
            'show_left_ticks': True,
            'show_right_ticks': True,
            'show_voltage_top_spine': True,
            'show_voltage_bottom_spine': True,
            'show_voltage_left_spine': True,
            'show_voltage_right_spine': True,
            'show_voltage_ylabel': True,
            'show_bottom_top_spine': True,  # 重命名为更通用的名称
            'show_bottom_bottom_spine': True,
            'show_bottom_left_spine': True,
            'show_bottom_right_spine': True,
            'show_bottom_xlabel': True,
            'show_bottom_ylabel': True,
            'voltage_x_min': None,
            'voltage_x_max': None,
            'voltage_y_min': None,
            'voltage_y_max': None,
            'bottom_x_min': None,  # 重命名为更通用的名称
            'bottom_x_max': None,
            'bottom_y_min': None,
            'bottom_y_max': None,
            'show_conductance': False,
        }

    def create_left_panel(self):
        # Create a main widget to hold everything
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # 减少边距

        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create a widget to hold all the controls
        left_panel = QWidget()
        layout = QVBoxLayout(left_panel)
        layout.setSpacing(10)  # 控件之间的间距

        # Data display selection
        data_display_group = QGroupBox("Data Display")
        data_display_layout = QVBoxLayout()
        self.current_radio = QRadioButton("Current")
        self.conductance_radio = QRadioButton("Conductance")
        self.current_radio.setChecked(True)
        self.conductance_radio.setEnabled(self.has_conductance)
        
        self.current_radio.toggled.connect(self.toggle_data_display)
        self.conductance_radio.toggled.connect(self.toggle_data_display)
        
        data_display_layout.addWidget(self.current_radio)
        data_display_layout.addWidget(self.conductance_radio)
        data_display_group.setLayout(data_display_layout)
        layout.addWidget(data_display_group)

        # Figure size controls
        size_group = QGroupBox("Figure Size")
        size_layout = QFormLayout()
        
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1, 12)
        self.width_spin.setSingleStep(0.1)
        self.width_spin.setValue(self.plot_settings['figure_width'])
        self.width_spin.valueChanged.connect(self.update_figure_size)
        
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 12)
        self.height_spin.setSingleStep(0.1)
        self.height_spin.setValue(self.plot_settings['figure_height'])
        self.height_spin.valueChanged.connect(self.update_figure_size)
        
        size_layout.addRow("Width (inches):", self.width_spin)
        size_layout.addRow("Height (inches):", self.height_spin)
        
        self.aspect_ratio_check = QCheckBox("Lock Aspect Ratio")
        self.aspect_ratio_check.setChecked(False)
        self.aspect_ratio_check.stateChanged.connect(self.toggle_aspect_ratio)
        size_layout.addRow(self.aspect_ratio_check)
        
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # Color controls
        color_group = QGroupBox("Colors")
        color_layout = QFormLayout()
        self.voltage_color_button = QPushButton()
        self.voltage_color_button.setStyleSheet(f"background-color: {self.plot_settings['voltage_color']};")
        self.voltage_color_button.clicked.connect(lambda: self.choose_color('voltage_color'))
        self.current_color_button = QPushButton()
        self.current_color_button.setStyleSheet(f"background-color: {self.plot_settings['current_color']};")
        self.current_color_button.clicked.connect(lambda: self.choose_color('current_color'))
        self.conductance_color_button = QPushButton()
        self.conductance_color_button.setStyleSheet(f"background-color: {self.plot_settings['conductance_color']};")
        self.conductance_color_button.clicked.connect(lambda: self.choose_color('conductance_color'))
        
        # 设置颜色按钮的最小尺寸
        min_button_size = QSize(40, 20)
        for button in [self.voltage_color_button, self.current_color_button, self.conductance_color_button]:
            button.setMinimumSize(min_button_size)
        
        color_layout.addRow("Voltage Color:", self.voltage_color_button)
        color_layout.addRow("Current Color:", self.current_color_button)
        color_layout.addRow("Conductance Color:", self.conductance_color_button)
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        # Font controls
        font_group = QGroupBox("Font")
        font_layout = QFormLayout()
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(self.plot_settings['font_family']))
        self.font_combo.currentFontChanged.connect(self.update_font)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 24)
        self.font_size_spin.setValue(self.plot_settings['font_size'])
        self.font_size_spin.valueChanged.connect(self.update_font_size)
        self.font_weight_check = QCheckBox("Bold")
        self.font_weight_check.setChecked(True)
        self.font_weight_check.stateChanged.connect(self.update_font_weight)
        font_layout.addRow("Font:", self.font_combo)
        font_layout.addRow("Font Size:", self.font_size_spin)
        font_layout.addRow("Font Weight:", self.font_weight_check)
        font_group.setLayout(font_layout)
        layout.addWidget(font_group)

        # Line style controls
        line_group = QGroupBox("Line Style")
        line_layout = QFormLayout()
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 5)
        self.line_width_spin.setSingleStep(0.1)
        self.line_width_spin.setValue(self.plot_settings['line_width'])
        self.line_width_spin.valueChanged.connect(self.update_line_width)
        line_layout.addRow("Line Width:", self.line_width_spin)
        line_group.setLayout(line_layout)
        layout.addWidget(line_group)

        # Axis style controls
        axis_group = QGroupBox("Axis Style")
        axis_layout = QFormLayout()
        self.axis_line_width_spin = QDoubleSpinBox()
        self.axis_line_width_spin.setRange(0.5, 5)
        self.axis_line_width_spin.setSingleStep(0.1)
        self.axis_line_width_spin.setValue(self.plot_settings['axis_line_width'])
        self.axis_line_width_spin.valueChanged.connect(self.update_axis_line_width)
        self.tick_line_width_spin = QDoubleSpinBox()
        self.tick_line_width_spin.setRange(0.5, 5)
        self.tick_line_width_spin.setSingleStep(0.1)
        self.tick_line_width_spin.setValue(self.plot_settings['tick_line_width'])
        self.tick_line_width_spin.valueChanged.connect(self.update_tick_line_width)
        axis_layout.addRow("Axis Line Width:", self.axis_line_width_spin)
        axis_layout.addRow("Tick Line Width:", self.tick_line_width_spin)
        axis_group.setLayout(axis_layout)
        layout.addWidget(axis_group)

        # Tick visibility controls
        tick_group = QGroupBox("Tick Visibility")
        tick_layout = QGridLayout()
        self.top_tick_check = QCheckBox("Top")
        self.bottom_tick_check = QCheckBox("Bottom")
        self.left_tick_check = QCheckBox("Left")
        self.right_tick_check = QCheckBox("Right")
        
        self.top_tick_check.setChecked(self.plot_settings['show_top_ticks'])
        self.bottom_tick_check.setChecked(self.plot_settings['show_bottom_ticks'])
        self.left_tick_check.setChecked(self.plot_settings['show_left_ticks'])
        self.right_tick_check.setChecked(self.plot_settings['show_right_ticks'])
        
        self.top_tick_check.stateChanged.connect(lambda state: self.update_tick_visibility('top', state))
        self.bottom_tick_check.stateChanged.connect(lambda state: self.update_tick_visibility('bottom', state))
        self.left_tick_check.stateChanged.connect(lambda state: self.update_tick_visibility('left', state))
        self.right_tick_check.stateChanged.connect(lambda state: self.update_tick_visibility('right', state))
        
        tick_layout.addWidget(self.top_tick_check, 0, 0)
        tick_layout.addWidget(self.bottom_tick_check, 1, 0)
        tick_layout.addWidget(self.left_tick_check, 0, 1)
        tick_layout.addWidget(self.right_tick_check, 1, 1)
        tick_group.setLayout(tick_layout)
        layout.addWidget(tick_group)

        # Visibility controls
        visibility_group = QGroupBox("Visibility Controls")
        visibility_layout = QGridLayout()

        # Voltage plot controls
        visibility_layout.addWidget(QLabel("Voltage Plot:"), 0, 0, 1, 2)
        self.voltage_top_spine_check = QCheckBox("Top Spine")
        self.voltage_bottom_spine_check = QCheckBox("Bottom Spine")
        self.voltage_left_spine_check = QCheckBox("Left Spine")
        self.voltage_right_spine_check = QCheckBox("Right Spine")
        self.voltage_ylabel_check = QCheckBox("Y Label")

        visibility_layout.addWidget(self.voltage_top_spine_check, 1, 0)
        visibility_layout.addWidget(self.voltage_bottom_spine_check, 1, 1)
        visibility_layout.addWidget(self.voltage_left_spine_check, 2, 0)
        visibility_layout.addWidget(self.voltage_right_spine_check, 2, 1)
        visibility_layout.addWidget(self.voltage_ylabel_check, 3, 0)

        # Bottom plot controls (Current/Conductance)
        visibility_layout.addWidget(QLabel("Current/Conductance Plot:"), 4, 0, 1, 2)
        self.bottom_top_spine_check = QCheckBox("Top Spine")
        self.bottom_bottom_spine_check = QCheckBox("Bottom Spine")
        self.bottom_left_spine_check = QCheckBox("Left Spine")
        self.bottom_right_spine_check = QCheckBox("Right Spine")
        self.bottom_xlabel_check = QCheckBox("X Label")
        self.bottom_ylabel_check = QCheckBox("Y Label")

        visibility_layout.addWidget(self.bottom_top_spine_check, 5, 0)
        visibility_layout.addWidget(self.bottom_bottom_spine_check, 5, 1)
        visibility_layout.addWidget(self.bottom_left_spine_check, 6, 0)
        visibility_layout.addWidget(self.bottom_right_spine_check, 6, 1)
        visibility_layout.addWidget(self.bottom_xlabel_check, 7, 0)
        visibility_layout.addWidget(self.bottom_ylabel_check, 7, 1)

        # Set initial checkbox states and connect signals
        for check, setting in [
            (self.voltage_top_spine_check, 'show_voltage_top_spine'),
            (self.voltage_bottom_spine_check, 'show_voltage_bottom_spine'),
            (self.voltage_left_spine_check, 'show_voltage_left_spine'),
            (self.voltage_right_spine_check, 'show_voltage_right_spine'),
            (self.voltage_ylabel_check, 'show_voltage_ylabel'),
            (self.bottom_top_spine_check, 'show_bottom_top_spine'),
            (self.bottom_bottom_spine_check, 'show_bottom_bottom_spine'),
            (self.bottom_left_spine_check, 'show_bottom_left_spine'),
            (self.bottom_right_spine_check, 'show_bottom_right_spine'),
            (self.bottom_xlabel_check, 'show_bottom_xlabel'),
            (self.bottom_ylabel_check, 'show_bottom_ylabel'),
        ]:
            check.setChecked(self.plot_settings[setting])
            check.stateChanged.connect(self.update_visibility)

        visibility_group.setLayout(visibility_layout)
        layout.addWidget(visibility_group)

        # Axis range controls
        range_group = QGroupBox("Axis Range")
        range_layout = QGridLayout()
        
        self.voltage_x_min_input = QLineEdit()
        self.voltage_x_max_input = QLineEdit()
        self.voltage_y_min_input = QLineEdit()
        self.voltage_y_max_input = QLineEdit()
        self.bottom_x_min_input = QLineEdit()
        self.bottom_x_max_input = QLineEdit()
        self.bottom_y_min_input = QLineEdit()
        self.bottom_y_max_input = QLineEdit()
        
        range_layout.addWidget(QLabel("Voltage X Min:"), 0, 0)
        range_layout.addWidget(self.voltage_x_min_input, 0, 1)
        range_layout.addWidget(QLabel("Voltage X Max:"), 0, 2)
        range_layout.addWidget(self.voltage_x_max_input, 0, 3)
        range_layout.addWidget(QLabel("Voltage Y Min:"), 1, 0)
        range_layout.addWidget(self.voltage_y_min_input, 1, 1)
        range_layout.addWidget(QLabel("Voltage Y Max:"), 1, 2)
        range_layout.addWidget(self.voltage_y_max_input, 1, 3)
        
        bottom_label = "Current/Conductance"
        range_layout.addWidget(QLabel(f"{bottom_label} X Min:"), 2, 0)
        range_layout.addWidget(self.bottom_x_min_input, 2, 1)
        range_layout.addWidget(QLabel(f"{bottom_label} X Max:"), 2, 2)
        range_layout.addWidget(self.bottom_x_max_input, 2, 3)
        range_layout.addWidget(QLabel(f"{bottom_label} Y Min:"), 3, 0)
        range_layout.addWidget(self.bottom_y_min_input, 3, 1)
        range_layout.addWidget(QLabel(f"{bottom_label} Y Max:"), 3, 2)
        range_layout.addWidget(self.bottom_y_max_input, 3, 3)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)

        # Update button
        self.update_button = QPushButton("Update Plot")
        self.update_button.clicked.connect(self.update_plot)
        layout.addWidget(self.update_button)

        # Set up the scroll area
        scroll.setWidget(left_panel)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll)

        # Set a minimum width for the left panel to prevent controls from being too squeezed
        left_panel.setMinimumWidth(250)
        
        # Add style sheet for better visual appearance
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        return main_widget
    
    def update_figure_size(self):
        new_width = self.width_spin.value()
        new_height = self.height_spin.value()
        
        if self.aspect_ratio_check.isChecked():
            # Maintain aspect ratio
            current_ratio = self.plot_settings['figure_width'] / self.plot_settings['figure_height']
            if self.sender() == self.width_spin:
                new_height = new_width / current_ratio
                self.height_spin.setValue(new_height)
            else:
                new_width = new_height * current_ratio
                self.width_spin.setValue(new_width)

        self.plot_settings['figure_width'] = new_width
        self.plot_settings['figure_height'] = new_height
        
        self.update_plot()

    def toggle_aspect_ratio(self, state):
        if state == 2:  # 2 represents the checked state
            # If aspect ratio is locked, adjust height to match current ratio
            current_ratio = self.plot_settings['figure_width'] / self.plot_settings['figure_height']
            new_height = self.width_spin.value() / current_ratio
            self.height_spin.setValue(new_height)
    
    def update_tick_visibility(self, side, state):
        self.plot_settings[f'show_{side}_ticks'] = state == 2  # 2 means checked
        self.update_plot()

    def update_visibility(self):
        # Update plot settings based on checkbox states
        self.plot_settings['show_voltage_top_spine'] = self.voltage_top_spine_check.isChecked()
        self.plot_settings['show_voltage_bottom_spine'] = self.voltage_bottom_spine_check.isChecked()
        self.plot_settings['show_voltage_left_spine'] = self.voltage_left_spine_check.isChecked()
        self.plot_settings['show_voltage_right_spine'] = self.voltage_right_spine_check.isChecked()
        self.plot_settings['show_voltage_ylabel'] = self.voltage_ylabel_check.isChecked()
        
        self.plot_settings['show_bottom_top_spine'] = self.bottom_top_spine_check.isChecked()
        self.plot_settings['show_bottom_bottom_spine'] = self.bottom_bottom_spine_check.isChecked()
        self.plot_settings['show_bottom_left_spine'] = self.bottom_left_spine_check.isChecked()
        self.plot_settings['show_bottom_right_spine'] = self.bottom_right_spine_check.isChecked()
        self.plot_settings['show_bottom_xlabel'] = self.bottom_xlabel_check.isChecked()
        self.plot_settings['show_bottom_ylabel'] = self.bottom_ylabel_check.isChecked()

        self.update_plot()

    def toggle_data_display(self):
        self.plot_settings['show_conductance'] = self.conductance_radio.isChecked()
        self.update_plot()

    def create_right_panel(self):
        right_panel = QWidget()
        layout = QVBoxLayout(right_panel)
        
        self.figure = Figure(figsize=self.plot_settings['figure_size'], dpi=self.plot_settings['dpi'])
        
        gs = self.figure.add_gridspec(4, 1)
        self.ax_voltage = self.figure.add_subplot(gs[0, 0])
        self.ax_current = self.figure.add_subplot(gs[1:, 0], sharex=self.ax_voltage)
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(right_panel)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        return right_panel

    def choose_color(self, color_key):
        color = QColorDialog.getColor()
        if color.isValid():
            self.plot_settings[color_key] = color.name()
            # 更新颜色按钮显示
            if color_key == 'voltage_color':
                self.voltage_color_button.setStyleSheet(f"background-color: {color.name()};")
            elif color_key == 'current_color':
                self.current_color_button.setStyleSheet(f"background-color: {color.name()};")
            elif color_key == 'conductance_color':
                self.conductance_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.update_plot()

    def update_font(self, font):
        self.plot_settings['font_family'] = font.family()

    def update_font_size(self, size):
        self.plot_settings['font_size'] = size
        self.plot_settings['title_font_size'] = size + 2
        self.plot_settings['label_font_size'] = size
        self.plot_settings['tick_font_size'] = size - 2

    def update_font_weight(self, state):
        self.plot_settings['font_weight'] = 'bold' if state == 2 else 'normal'

    def update_line_width(self, width):
        self.plot_settings['line_width'] = width

    def update_axis_line_width(self, width):
        self.plot_settings['axis_line_width'] = width

    def update_tick_line_width(self, width):
        self.plot_settings['tick_line_width'] = width

    def setup_plot_style(self):
        plt.style.use('default')
        
        plt.rcParams.update({
            'font.family': self.plot_settings['font_family'],
            'font.size': self.plot_settings['font_size'],
            'font.weight': self.plot_settings['font_weight'],
            'axes.labelsize': self.plot_settings['label_font_size'],
            'axes.titlesize': self.plot_settings['title_font_size'],
            'xtick.labelsize': self.plot_settings['tick_font_size'],
            'ytick.labelsize': self.plot_settings['tick_font_size'],
            'lines.linewidth': self.plot_settings['line_width'],
            'lines.markersize': self.plot_settings['marker_size'],
            'axes.linewidth': self.plot_settings['axis_line_width'],
            'axes.edgecolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'text.color': 'black',
            'figure.figsize': self.plot_settings['figure_size'],
            'figure.dpi': self.plot_settings['dpi'],
            'legend.fontsize': self.plot_settings['tick_font_size'],
            'legend.frameon': False,
            'legend.handlelength': 1.5,
            'xtick.major.width': self.plot_settings['tick_line_width'],
            'ytick.major.width': self.plot_settings['tick_line_width'],
            'xtick.minor.width': self.plot_settings['tick_line_width'] * 0.8,
            'ytick.minor.width': self.plot_settings['tick_line_width'] * 0.8,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'axes.prop_cycle': plt.cycler('color', [self.plot_settings['voltage_color'], 
                                                    self.plot_settings['current_color'], 
                                                    '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']), 
            'figure.figsize': (self.plot_settings['figure_width'], self.plot_settings['figure_height'])
        })

    def update_plot(self):
        self.setup_plot_style()

        # Update figure size
        self.figure.set_size_inches(self.plot_settings['figure_width'], self.plot_settings['figure_height'])
        
        # Clear previous plots
        self.ax_voltage.clear()
        self.ax_current.clear()

        if self.data:
            # Plot voltage data
            self.ax_voltage.plot(self.data['time'], self.data['voltage'], 
                                color=self.plot_settings['voltage_color'],
                                linewidth=self.plot_settings['line_width'])
            self.ax_voltage.set_ylabel('Voltage (V)', fontweight='bold')

            # Plot current or conductance data based on selection
            if self.plot_settings['show_conductance'] and self.has_conductance:
                y_data = self.data['conductance']
                ylabel = 'Conductance (nS)'
                color = self.plot_settings['conductance_color']
            else:
                y_data = self.data['current']
                ylabel = 'Current (nA)'
                color = self.plot_settings['current_color']

            self.ax_current.plot(self.data['time'], y_data, 
                                color=color,
                                linewidth=self.plot_settings['line_width'])
            self.ax_current.set_ylabel(ylabel, fontweight='bold')
            self.ax_current.set_xlabel('Time (s)', fontweight='bold')

            # Update plot settings from input fields
            self.update_plot_settings_from_inputs()

            # Set axis limits for voltage plot
            if self.plot_settings['voltage_x_min'] is not None and self.plot_settings['voltage_x_max'] is not None:
                self.ax_voltage.set_xlim(self.plot_settings['voltage_x_min'], self.plot_settings['voltage_x_max'])
            else:
                self.ax_voltage.set_xlim(self.data['time'][0], self.data['time'][-1])

            if self.plot_settings['voltage_y_min'] is not None and self.plot_settings['voltage_y_max'] is not None:
                self.ax_voltage.set_ylim(self.plot_settings['voltage_y_min'], self.plot_settings['voltage_y_max'])

            # Set axis limits for bottom plot (current or conductance)
            if self.plot_settings['bottom_x_min'] is not None and self.plot_settings['bottom_x_max'] is not None:
                self.ax_current.set_xlim(self.plot_settings['bottom_x_min'], self.plot_settings['bottom_x_max'])
            else:
                self.ax_current.set_xlim(self.data['time'][0], self.data['time'][-1])

            if self.plot_settings['bottom_y_min'] is not None and self.plot_settings['bottom_y_max'] is not None:
                self.ax_current.set_ylim(self.plot_settings['bottom_y_min'], self.plot_settings['bottom_y_max'])

        # Apply spine visibility settings for voltage plot
        self.ax_voltage.spines['top'].set_visible(self.plot_settings['show_voltage_top_spine'])
        self.ax_voltage.spines['bottom'].set_visible(self.plot_settings['show_voltage_bottom_spine'])
        self.ax_voltage.spines['left'].set_visible(self.plot_settings['show_voltage_left_spine'])
        self.ax_voltage.spines['right'].set_visible(self.plot_settings['show_voltage_right_spine'])
        self.ax_voltage.yaxis.label.set_visible(self.plot_settings['show_voltage_ylabel'])

        # Apply spine visibility settings for bottom plot
        self.ax_current.spines['top'].set_visible(self.plot_settings['show_bottom_top_spine'])
        self.ax_current.spines['bottom'].set_visible(self.plot_settings['show_bottom_bottom_spine'])
        self.ax_current.spines['left'].set_visible(self.plot_settings['show_bottom_left_spine'])
        self.ax_current.spines['right'].set_visible(self.plot_settings['show_bottom_right_spine'])
        self.ax_current.xaxis.label.set_visible(self.plot_settings['show_bottom_xlabel'])
        self.ax_current.yaxis.label.set_visible(self.plot_settings['show_bottom_ylabel'])

        # Setup axes properties
        for ax in [self.ax_voltage, self.ax_current]:
            ax.tick_params(which='both', 
                        top=self.plot_settings['show_top_ticks'],
                        bottom=self.plot_settings['show_bottom_ticks'],
                        left=self.plot_settings['show_left_ticks'],
                        right=self.plot_settings['show_right_ticks'],
                        labeltop=False,
                        labelbottom=True,
                        labelleft=True,
                        labelright=False,
                        width=self.plot_settings['tick_line_width'])
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            for spine in ax.spines.values():
                spine.set_linewidth(self.plot_settings['axis_line_width'])

        # Hide x-axis labels for voltage plot
        self.ax_voltage.tick_params(labelbottom=False)
        
        # Show x-axis labels for bottom plot
        self.ax_current.tick_params(labelbottom=True)

        # Adjust subplot spacing
        self.figure.subplots_adjust(hspace=0.1)
        
        # Apply tight layout and update canvas
        self.figure.tight_layout()
        self.canvas.draw()

    def update_plot_settings_from_inputs(self):
        try:
            self.plot_settings['voltage_x_min'] = float(self.voltage_x_min_input.text()) if self.voltage_x_min_input.text() else None
            self.plot_settings['voltage_x_max'] = float(self.voltage_x_max_input.text()) if self.voltage_x_max_input.text() else None
            self.plot_settings['voltage_y_min'] = float(self.voltage_y_min_input.text()) if self.voltage_y_min_input.text() else None
            self.plot_settings['voltage_y_max'] = float(self.voltage_y_max_input.text()) if self.voltage_y_max_input.text() else None
            self.plot_settings['bottom_x_min'] = float(self.bottom_x_min_input.text()) if self.bottom_x_min_input.text() else None
            self.plot_settings['bottom_x_max'] = float(self.bottom_x_max_input.text()) if self.bottom_x_max_input.text() else None
            self.plot_settings['bottom_y_min'] = float(self.bottom_y_min_input.text()) if self.bottom_y_min_input.text() else None
            self.plot_settings['bottom_y_max'] = float(self.bottom_y_max_input.text()) if self.bottom_y_max_input.text() else None
            self.plot_settings['show_conductance'] = self.conductance_radio.isChecked()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for axis ranges.")


    def setup_axes(self):
        for ax in [self.ax_voltage, self.ax_current]:
            ax.tick_params(which='both', 
                           top=self.plot_settings['show_top_ticks'],
                           bottom=self.plot_settings['show_bottom_ticks'],
                           left=self.plot_settings['show_left_ticks'],
                           right=self.plot_settings['show_right_ticks'],
                           labeltop=False,
                           labelbottom=True,
                           labelleft=True,
                           labelright=False,
                           width=self.plot_settings['tick_line_width'])
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            for spine in ax.spines.values():
                spine.set_linewidth(self.plot_settings['axis_line_width'])

        self.ax_voltage.tick_params(labelbottom=False)
        self.ax_current.tick_params(labelbottom=True)

        self.figure.subplots_adjust(hspace=0.1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update plot size when window is resized
        self.update_plot()

    def save_plot(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", 
                                                   "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)", 
                                                   options=options)
        if file_name:
            self.figure.savefig(file_name, dpi=self.plot_settings['dpi'], bbox_inches='tight')
            QMessageBox.information(self, "Success", f"Plot saved as {file_name}")

class PSDPlotCustomizationWindow(QMainWindow):
    def __init__(self, parent=None, psd_data=None):
        super().__init__(parent)
        self.psd_data = psd_data
        self.setWindowTitle("PSD Plot Customization")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize plot settings
        self.init_plot_settings()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: Controls
        left_widget = self.create_left_panel()
        main_layout.addWidget(left_widget)

        # Right side: Plot
        self.right_widget = self.create_right_panel()
        main_layout.addWidget(self.right_widget)

        # Set initial sizes
        main_layout.setStretchFactor(left_widget, 1)
        main_layout.setStretchFactor(self.right_widget, 3)

        # Use a timer to delay the initial plot update
        QTimer.singleShot(100, self.update_plot)

    def init_plot_settings(self):
        self.plot_settings = {
            'x_min': None,
            'x_max': None,
            'y_min': None,
            'y_max': None,
            'psd_color': "#0C5DA5",  # Dark blue
            'font_family': 'Arial',
            'font_size': 8,
            'title_font_size': 12,
            'label_font_size': 9,
            'tick_font_size': 8,
            'font_weight': 'bold',
            'line_width': 1.0,
            'marker_size': 4,
            'axis_line_width': 1.0,
            'tick_line_width': 1.0,
            'figure_size': (3.5, 2.625),
            'figure_width': 3.5,
            'figure_height': 2.625,
            'dpi': 300,
            'show_grid': True,
            'grid_alpha': 0.3,
            'grid_linestyle': ':',
            'show_top_ticks': True,
            'show_bottom_ticks': True,
            'show_left_ticks': True,
            'show_right_ticks': True,
            'show_top_spine': True,
            'show_bottom_spine': True,
            'show_left_spine': True,
            'show_right_spine': True,
            'show_xlabel': True,
            'show_ylabel': True,
            'x_scale': 'log',
            'y_scale': 'log',
            'show_legend': True,
            'legend_location': 'best',
            'legend_font_size': 8,
            'curve_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'curve_alphas': [1.0] * 5  # Default alpha for each curve
        }

    def create_left_panel(self):
        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Main widget that will contain all controls
        left_panel = QWidget()
        layout = QVBoxLayout(left_panel)

        # Figure size controls
        size_group = QGroupBox("Figure Size")
        size_layout = QFormLayout()
        
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1, 12)
        self.width_spin.setSingleStep(0.1)
        self.width_spin.setValue(self.plot_settings['figure_width'])
        self.width_spin.valueChanged.connect(self.update_plot)
        
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 12)
        self.height_spin.setSingleStep(0.1)
        self.height_spin.setValue(self.plot_settings['figure_height'])
        self.height_spin.valueChanged.connect(self.update_plot)
        
        size_layout.addRow("Width (inches):", self.width_spin)
        size_layout.addRow("Height (inches):", self.height_spin)
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # Font controls
        font_group = QGroupBox("Font")
        font_layout = QFormLayout()
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(self.plot_settings['font_family']))
        self.font_combo.currentFontChanged.connect(self.update_font)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 24)
        self.font_size_spin.setValue(self.plot_settings['font_size'])
        self.font_size_spin.valueChanged.connect(self.update_font_size)
        self.font_weight_check = QCheckBox("Bold")
        self.font_weight_check.setChecked(True)
        self.font_weight_check.stateChanged.connect(self.update_font_weight)
        font_layout.addRow("Font:", self.font_combo)
        font_layout.addRow("Font Size:", self.font_size_spin)
        font_layout.addRow("Font Weight:", self.font_weight_check)
        font_group.setLayout(font_layout)
        layout.addWidget(font_group)

        # Scale controls
        scale_group = QGroupBox("Scale Settings")
        scale_layout = QFormLayout()
        self.x_scale_combo = QComboBox()
        self.x_scale_combo.addItems(['linear', 'log'])
        self.x_scale_combo.setCurrentText(self.plot_settings['x_scale'])
        self.x_scale_combo.currentTextChanged.connect(self.update_plot)
        
        self.y_scale_combo = QComboBox()
        self.y_scale_combo.addItems(['linear', 'log'])
        self.y_scale_combo.setCurrentText(self.plot_settings['y_scale'])
        self.y_scale_combo.currentTextChanged.connect(self.update_plot)
        
        scale_layout.addRow("X Scale:", self.x_scale_combo)
        scale_layout.addRow("Y Scale:", self.y_scale_combo)
        scale_group.setLayout(scale_layout)
        layout.addWidget(scale_group)

        # Line style controls
        line_group = QGroupBox("Line Style")
        line_layout = QFormLayout()
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 5)
        self.line_width_spin.setSingleStep(0.1)
        self.line_width_spin.setValue(self.plot_settings['line_width'])
        self.line_width_spin.valueChanged.connect(self.update_line_width)
        line_layout.addRow("Line Width:", self.line_width_spin)
        line_group.setLayout(line_layout)
        layout.addWidget(line_group)

        # Axis style controls
        axis_group = QGroupBox("Axis Style")
        axis_layout = QFormLayout()
        self.axis_line_width_spin = QDoubleSpinBox()
        self.axis_line_width_spin.setRange(0.5, 5)
        self.axis_line_width_spin.setSingleStep(0.1)
        self.axis_line_width_spin.setValue(self.plot_settings['axis_line_width'])
        self.axis_line_width_spin.valueChanged.connect(self.update_axis_line_width)
        self.tick_line_width_spin = QDoubleSpinBox()
        self.tick_line_width_spin.setRange(0.5, 5)
        self.tick_line_width_spin.setSingleStep(0.1)
        self.tick_line_width_spin.setValue(self.plot_settings['tick_line_width'])
        self.tick_line_width_spin.valueChanged.connect(self.update_tick_line_width)
        axis_layout.addRow("Axis Line Width:", self.axis_line_width_spin)
        axis_layout.addRow("Tick Line Width:", self.tick_line_width_spin)
        axis_group.setLayout(axis_layout)
        layout.addWidget(axis_group)

        # Tick visibility controls
        tick_group = QGroupBox("Tick Visibility")
        tick_layout = QGridLayout()
        self.top_tick_check = QCheckBox("Top")
        self.bottom_tick_check = QCheckBox("Bottom")
        self.left_tick_check = QCheckBox("Left")
        self.right_tick_check = QCheckBox("Right")
        
        self.top_tick_check.setChecked(self.plot_settings['show_top_ticks'])
        self.bottom_tick_check.setChecked(self.plot_settings['show_bottom_ticks'])
        self.left_tick_check.setChecked(self.plot_settings['show_left_ticks'])
        self.right_tick_check.setChecked(self.plot_settings['show_right_ticks'])
        
        self.top_tick_check.stateChanged.connect(lambda state: self.update_tick_visibility('top', state))
        self.bottom_tick_check.stateChanged.connect(lambda state: self.update_tick_visibility('bottom', state))
        self.left_tick_check.stateChanged.connect(lambda state: self.update_tick_visibility('left', state))
        self.right_tick_check.stateChanged.connect(lambda state: self.update_tick_visibility('right', state))
        
        tick_layout.addWidget(self.top_tick_check, 0, 0)
        tick_layout.addWidget(self.bottom_tick_check, 1, 0)
        tick_layout.addWidget(self.left_tick_check, 0, 1)
        tick_layout.addWidget(self.right_tick_check, 1, 1)
        tick_group.setLayout(tick_layout)
        layout.addWidget(tick_group)

        # Grid controls
        grid_group = QGroupBox("Grid Settings")
        grid_layout = QFormLayout()
        
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(self.plot_settings['show_grid'])
        self.grid_check.stateChanged.connect(self.update_plot)
        
        self.grid_alpha_spin = QDoubleSpinBox()
        self.grid_alpha_spin.setRange(0.1, 1.0)
        self.grid_alpha_spin.setSingleStep(0.1)
        self.grid_alpha_spin.setValue(self.plot_settings['grid_alpha'])
        self.grid_alpha_spin.valueChanged.connect(self.update_plot)
        
        self.grid_style_combo = QComboBox()
        self.grid_style_combo.addItems([':', '--', '-', '-.'])
        self.grid_style_combo.setCurrentText(self.plot_settings['grid_linestyle'])
        self.grid_style_combo.currentTextChanged.connect(self.update_plot)
        
        grid_layout.addRow("", self.grid_check)
        grid_layout.addRow("Grid Alpha:", self.grid_alpha_spin)
        grid_layout.addRow("Grid Style:", self.grid_style_combo)
        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)

        # Axis range controls
        range_group = QGroupBox("Axis Range")
        range_layout = QGridLayout()
        
        self.x_min_input = QLineEdit()
        self.x_max_input = QLineEdit()
        self.y_min_input = QLineEdit()
        self.y_max_input = QLineEdit()
        
        range_layout.addWidget(QLabel("X Min:"), 0, 0)
        range_layout.addWidget(self.x_min_input, 0, 1)
        range_layout.addWidget(QLabel("X Max:"), 0, 2)
        range_layout.addWidget(self.x_max_input, 0, 3)
        range_layout.addWidget(QLabel("Y Min:"), 1, 0)
        range_layout.addWidget(self.y_min_input, 1, 1)
        range_layout.addWidget(QLabel("Y Max:"), 1, 2)
        range_layout.addWidget(self.y_max_input, 1, 3)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)

        # Legend controls
        legend_group = QGroupBox("Legend Settings")
        legend_layout = QFormLayout()
        
        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(self.plot_settings['show_legend'])
        self.legend_check.stateChanged.connect(self.update_plot)
        
        self.legend_loc_combo = QComboBox()
        self.legend_loc_combo.addItems(['best', 'upper right', 'upper left', 'lower left', 'lower right', 
                                    'center left', 'center right', 'upper center', 'lower center'])
        self.legend_loc_combo.setCurrentText(self.plot_settings['legend_location'])
        self.legend_loc_combo.currentTextChanged.connect(self.update_plot)
        
        legend_layout.addRow("", self.legend_check)
        legend_layout.addRow("Position:", self.legend_loc_combo)
        legend_group.setLayout(legend_layout)
        layout.addWidget(legend_group)

        # In the create_left_panel method, add this inside the legend_group section:
        legend_layout.addRow("", self.legend_check)
        legend_layout.addRow("Position:", self.legend_loc_combo)

        # Add legend font size control
        self.legend_font_size = QSpinBox()
        self.legend_font_size.setRange(6, 24)
        self.legend_font_size.setValue(self.plot_settings['tick_font_size'])  # Default to tick font size
        self.legend_font_size.valueChanged.connect(self.update_plot)
        legend_layout.addRow("Font Size:", self.legend_font_size)

        # Curve controls
        curve_group = QGroupBox("Curve Settings")
        curve_layout = QGridLayout()

        for i in range(min(5, len(self.psd_data))):
            name = self.psd_data[i][0]
            
            # Color button
            color_button = QPushButton()
            color_button.setFixedWidth(50)
            color_button.setStyleSheet(f"background-color: {self.plot_settings['curve_colors'][i]}")
            color_button.clicked.connect(lambda checked, idx=i: self.choose_curve_color(idx))
            
            # Alpha control
            alpha_spin = QDoubleSpinBox()
            alpha_spin.setRange(0, 1)
            alpha_spin.setSingleStep(0.1)
            alpha_spin.setValue(self.plot_settings['curve_alphas'][i])
            alpha_spin.valueChanged.connect(lambda value, idx=i: self.update_curve_alpha(idx, value))
            
            curve_layout.addWidget(QLabel(f"Curve {i+1}:"), i, 0)
            curve_layout.addWidget(color_button, i, 1)
            curve_layout.addWidget(QLabel("Alpha:"), i, 2)
            curve_layout.addWidget(alpha_spin, i, 3)
            
        curve_group.setLayout(curve_layout)
        layout.addWidget(curve_group)

        # Update and Save buttons
        self.update_button = QPushButton("Update Plot")
        self.update_button.clicked.connect(self.update_plot)
        layout.addWidget(self.update_button)

        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        layout.addWidget(self.save_button)

        layout.addStretch()
        
        # Set the widget to the scroll area
        scroll.setWidget(left_panel)
        
        return scroll

    def create_right_panel(self):
        right_panel = QWidget()
        layout = QVBoxLayout(right_panel)
        
        self.figure = Figure(figsize=self.plot_settings['figure_size'], 
                           dpi=self.plot_settings['dpi'])
        self.ax = self.figure.add_subplot(111)
        
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        return right_panel

    def choose_color(self, color_key):
        color = QColorDialog.getColor()
        if color.isValid():
            self.plot_settings[color_key] = color.name()
            if color_key == 'psd_color':
                self.psd_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.update_plot()

    def update_font(self, font):
        self.plot_settings['font_family'] = font.family()
        self.update_plot()

    def update_font_size(self, size):
        self.plot_settings['font_size'] = size
        self.plot_settings['title_font_size'] = size + 2
        self.plot_settings['label_font_size'] = size
        self.plot_settings['tick_font_size'] = size - 2
        self.update_plot()

    def update_font_weight(self, state):
        self.plot_settings['font_weight'] = 'bold' if state == Qt.CheckState.Checked else 'normal'
        self.update_plot()

    def update_line_width(self, width):
        self.plot_settings['line_width'] = width
        self.update_plot()

    def update_axis_line_width(self, width):
        self.plot_settings['axis_line_width'] = width
        self.update_plot()

    def update_tick_line_width(self, width):
        self.plot_settings['tick_line_width'] = width
        self.update_plot()

    def update_tick_visibility(self, side, state):
        self.plot_settings[f'show_{side}_ticks'] = state == Qt.CheckState.Checked
        self.update_plot()

    def setup_rcparams(self):
        plt.rcParams.update({
            'font.family': self.plot_settings['font_family'],
            'font.size': self.plot_settings['font_size'],
            'font.weight': self.plot_settings['font_weight'],
            'axes.labelsize': self.plot_settings['label_font_size'],
            'axes.titlesize': self.plot_settings['title_font_size'],
            'xtick.labelsize': self.plot_settings['tick_font_size'],
            'ytick.labelsize': self.plot_settings['tick_font_size'],
            'axes.linewidth': self.plot_settings['axis_line_width'],
            'lines.linewidth': self.plot_settings['line_width'],
            'xtick.major.width': self.plot_settings['tick_line_width'],
            'ytick.major.width': self.plot_settings['tick_line_width'],
            'xtick.direction': 'in',
            'ytick.direction': 'in'
        })

    def choose_curve_color(self, index):
        color = QColorDialog.getColor()
        if color.isValid():
            self.plot_settings['curve_colors'][index] = color.name()
            sender = self.sender()
            sender.setStyleSheet(f"background-color: {color.name()}")
            self.update_plot()

    def update_curve_alpha(self, index, value):
        self.plot_settings['curve_alphas'][index] = value
        self.update_plot()

    def update_plot(self):
        if not self.psd_data:
            return

        try:
            self.plot_settings.update({
                'figure_width': self.width_spin.value(),
                'figure_height': self.height_spin.value(),
                'font_family': self.font_combo.currentFont().family(),
                'font_size': self.font_size_spin.value(),
                'line_width': self.line_width_spin.value(),
                'axis_line_width': self.axis_line_width_spin.value(),
                'tick_line_width': self.tick_line_width_spin.value(),
                'x_scale': self.x_scale_combo.currentText(),
                'y_scale': self.y_scale_combo.currentText(),
                'show_grid': self.grid_check.isChecked(),
                'grid_alpha': self.grid_alpha_spin.value(),
                'grid_linestyle': self.grid_style_combo.currentText(),
                'show_legend': self.legend_check.isChecked(),
                'legend_location': self.legend_loc_combo.currentText(),
                'legend_font_size': self.legend_font_size.value(),
                'x_min': float(self.x_min_input.text()) if self.x_min_input.text() else None,
                'x_max': float(self.x_max_input.text()) if self.x_max_input.text() else None,
                'y_min': float(self.y_min_input.text()) if self.y_min_input.text() else None,
                'y_max': float(self.y_max_input.text()) if self.y_max_input.text() else None
            })
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid numbers for axis ranges")
            return

        # Clear previous plot
        self.ax.clear()

        # Apply font settings
        plt.rcParams.update({
            'font.family': self.plot_settings['font_family'],
            'font.size': self.plot_settings['font_size'],
            'font.weight': self.plot_settings['font_weight'],
            'axes.labelsize': self.plot_settings['label_font_size'],
            'axes.titlesize': self.plot_settings['title_font_size'],
            'xtick.labelsize': self.plot_settings['tick_font_size'],
            'ytick.labelsize': self.plot_settings['tick_font_size']
        })

        # Plot PSD data
        for i, (name, frequencies, psd) in enumerate(self.psd_data):
            color = self.plot_settings['curve_colors'][i % len(self.plot_settings['curve_colors'])]
            alpha = self.plot_settings['curve_alphas'][i % len(self.plot_settings['curve_alphas'])]
            self.ax.plot(frequencies, psd,
                        color=color,
                        alpha=alpha,
                        linewidth=self.plot_settings['line_width'],
                        label=name)

        # Set scales
        self.ax.set_xscale(self.plot_settings['x_scale'])
        self.ax.set_yscale(self.plot_settings['y_scale'])

        # Set labels
        if self.plot_settings['show_xlabel']:
            self.ax.set_xlabel('Frequency (Hz)', 
                            fontsize=self.plot_settings['label_font_size'],
                            fontweight=self.plot_settings['font_weight'])
        if self.plot_settings['show_ylabel']:
            self.ax.set_ylabel('PSD (nA²/Hz)',
                            fontsize=self.plot_settings['label_font_size'],
                            fontweight=self.plot_settings['font_weight'])

        # Set axis limits
        if all(v is not None for v in [self.plot_settings['x_min'], self.plot_settings['x_max']]):
            self.ax.set_xlim(self.plot_settings['x_min'], self.plot_settings['x_max'])
        if all(v is not None for v in [self.plot_settings['y_min'], self.plot_settings['y_max']]):
            self.ax.set_ylim(self.plot_settings['y_min'], self.plot_settings['y_max'])

        # Configure ticks
        self.ax.tick_params(which='both',
                        top=self.plot_settings['show_top_ticks'],
                        bottom=self.plot_settings['show_bottom_ticks'],
                        left=self.plot_settings['show_left_ticks'],
                        right=self.plot_settings['show_right_ticks'],
                        direction='in',
                        width=self.plot_settings['tick_line_width'])

        # Set spine visibility and width
        for spine in self.ax.spines.values():
            spine.set_linewidth(self.plot_settings['axis_line_width'])
        self.ax.spines['top'].set_visible(self.plot_settings['show_top_spine'])
        self.ax.spines['bottom'].set_visible(self.plot_settings['show_bottom_spine'])
        self.ax.spines['left'].set_visible(self.plot_settings['show_left_spine'])
        self.ax.spines['right'].set_visible(self.plot_settings['show_right_spine'])

        # Configure grid
        self.ax.grid(self.plot_settings['show_grid'], 
                    which='both',
                    alpha=self.plot_settings['grid_alpha'],
                    linestyle=self.plot_settings['grid_linestyle'])

        # Update figure size
        self.figure.set_size_inches(self.plot_settings['figure_width'],
                                self.plot_settings['figure_height'])

        # Add legend with custom font size
        if self.plot_settings['show_legend'] and len(self.psd_data) > 0:
            self.ax.legend(loc=self.plot_settings['legend_location'],
                        fontsize=self.plot_settings['legend_font_size'])

        # Adjust layout and redraw
        self.figure.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", 
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_name:
            try:
                self.figure.savefig(file_name, dpi=self.plot_settings['dpi'], bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot saved as {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot: {str(e)}")


class PreprocessWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, main_window: 'MainWindow', file_list: List[Tuple[str, str, str, str]] = None):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Data Preprocessing")
        self.setGeometry(100, 100, 1600, 900)

        # Set application style
        self.setStyle(QStyleFactory.create("Fusion"))
        
        # Set color palette
        palette = self.create_light_palette()
        self.setPalette(palette)

        # Central data storage
        self.data = {
            'time': None,
            'voltage': None,
            'current': None,
            'conductance': None
        }
        self.is_conductance_mode = False
        self.original_current = None

        # Temporary storage for operations
        self.temp_data = {
            'time': None,
            'voltage': None,
            'current': None,
            'conductance': None
        }

        # Update display settings
        self.update_display_settings()

        # Store previous settings for undo
        self.previous_settings = None

        # PSD data storage
        self.psd_data = []

        # Initialize PSD analysis
        self.psd_analysis = PSDAnalysis()

        # Filter history
        self.filter_history = []

        # Fitting history
        self.fitting_history = []

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left side: File tree and tabs
        left_widget = self.create_left_panel()
        splitter.addWidget(left_widget)

        # Right side: Plots
        right_widget = self.create_right_panel()
        splitter.addWidget(right_widget)

        # Set initial sizes for splitter
        splitter.setSizes([400, 1200])

        if file_list:
            self.update_file_list(file_list)
        
    def create_light_palette(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 163, 224))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        return palette

    
    def update_display_settings(self):
        self.show_settings = {
            'x_start': None,
            'x_end': None,
            'y_start': None,
            'y_end': None,
            'voltage_color': '#0C5DA5',  # Dark blue
            'current_color': '#00B945',  # Green
            'font_family': 'Arial',
            'font_size': 16,
            'font_size_title': 12,
            'font_size_label': 10,
            'font_size_tick': 8,
            'font_weight_title': 'bold',
            'font_weight_label': 'bold',
            'font_weight_tick': 'normal'
        }


    def create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # File tree
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(['Experiment'])
        self.file_tree.itemDoubleClicked.connect(self.plot_file)
        left_layout.addWidget(self.file_tree)

        # Tab widget for operations
        self.tab_widget = QTabWidget()
        left_layout.addWidget(self.tab_widget)

        # Create operation tabs
        self.create_trim_tab()
        self.create_cut_tab()
        self.create_psd_tab()
        self.create_filter_tab()
        self.create_fitting_tab()
        self.create_conductance_tab()
        self.create_plot_tab() 

        return left_panel
    
    def create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create plot widgets
        self.create_plot_widgets()
        right_layout.addWidget(self.right_splitter)

        return right_panel


    def create_plot_widgets(self):
        """Create and setup plot widgets with consistent style"""
        self.setup_plot_style()
        
        # Create right-side vertical splitter
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Main plot (Voltage and Current/Conductance)
        main_plot_widget = QWidget()
        main_plot_layout = QVBoxLayout(main_plot_widget)
        
        # Create figure with specific size ratio
        self.figure_main = Figure(figsize=(8, 6))
        
        # Create subplot grid with specific height ratios
        gs_main = self.figure_main.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.1)
        self.ax_voltage = self.figure_main.add_subplot(gs_main[0])
        self.ax_current = self.figure_main.add_subplot(gs_main[1], sharex=self.ax_voltage)
        
        # Set up axes
        self.display_settings.setup_axis(self.ax_voltage, ylabel='Voltage (V)')
        self.display_settings.setup_axis(self.ax_current, xlabel='Time (s)', 
                                    ylabel='Current (nA)')
        
        # Hide x-axis labels for voltage plot
        self.ax_voltage.tick_params(labelbottom=False)
        
        self.canvas_main = FigureCanvas(self.figure_main)
        self.toolbar_main = NavigationToolbar(self.canvas_main, self)
        
        main_plot_layout.addWidget(self.toolbar_main)
        main_plot_layout.addWidget(self.canvas_main)
        
        # PSD plot
        psd_plot_widget = QWidget()
        psd_plot_layout = QVBoxLayout(psd_plot_widget)
        
        self.figure_psd = Figure(figsize=(8, 3))
        self.ax_psd = self.figure_psd.add_subplot(111)
        
        # Set up PSD axis
        self.display_settings.setup_axis(self.ax_psd, xlabel='Frequency (Hz)',
                                    ylabel='PSD (nA²/Hz)')
        
        self.canvas_psd = FigureCanvas(self.figure_psd)
        self.toolbar_psd = NavigationToolbar(self.canvas_psd, self)
        
        psd_plot_layout.addWidget(self.toolbar_psd)
        psd_plot_layout.addWidget(self.canvas_psd)
        
        # Add widgets to splitter
        self.right_splitter.addWidget(main_plot_widget)
        self.right_splitter.addWidget(psd_plot_widget)
        
        # Set initial splitter sizes (70% main plot, 30% PSD)
        self.right_splitter.setSizes([700, 300])

    def setup_plot_style(self):
        """Initialize and setup plot style"""
        self.display_settings = DisplaySettings()
        self.display_settings.apply_rcparams()

    def setup_axes(self):
        for ax in [self.ax_voltage, self.ax_current, self.ax_psd]:
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            
            try:
                # Try to use AutoMinorLocator
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            except AttributeError:
                # Fallback to a simple minor locator if AutoMinorLocator is not available
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=0.2))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.2))
                logging.warning("AutoMinorLocator not available, using MultipleLocator as fallback.")

        self.ax_voltage.tick_params(labelbottom=False)
        self.ax_current.tick_params(labelbottom=True)

        # Adjust spacing between subplots
        self.figure_main.subplots_adjust(hspace=0.1)
        self.figure_psd.subplots_adjust(bottom=0.15, top=0.95)

##########################更新文件列表###########################

    def update_file_list(self, file_list):
        try:
            logging.info(f"Updating file list in PreprocessWindow with {len(file_list)} items")
            self.file_tree.clear()

            for item in file_list:
                if len(item) != 4:
                    logging.warning(f"Skipping invalid file list item: {item}")
                    continue

                experiment, _, file, full_path = item

                experiment_items = self.file_tree.findItems(experiment, Qt.MatchFlag.MatchExactly, 0)
                if experiment_items:
                    experiment_item = experiment_items[0]
                else:
                    experiment_item = QTreeWidgetItem(self.file_tree, [experiment])

                file_item = QTreeWidgetItem(experiment_item, [file])
                file_item.setToolTip(0, full_path)

            self.file_tree.expandAll()
            logging.info(f"File list updated successfully")
        except Exception as e:
            logging.error(f"Error updating file list in PreprocessWindow: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "Warning", f"Failed to update file list: {str(e)}")

#######################画图###############################################

    def plot_file(self, item, column):
        file_path = item.toolTip(0)
        try:
            logging.info(f"Attempting to plot file: {file_path}")

            # Check file extension
            _, ext = os.path.splitext(file_path)

            if ext.lower() == '.h5':
                # Load data from HDF5 file
                with h5py.File(file_path, 'r') as hf:
                    self.data['time'] = hf['time'][:]
                    self.data['voltage'] = hf['voltage'][:]
                    self.data['current'] = hf['current'][:]
                    if 'conductance' in hf:
                        self.data['conductance'] = hf['conductance'][:]
                        self.is_conductance_mode = True
                    else:
                        self.data['conductance'] = None
                        self.is_conductance_mode = False
            elif ext.lower() == '.tdms':
                # Use MainWindow's read_tdms_file method
                self.main_window.read_tdms_file(file_path)

                # Access data from MainWindow
                self.data['time'] = self.main_window.current_time_voltage
                self.data['voltage'] = self.main_window.current_voltage
                self.data['current'] = self.main_window.current_current
                self.data['conductance'] = None
                self.is_conductance_mode = False
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            # Reset conductance related states
            self.reset_conductance_state()

            # Validate data
            for key in ['time', 'voltage', 'current']:
                if self.data[key] is None or len(self.data[key]) == 0:
                    raise ValueError(f"Invalid or empty {key} data")

            # Remove any NaN or inf values
            mask = np.isfinite(self.data['time']) & np.isfinite(self.data['voltage']) & np.isfinite(self.data['current'])
            for key in ['time', 'voltage', 'current']:
                self.data[key] = self.data[key][mask]

            if len(self.data['time']) == 0:
                raise ValueError("No valid data points after removing NaN and inf values")

            # Clear existing plots
            self.ax_voltage.clear()
            self.ax_current.clear()
            self.ax_psd.clear()

            # Plot voltage data
            self.ax_voltage.plot(self.data['time'], self.data['voltage'], 
                                color=self.show_settings['voltage_color'], 
                                linewidth=1.0)
            self.ax_voltage.set_ylabel('Voltage (V)', 
                                    fontsize=self.show_settings['font_size_label'], 
                                    fontweight=self.show_settings['font_weight_label'])

            # Expand y-axis range for voltage plot
            v_min, v_max = np.min(self.data['voltage']), np.max(self.data['voltage'])
            v_range = v_max - v_min
            self.ax_voltage.set_ylim(v_min - 0.1 * v_range, v_max + 0.1 * v_range)

            # Plot current or conductance data
            if self.is_conductance_mode and self.data['conductance'] is not None:
                y_data = self.data['conductance']
                ylabel = 'Conductance (nS)'
            else:
                y_data = self.data['current']
                ylabel = 'Current (nA)'

            self.ax_current.plot(self.data['time'], y_data, 
                                color=self.show_settings['current_color'], 
                                linewidth=1.0)
            self.ax_current.set_ylabel(ylabel, 
                                    fontsize=self.show_settings['font_size_label'], 
                                    fontweight=self.show_settings['font_weight_label'])
            self.ax_current.set_xlabel('Time (s)', 
                                    fontsize=self.show_settings['font_size_label'], 
                                    fontweight=self.show_settings['font_weight_label'])

            # Expand y-axis range for current/conductance plot
            y_min, y_max = np.min(y_data), np.max(y_data)
            y_range = y_max - y_min
            self.ax_current.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            self.ax_voltage.set_xlim(self.data['time'][0], self.data['time'][-1])
            self.ax_current.set_xlim(self.data['time'][0], self.data['time'][-1])

            self.setup_axes()

            # Update PSD plot
            self.update_psd_plot()

            # Refresh canvases
            self.figure_main.tight_layout()
            self.canvas_main.draw()
            self.figure_psd.tight_layout()
            self.canvas_psd.draw()

            logging.info(f"File plotted successfully: {file_path}")

        except ValueError as e:
            error_msg = f"Invalid data or unsupported file format: {str(e)}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.warning(self, "Warning", error_msg)
        except Exception as e:
            error_msg = f"Unexpected error while plotting file: {str(e)}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.warning(self, "Error", error_msg)

    def plot_data(self):
        logging.info(f"Plotting data. Current conductance mode: {self.is_conductance_mode}")
        self.ax_voltage.clear()
        self.ax_current.clear()

        if self.data['time'] is None or self.data['voltage'] is None or (
                self.data['current'] is None and self.data['conductance'] is None):
            logging.warning("No data available to plot")
            return

        time = self.data['time']
        voltage = self.data['voltage']

        self.ax_voltage.plot(time, voltage, color=self.show_settings['voltage_color'], linewidth=1.0)
        self.ax_voltage.set_ylabel('Voltage (V)', fontweight=self.show_settings['font_weight_label'],
                                   fontsize=self.show_settings['font_size_label'])

        # Expand y-axis range for voltage plot
        v_min, v_max = np.min(voltage), np.max(voltage)
        v_range = v_max - v_min
        self.ax_voltage.set_ylim(v_min - 0.1 * v_range, v_max + 0.1 * v_range)

        if self.is_conductance_mode and self.data['conductance'] is not None:
            y_data = self.data['conductance']
            ylabel = 'Conductance (nS)'
            logging.info("Plotting conductance data")
        else:
            y_data = self.data['current']
            ylabel = 'Current (nA)'
            self.is_conductance_mode = False  # Ensure mode is correct if conductance data is not available
            logging.info("Plotting current data")

        self.ax_current.plot(time, y_data, color=self.show_settings['current_color'], linewidth=1.0)
        self.ax_current.set_ylabel(ylabel, fontweight=self.show_settings['font_weight_label'],
                                   fontsize=self.show_settings['font_size_label'])
        self.ax_current.set_xlabel('Time (s)', fontweight=self.show_settings['font_weight_label'],
                                   fontsize=self.show_settings['font_size_label'])

        # Expand y-axis range for current/conductance plot
        y_min, y_max = np.min(y_data), np.max(y_data)
        y_range = y_max - y_min
        self.ax_current.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        self.ax_voltage.set_xlim(time[0], time[-1])
        self.ax_current.set_xlim(time[0], time[-1])

        self.setup_axes()

        self.canvas_main.draw()

        logging.info(f"Data plotted successfully. Final conductance mode: {self.is_conductance_mode}")

#######################################Trim########################################

    def create_trim_tab(self):
        trim_tab = QWidget()
        trim_layout = QVBoxLayout(trim_tab)

        # Z-score and Trim Range
        settings_group = QGroupBox("Trim Settings")
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("Z-score:"), 0, 0)
        self.trim_threshold = QLineEdit("0.5")
        settings_layout.addWidget(self.trim_threshold, 0, 1)

        settings_layout.addWidget(QLabel("Trim Range (s):"), 1, 0)
        settings_layout.addWidget(QLabel("Start:"), 1, 1)
        self.trim_start = QLineEdit()
        settings_layout.addWidget(self.trim_start, 1, 2)
        settings_layout.addWidget(QLabel("End:"), 1, 3)
        self.trim_end = QLineEdit()
        settings_layout.addWidget(self.trim_end, 1, 4)

        settings_group.setLayout(settings_layout)
        trim_layout.addWidget(settings_group)

        # Buttons
        button_group = QGroupBox("Actions")
        button_layout = QHBoxLayout()

        self.trim_apply = QPushButton("Apply")
        self.trim_undo = QPushButton("Undo")
        self.trim_accept = QPushButton("Accept")
        self.trim_save = QPushButton("Save")

        self.trim_apply.clicked.connect(self.apply_trim)
        self.trim_undo.clicked.connect(self.undo_trim)
        self.trim_accept.clicked.connect(self.accept_trim)
        self.trim_save.clicked.connect(self.save_trim)

        button_layout.addWidget(self.trim_apply)
        button_layout.addWidget(self.trim_undo)
        button_layout.addWidget(self.trim_accept)
        button_layout.addWidget(self.trim_save)

        button_group.setLayout(button_layout)
        trim_layout.addWidget(button_group)

        self.tab_widget.addTab(trim_tab, "Trim")

    def apply_trim(self):
        try:
            # 1. 确保temp_data是data的最新副本
            for key in self.data:
                self.temp_data[key] = self.data[key].copy() if self.data[key] is not None else None

            if self.temp_data['time'] is None or (self.temp_data['current'] is None and self.temp_data['conductance'] is None):
                raise ValueError("No data available for trimming")

            threshold = float(self.trim_threshold.text())
            start = float(self.trim_start.text()) if self.trim_start.text() else self.temp_data['time'][0]
            end = float(self.trim_end.text()) if self.trim_end.text() else self.temp_data['time'][-1]

            # 2. 创建mask来选择指定范围内的数据
            mask = (self.temp_data['time'] >= start) & (self.temp_data['time'] <= end)
            
            # 3. 对指定范围内的数据进行trim操作
            selected_time = self.temp_data['time'][mask]
            if self.is_conductance_mode:
                selected_data = self.temp_data['conductance'][mask]
            else:
                selected_data = self.temp_data['current'][mask]

            cleaned_time, cleaned_data = self.remove_outliers(selected_data, selected_time, threshold)

            if len(cleaned_time) == 0:
                raise ValueError("All data points were removed as outliers. Try adjusting the threshold.")

            # 4. 使用插值将清理后的数据调整到与原始选择区域相同的长度
            interp_cleaned_data = np.interp(selected_time, cleaned_time, cleaned_data)

            # 5. 更新temp_data，只替换指定范围内的数据
            for key in self.temp_data:
                if self.temp_data[key] is not None:
                    if key == 'time':
                        continue  # 保持时间数据不变
                    elif (key == 'conductance' and self.is_conductance_mode) or (key == 'current' and not self.is_conductance_mode):
                        self.temp_data[key][mask] = interp_cleaned_data
                    else:
                        # 对于其他数据（如电压），我们保持原样
                        pass

            # 6. 更新图表以显示处理结果
            self.update_trim_plot()

            QMessageBox.information(self, "Success", "Trim applied successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply trim: {str(e)}")
            logging.error(f"Error in apply_trim: {str(e)}", exc_info=True)

    def remove_outliers(self, data, time, threshold=3.5):
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores < threshold

        # 确保至少保留一些数据点
        if np.sum(outlier_mask) < len(data) * 0.1:  # 保留至少10%的数据点
            outlier_mask = z_scores < np.percentile(z_scores, 90)

        cleaned_data = data[outlier_mask]
        cleaned_time = time[outlier_mask]

        print(f"原始数据点: {len(data)}")
        print(f"清理后数据点: {len(cleaned_data)}")
        print(f"移除了 {len(data) - len(cleaned_data)} 个异常值")

        return cleaned_time, cleaned_data

    def undo_trim(self):
    # Reset temp_data to original data
        for key in self.data:
            self.temp_data[key] = self.data[key].copy() if self.data[key] is not None else None
        self.update_trim_plot()

    def accept_trim(self):
        # 将 temp_data 中的处理结果复制到 data
        for key in self.temp_data:
            if self.temp_data[key] is not None:
                self.data[key] = self.temp_data[key].copy()

        # 更新主图表
        self.plot_data()

        QMessageBox.information(self, "Success", "Trim changes have been accepted and applied.")

    def update_trim_plot(self):
        # Clear current plots
        self.ax_voltage.clear()
        self.ax_current.clear()

        # Plot voltage data
        self.ax_voltage.plot(self.data['time'], self.data['voltage'], color='blue', alpha=0.5, label='Original')
        self.ax_voltage.plot(self.temp_data['time'], self.temp_data['voltage'], color='red', label='Trimmed')
        self.ax_voltage.set_ylabel('Voltage (V)')

        # Plot original and trimmed data
        if self.is_conductance_mode:
            self.ax_current.plot(self.data['time'], self.data['conductance'], color='blue', alpha=0.5,
                                label='Original Data')
            self.ax_current.plot(self.temp_data['time'], self.temp_data['conductance'], color='red',
                                label='Trimmed Data')
            self.ax_current.set_ylabel('Conductance (nS)')
        else:
            self.ax_current.plot(self.data['time'], self.data['current'], color='blue', alpha=0.5,
                                label='Original Data')
            self.ax_current.plot(self.temp_data['time'], self.temp_data['current'], color='red',
                                label='Trimmed Data')
            self.ax_current.set_ylabel('Current (nA)')

        self.ax_current.set_xlabel('Time (s)')
        self.ax_current.legend()

        # Set x-axis limits to match data range
        x_min, x_max = self.data['time'][0], self.data['time'][-1]
        self.ax_voltage.set_xlim(x_min, x_max)
        self.ax_current.set_xlim(x_min, x_max)

        # Ensure Voltage and Current plots don't have grid lines
        self.ax_voltage.grid(False)
        self.ax_current.grid(False)

        # Use tight_layout to automatically adjust the layout
        self.figure_main.tight_layout()

        # Refresh main canvas
        self.canvas_main.draw()

    def save_trim(self):
        try:
            if self.data['time'] is None or (self.data['current'] is None and self.data['conductance'] is None):
                raise ValueError("No data available to save")

            current_item = self.file_tree.currentItem()
            if current_item is None:
                raise ValueError("No file selected")

            full_path = current_item.toolTip(0)
            if not full_path:
                raise ValueError("Unable to get file path")

            original_dir = os.path.dirname(full_path)
            file_name = os.path.basename(full_path)

            processing_dir = os.path.join(original_dir, 'ProcessingData')
            if not os.path.exists(processing_dir):
                os.makedirs(processing_dir)

            index = 1
            while True:
                new_file_name = f"{os.path.splitext(file_name)[0]}_Trimmed_{index}.h5"
                new_full_path = os.path.join(processing_dir, new_file_name)
                if not os.path.exists(new_full_path):
                    break
                index += 1

            with h5py.File(new_full_path, 'w') as hf:
                for key in self.data:
                    if self.data[key] is not None:
                        hf.create_dataset(key, data=self.data[key], compression="gzip", compression_opts=9)

            QMessageBox.information(self, "Success", f"Data saved as {new_file_name}\nFull path: {new_full_path}")

            logging.info(f"Data saved to: {new_full_path}")

        except Exception as e:
            error_msg = f"Error saving data: {str(e)}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)



##################################Cut##############################################

    def create_cut_tab(self):
        cut_tab = QWidget()
        cut_layout = QVBoxLayout(cut_tab)

        # Cut Range and Y Range
        range_group = QGroupBox("Range Settings")
        range_layout = QGridLayout()

        range_layout.addWidget(QLabel("Cut Range (s):"), 0, 0)
        range_layout.addWidget(QLabel("Start:"), 0, 1)
        self.cut_start = QLineEdit()
        range_layout.addWidget(self.cut_start, 0, 2)
        range_layout.addWidget(QLabel("End:"), 0, 3)
        self.cut_end = QLineEdit()
        range_layout.addWidget(self.cut_end, 0, 4)

        range_layout.addWidget(QLabel("Y Range:"), 1, 0)
        range_layout.addWidget(QLabel("Start:"), 1, 1)
        self.y_start = QLineEdit()
        range_layout.addWidget(self.y_start, 1, 2)
        range_layout.addWidget(QLabel("End:"), 1, 3)
        self.y_end = QLineEdit()
        range_layout.addWidget(self.y_end, 1, 4)

        range_group.setLayout(range_layout)
        cut_layout.addWidget(range_group)

        # Display Settings
        display_group = QGroupBox("Display Settings")
        display_layout = QGridLayout()

        display_layout.addWidget(QLabel("Voltage Color:"), 0, 0)
        self.voltage_color_button = QPushButton()
        self.voltage_color_button.setStyleSheet(f"background-color: {self.show_settings['voltage_color']};")
        self.voltage_color_button.clicked.connect(lambda: self.choose_color('voltage_color'))
        display_layout.addWidget(self.voltage_color_button, 0, 1)

        display_layout.addWidget(QLabel("Current/Conductance Color:"), 1, 0)
        self.current_color_button = QPushButton()
        self.current_color_button.setStyleSheet(f"background-color: {self.show_settings['current_color']};")
        self.current_color_button.clicked.connect(lambda: self.choose_color('current_color'))
        display_layout.addWidget(self.current_color_button, 1, 1)

        display_layout.addWidget(QLabel("Font:"), 2, 0)
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(self.show_settings['font_family']))
        self.font_combo.currentFontChanged.connect(self.update_font)
        display_layout.addWidget(self.font_combo, 2, 1)

        display_layout.addWidget(QLabel("Font Size:"), 3, 0)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 24)
        self.font_size_spin.setValue(self.show_settings['font_size'])
        self.font_size_spin.valueChanged.connect(self.update_font_size)
        display_layout.addWidget(self.font_size_spin, 3, 1)

        self.bold_checkbox = QCheckBox("Bold Font")
        self.bold_checkbox.setChecked(self.show_settings['font_weight_label'] == 'bold')
        self.bold_checkbox.stateChanged.connect(self.update_font_weight)
        display_layout.addWidget(self.bold_checkbox, 4, 0, 1, 2)

        display_group.setLayout(display_layout)
        cut_layout.addWidget(display_group)

        # Buttons
        button_group = QGroupBox("Actions")
        button_layout = QHBoxLayout()
        self.cut_apply = QPushButton("Apply")
        self.cut_undo = QPushButton("Undo")
        self.cut_accept = QPushButton("Accept")
        self.cut_save = QPushButton("Save")

        self.cut_apply.clicked.connect(self.apply_cut)
        self.cut_undo.clicked.connect(self.undo_cut)
        self.cut_accept.clicked.connect(self.accept_cut)
        self.cut_save.clicked.connect(self.save_cut)

        button_layout.addWidget(self.cut_apply)
        button_layout.addWidget(self.cut_undo)
        button_layout.addWidget(self.cut_accept)
        button_layout.addWidget(self.cut_save)

        button_group.setLayout(button_layout)
        cut_layout.addWidget(button_group)

        self.tab_widget.addTab(cut_tab, "Cut")

    def choose_color(self, color_key):
        color = QColorDialog.getColor()
        if color.isValid():
            self.show_settings[color_key] = color.name()
            if color_key == 'voltage_color':
                self.voltage_color_button.setStyleSheet(f"background-color: {color.name()};")
            elif color_key == 'current_color':
                self.current_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.update_plot()

    def update_font(self, font):
        self.show_settings['font_family'] = font.family()
        self.update_plot()

    def update_font_size(self, size):
        self.show_settings['font_size'] = size
        self.show_settings['font_size_label'] = size
        self.show_settings['font_size_tick'] = size - 2
        self.update_plot()

    def update_font_weight(self, state):
        self.show_settings['font_weight_label'] = 'bold' if state == Qt.CheckState.Checked else 'normal'
        self.update_plot()

    def update_plot(self):
        self.setup_plot_style()
        self.plot_data()
        self.update_psd_plot()
        
        # 添加这些行来调整子图之间的间距
        self.figure_main.subplots_adjust(hspace=0.05, top=0.95, bottom=0.1, left=0.1, right=0.95)
        self.figure_psd.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.95)

    def apply_cut(self):
        if self.data['time'] is None or (self.data['current'] is None and self.data['conductance'] is None):
            QMessageBox.warning(self, "Warning", "No data available for cutting")
            return

        try:
            start = float(self.cut_start.text()) if self.cut_start.text() else self.data['time'][0]
            end = float(self.cut_end.text()) if self.cut_end.text() else self.data['time'][-1]

            mask = (self.data['time'] >= start) & (self.data['time'] <= end)

            # Store original data in temp_data for undo
            for key in self.data:
                self.temp_data[key] = self.data[key][mask] if self.data[key] is not None else None

            # Apply Y range if specified
            y_start = float(self.y_start.text()) if self.y_start.text() else None
            y_end = float(self.y_end.text()) if self.y_end.text() else None

            self.update_cut_plot(y_start, y_end)
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {str(e)}")

    def undo_cut(self):
        # Restore original data from temp_data
        for key in self.temp_data:
            if self.temp_data[key] is not None:
                self.temp_data[key] = self.data[key].copy()

        self.update_cut_plot()

    def accept_cut(self):
        # Move temp_data to central data storage
        for key in self.temp_data:
            if self.temp_data[key] is not None:
                self.data[key] = self.temp_data[key].copy()

        # Reset Y range settings
        self.y_start.clear()
        self.y_end.clear()

        # Update the main plot
        self.plot_data()

        # Update PSD plot if necessary
        self.update_psd_plot()

        # Inform the user
        QMessageBox.information(self, "Success", "Cut changes have been accepted and applied.")

    def update_cut_plot(self, y_start=None, y_end=None):
        # Clear current plots
        self.ax_voltage.clear()
        self.ax_current.clear()

        # Plot voltage data
        self.ax_voltage.plot(self.temp_data['time'], self.temp_data['voltage'], 
                             color=self.show_settings['voltage_color'])
        self.ax_voltage.set_ylabel('Voltage (V)', 
                                   fontsize=self.show_settings['font_size_label'],
                                   fontweight=self.show_settings['font_weight_label'])

        # Plot current or conductance data
        if self.is_conductance_mode:
            self.ax_current.plot(self.temp_data['time'], self.temp_data['conductance'], 
                                 color=self.show_settings['current_color'])
            self.ax_current.set_ylabel('Conductance (nS)', 
                                       fontsize=self.show_settings['font_size_label'],
                                       fontweight=self.show_settings['font_weight_label'])
        else:
            self.ax_current.plot(self.temp_data['time'], self.temp_data['current'], 
                                 color=self.show_settings['current_color'])
            self.ax_current.set_ylabel('Current (nA)', 
                                       fontsize=self.show_settings['font_size_label'],
                                       fontweight=self.show_settings['font_weight_label'])

        self.ax_current.set_xlabel('Time (s)', 
                                   fontsize=self.show_settings['font_size_label'],
                                   fontweight=self.show_settings['font_weight_label'])

        # Set x-axis limits to match data range
        x_min, x_max = self.temp_data['time'][0], self.temp_data['time'][-1]
        self.ax_voltage.set_xlim(x_min, x_max)
        self.ax_current.set_xlim(x_min, x_max)

        # Apply Y range if specified
        if y_start is not None and y_end is not None:
            self.ax_current.set_ylim(y_start, y_end)

        # Ensure Voltage and Current plots don't have grid lines
        self.ax_voltage.grid(False)
        self.ax_current.grid(False)

        # Apply axis settings
        self.setup_axes()

        # Refresh main canvas
        self.canvas_main.draw()

    def save_cut(self):
        try:
            if self.data['time'] is None or self.data['voltage'] is None or self.data['current'] is None:
                raise ValueError("No data available to save")

            current_item = self.file_tree.currentItem()
            if current_item is None:
                raise ValueError("No file selected")

            full_path = current_item.toolTip(0)
            if not full_path:
                raise ValueError("Unable to get file path")

            original_dir = os.path.dirname(full_path)
            file_name = os.path.basename(full_path)

            processing_dir = os.path.join(original_dir, 'ProcessingData')
            if not os.path.exists(processing_dir):
                os.makedirs(processing_dir)

            index = 1
            while True:
                new_file_name = f"{os.path.splitext(file_name)[0]}_Cut_{index}.h5"
                new_full_path = os.path.join(processing_dir, new_file_name)
                if not os.path.exists(new_full_path):
                    break
                index += 1

            with h5py.File(new_full_path, 'w') as hf:
                for key in self.data:
                    if self.data[key] is not None:
                        hf.create_dataset(key, data=self.data[key], compression="gzip", compression_opts=9)

                # Save display settings
                settings_group = hf.create_group('display_settings')
                settings_group.attrs['voltage_color'] = self.voltage_color.currentText()
                settings_group.attrs['current_color'] = self.current_color.currentText()
                settings_group.attrs['font_size'] = self.label_font_size.value()
                settings_group.attrs['font_bold'] = self.label_font_bold.isChecked()
                if self.y_start.text() and self.y_end.text():
                    settings_group.attrs['y_start'] = float(self.y_start.text())
                    settings_group.attrs['y_end'] = float(self.y_end.text())

            QMessageBox.information(self, "Success", f"Data saved as {new_file_name}\nFull path: {new_full_path}")

        except Exception as e:
            error_msg = f"Error saving data: {str(e)}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)

######################################PSD##############################################

    def create_psd_tab(self):
        psd_tab = QWidget()
        main_layout = QVBoxLayout(psd_tab)

        # Settings group
        settings_group = QGroupBox("PSD Settings")
        settings_layout = QGridLayout()

        # Range inputs
        settings_layout.addWidget(QLabel("Time Range (s):"), 0, 0)
        self.psd_start_input = QLineEdit()
        settings_layout.addWidget(self.psd_start_input, 0, 1)
        settings_layout.addWidget(QLabel("-"), 0, 2)
        self.psd_end_input = QLineEdit()
        settings_layout.addWidget(self.psd_end_input, 0, 3)

        # Cutoff frequency input
        settings_layout.addWidget(QLabel("Cutoff Frequency (Hz):"), 1, 0)
        self.psd_cutoff_input = QLineEdit()
        self.psd_cutoff_input.setText("5000")  # Default value
        settings_layout.addWidget(self.psd_cutoff_input, 1, 1, 1, 3)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # X-axis scale selection
        settings_layout.addWidget(QLabel("Frequency Scale:"), 2, 0)
        self.psd_xscale_combo = QComboBox()
        self.psd_xscale_combo.addItems(["Linear", "Log"])
        self.psd_xscale_combo.setCurrentText("Log")  # Default to log scale
        self.psd_xscale_combo.currentTextChanged.connect(self.update_psd_plot)
        settings_layout.addWidget(self.psd_xscale_combo, 2, 1, 1, 3)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # Metrics display group
        metrics_group = QGroupBox("PSD Metrics")
        metrics_layout = QVBoxLayout()
        self.metrics_display = QTextEdit()
        self.metrics_display.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_display)
        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

        # Buttons group
        button_group = QGroupBox("Actions")
        button_layout = QHBoxLayout()

        self.psd_apply_button = QPushButton("Apply")
        self.psd_undo_button = QPushButton("Undo")
        self.psd_delete_button = QPushButton("Delete")
        self.psd_save_button = QPushButton("Save")

        self.psd_apply_button.clicked.connect(self.apply_psd)
        self.psd_undo_button.clicked.connect(self.undo_psd)
        self.psd_delete_button.clicked.connect(self.delete_psd)
        self.psd_save_button.clicked.connect(self.save_psd)

        button_layout.addWidget(self.psd_apply_button)
        button_layout.addWidget(self.psd_undo_button)
        button_layout.addWidget(self.psd_delete_button)
        button_layout.addWidget(self.psd_save_button)

        button_group.setLayout(button_layout)
        main_layout.addWidget(button_group)

        # PSD list
        list_group = QGroupBox("PSD List")
        list_layout = QVBoxLayout()
        self.psd_list = QListWidget()
        self.psd_list.itemClicked.connect(self.display_psd_info)  # Optional: add method to display PSD info
        list_layout.addWidget(self.psd_list)
        list_group.setLayout(list_layout)
        main_layout.addWidget(list_group)

        self.tab_widget.addTab(psd_tab, "PSD")
    
    def display_psd_info(self, item):
        """显示选中的PSD信息"""
        try:
            # 找到对应的PSD数据
            psd_name = item.text()
            for name, frequencies, psd in self.psd_data:
                if name == psd_name:
                    # 计算并显示指标
                    metrics = self.psd_analysis.calculate_psd_metrics(frequencies, psd)  # 使用 psd_analysis
                    metrics_text = f"PSD Information for {psd_name}:\n\n"
                    for key, value in metrics.items():
                        metrics_text += f"{key}: {value:.2e}\n"
                    self.metrics_display.setText(metrics_text)
                    break
        except Exception as e:
            logging.error(f"Error displaying PSD info: {str(e)}")

    def apply_psd(self):
        """Apply PSD calculation to selected data"""
        try:
            if self.data['time'] is None or (self.data['current'] is None and self.data['conductance'] is None):
                raise ValueError("No data available for PSD calculation")
            
            # Get time range
            time_range = self.ax_current.get_xlim()
            start = float(self.psd_start_input.text()) if self.psd_start_input.text() else time_range[0]
            end = float(self.psd_end_input.text()) if self.psd_end_input.text() else time_range[1]
            
            # Select data range
            mask = (self.data['time'] >= start) & (self.data['time'] <= end)
            time_data = self.data['time'][mask]
            signal_data = self.data['conductance'][mask] if self.is_conductance_mode else self.data['current'][mask]
            
            # Calculate PSD using the analyzer
            frequencies, psd = self.psd_analysis.calculate_psd(time_data, signal_data)  # 使用 psd_analysis
            
            # Get cutoff frequency if specified
            cutoff = float(self.psd_cutoff_input.text()) if self.psd_cutoff_input.text() else None
            if cutoff is not None:
                frequencies, psd = self.psd_analysis.apply_frequency_filter(frequencies, psd, f_max=cutoff)  # 使用 psd_analysis
            
            # Calculate metrics
            metrics = self.psd_analysis.calculate_psd_metrics(frequencies, psd)  # 使用 psd_analysis
            
            # Store PSD data
            file_name = os.path.splitext(self.file_tree.currentItem().text(0))[0]
            psd_name = f"{file_name}_PSD_{start:.2f}-{end:.2f}"
            self.psd_data.append((psd_name, frequencies, psd))
            
            # Add to list widget
            item = QListWidgetItem(psd_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.psd_list.addItem(item)
            
            # Update metrics display
            metrics_text = "PSD Metrics:\n\n"
            for key, value in metrics.items():
                metrics_text += f"{key}: {value:.2e}\n"
            self.metrics_display.setText(metrics_text)
            
            # Update plot
            self.update_psd_plot()
            
            QMessageBox.information(self, "Success", "PSD calculation completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate PSD: {str(e)}")
            logging.error(f"PSD calculation error: {str(e)}", exc_info=True)


    def update_psd_plot(self):
        """Update PSD plot with scale selection"""
        try:
            self.ax_psd.clear()
            
            if not self.psd_data:
                self.ax_psd.set_title("No PSD data available", 
                                    fontsize=self.display_settings.fonts['sizes']['title'])
                self.canvas_psd.draw()
                return
            
            colors = [self.display_settings.colors['voltage'],
                    self.display_settings.colors['current'],
                    self.display_settings.colors['conductance']]
            
            # 设置 x 轴比例
            xscale = self.psd_xscale_combo.currentText().lower()
            self.ax_psd.set_xscale(xscale)
            
            for (name, frequencies, psd), color in zip(self.psd_data, colors):
                if len(frequencies) != len(psd):
                    continue
                    
                # Remove any non-finite values
                mask = np.isfinite(frequencies) & np.isfinite(psd)
                frequencies = frequencies[mask]
                psd = psd[mask]
                
                if len(frequencies) == 0:
                    continue
                    
                try:
                    # Always use semilogy for y-axis (PSD values)
                    if xscale == 'linear':
                        self.ax_psd.semilogy(frequencies, psd, color=color, 
                                        linewidth=self.display_settings.lines['widths']['data'],
                                        label=name)
                    else:  # log scale
                        self.ax_psd.loglog(frequencies, psd, color=color, 
                                        linewidth=self.display_settings.lines['widths']['data'],
                                        label=name)
                except ValueError as e:
                    logging.error(f"Error plotting PSD for {name}: {str(e)}")
                    continue
            
            # Configure axis
            self.ax_psd.set_xlabel('Frequency (Hz)', 
                                fontsize=self.display_settings.fonts['sizes']['label'],
                                fontweight=self.display_settings.fonts['weights']['label'])
            self.ax_psd.set_ylabel('PSD (nA²/Hz)', 
                                fontsize=self.display_settings.fonts['sizes']['label'],
                                fontweight=self.display_settings.fonts['weights']['label'])
            
            # Add legend if there's more than one plot
            if len(self.psd_data) > 1:
                self.ax_psd.legend(loc='upper right', 
                                fontsize=self.display_settings.fonts['sizes']['legend'])
            
            # Set reasonable y-axis limits
            if self.psd_data:
                psd_min = np.inf
                psd_max = -np.inf
                for _, _, psd_data in self.psd_data:
                    valid_psd = psd_data[psd_data > 0]  # Only consider positive values
                    if len(valid_psd) > 0:
                        psd_min = min(psd_min, np.min(valid_psd))
                        psd_max = max(psd_max, np.max(valid_psd))
                
                if psd_min < psd_max:
                    self.ax_psd.set_ylim(psd_min / 10, psd_max * 10)
            
            # Set x-axis limits based on scale
            if xscale == 'log':
                self.ax_psd.set_xlim(left=frequencies[0])  # Start from first non-zero frequency
            else:
                self.ax_psd.set_xlim(left=0)  # Start from zero for linear scale
            
            # Apply grid with scale-appropriate style
            if xscale == 'log':
                self.ax_psd.grid(True, which='both', linestyle=':', alpha=0.3)
            else:
                self.ax_psd.grid(True, which='major', linestyle='-', alpha=0.3)
                self.ax_psd.grid(True, which='minor', linestyle=':', alpha=0.15)
            
            self.figure_psd.tight_layout()
            self.canvas_psd.draw()
            
        except Exception as e:
            logging.error(f"Error in update_psd_plot: {str(e)}", exc_info=True)

    def undo_psd(self):
        if self.psd_data:
            self.psd_data.pop()
            self.psd_list.takeItem(self.psd_list.count() - 1)
            self.update_psd_plot()

    def delete_psd(self):
        selected_items = self.psd_list.selectedItems()
        for item in reversed(selected_items):
            index = self.psd_list.row(item)
            self.psd_list.takeItem(index)
            if 0 <= index < len(self.psd_data):
                del self.psd_data[index]
        self.update_psd_plot()

    def save_psd(self):
        checked_items = [self.psd_list.item(i) for i in range(self.psd_list.count())
                         if self.psd_list.item(i).checkState() == Qt.CheckState.Checked]

        if not checked_items:
            QMessageBox.warning(self, "Warning", "No PSD data selected for saving")
            return

        try:
            # Get the current file name and path
            current_item = self.file_tree.currentItem()
            if current_item is None:
                raise ValueError("No file selected")

            full_path = current_item.toolTip(0)
            if not full_path:
                raise ValueError("Unable to get file path")

            # Create ProcessingData folder
            original_dir = os.path.dirname(full_path)
            processing_dir = os.path.join(original_dir, 'ProcessingData')
            if not os.path.exists(processing_dir):
                os.makedirs(processing_dir)

            # Save each checked PSD data
            for item in checked_items:
                psd_name = item.text()
                _, frequencies, psd = next((data for data in self.psd_data if data[0] == psd_name), (None, None, None))

                if frequencies is None or psd is None:
                    continue

                # Create filename (psd_name already includes the correct format)
                file_name = f"{psd_name}.h5"
                save_path = os.path.join(processing_dir, file_name)

                # Save to HDF5 file
                with h5py.File(save_path, 'w') as hf:
                    hf.create_dataset('frequencies', data=frequencies)
                    hf.create_dataset('psd', data=psd)

            QMessageBox.information(self, "Success", f"PSD data saved to {processing_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save PSD data: {str(e)}")

    def on_new_data_loaded(self):
        self.psd_data = []  # Clear previous PSD data
        self.psd_list.clear()  # Clear the PSD list widget
        self.update_psd_plot()  # Clear the PSD plot

        # Set default range values
        if self.data['time'] is not None and len(self.data['time']) > 0:
            self.psd_start_input.setText(f"{self.data['time'][0]:.6f}")
            self.psd_start_input.setText(f"{self.data['time'][-1]:.6f}")
        else:
            self.psd_start_input.clear()
            self.psd_end_input.clear()

    def clear_psd(self):
        self.psd_data.clear()
        self.psd_list.clear()
        self.metrics_display.clear()
        self.update_psd_plot()

################################### Filter##############################
    def create_filter_tab(self):
        filter_tab = QWidget()
        main_layout = QVBoxLayout(filter_tab)

        # Settings group
        settings_group = QGroupBox("Filter Settings")
        settings_layout = QVBoxLayout()

        # Filter type selection
        filter_type_layout = QHBoxLayout()
        filter_type_layout.addWidget(QLabel("Filter Type:"))
        self.filter_type = QComboBox()
        self.filter_type.addItems(["Notch", "Median", "Highpass", "Lowpass", "Moving Average"])
        self.filter_type.currentTextChanged.connect(self.on_filter_type_changed)
        filter_type_layout.addWidget(self.filter_type)
        settings_layout.addLayout(filter_type_layout)

        # Parameter input
        self.param_widget = QWidget()
        self.param_layout = QFormLayout(self.param_widget)
        settings_layout.addWidget(self.param_widget)

        # Range input
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Start (s):"))
        self.filter_start = QLineEdit()
        range_layout.addWidget(self.filter_start)
        range_layout.addWidget(QLabel("End (s):"))
        self.filter_end = QLineEdit()
        range_layout.addWidget(self.filter_end)
        settings_layout.addLayout(range_layout)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # Buttons group
        button_group = QGroupBox("Actions")
        button_layout = QHBoxLayout()
        self.filter_apply = QPushButton("Apply")
        self.filter_undo = QPushButton("Undo")
        self.filter_accept = QPushButton("Accept")
        self.filter_save = QPushButton("Save")
        self.filter_apply.clicked.connect(self.apply_filter)
        self.filter_undo.clicked.connect(self.undo_filter)
        self.filter_accept.clicked.connect(self.accept_filter)
        self.filter_save.clicked.connect(self.save_filter)

        # Set a fixed size for all buttons to ensure they are the same width
        button_width = 80
        button_height = 30
        for button in [self.filter_apply, self.filter_undo, self.filter_accept, self.filter_save]:
            button.setFixedSize(button_width, button_height)

        button_layout.addWidget(self.filter_apply)
        button_layout.addWidget(self.filter_undo)
        button_layout.addWidget(self.filter_accept)
        button_layout.addWidget(self.filter_save)

        # Add some spacing between buttons
        button_layout.setSpacing(10)

        button_group.setLayout(button_layout)
        main_layout.addWidget(button_group)

        # Filter list group
        list_group = QGroupBox("Applied Filters")
        list_layout = QVBoxLayout()
        self.filter_list = QListWidget()
        list_layout.addWidget(self.filter_list)
        list_group.setLayout(list_layout)
        main_layout.addWidget(list_group)

        self.tab_widget.addTab(filter_tab, "Filter")

        # Initialize with Notch filter parameters
        self.on_filter_type_changed("Notch")

    def on_filter_type_changed(self, filter_type):
        # Clear previous parameter inputs
        for i in reversed(range(self.param_layout.rowCount())):
            self.param_layout.removeRow(i)

        if filter_type == "Notch":
            self.param_layout.addRow("Notch Frequency (Hz):", QLineEdit("60"))
        elif filter_type in ["Median", "Moving Average"]:
            self.param_layout.addRow("Window Size:", QLineEdit("3"))
        elif filter_type in ["Highpass", "Lowpass"]:
            self.param_layout.addRow("Cutoff Frequency (Hz):", QLineEdit("100"))

    def apply_filter(self):
        if self.data['time'] is None or (self.data['current'] is None and self.data['conductance'] is None):
            QMessageBox.warning(self, "Warning", "No data available for filtering")
            return

        filter_type = self.filter_type.currentText()

        # Get current display range
        current_xlim = self.ax_current.get_xlim()

        # Use current display range if input is empty
        start = float(self.filter_start.text()) if self.filter_start.text() else current_xlim[0]
        end = float(self.filter_end.text()) if self.filter_end.text() else current_xlim[1]

        mask = (self.data['time'] >= start) & (self.data['time'] <= end)
        selected_time = self.data['time'][mask]

        # Copy data to temp_data for filtering
        for key in self.data:
            self.temp_data[key] = self.data[key].copy() if self.data[key] is not None else None

        selected_data = self.temp_data['conductance'][mask] if self.is_conductance_mode else self.temp_data['current'][mask]

        fs = 1 / np.mean(np.diff(selected_time))

        # Get parameter value, use default if empty
        param_widget = self.param_layout.itemAt(1).widget()
        param_value = float(param_widget.text()) if param_widget.text() else None

        try:
            if filter_type == "Notch":
                param_value = param_value or 60  # Default 60Hz
                b, a = signal.iirnotch(param_value, Q=30, fs=fs)
                filtered_data = signal.filtfilt(b, a, selected_data)
            elif filter_type == "Median":
                param_value = int(param_value or 3)  # Default window size 3
                filtered_data = signal.medfilt(selected_data, kernel_size=param_value)
            elif filter_type in ["Highpass", "Lowpass"]:
                param_value = param_value or 100  # Default cutoff 100Hz
                nyq = 0.5 * fs
                normal_cutoff = param_value / nyq
                b, a = signal.butter(5, normal_cutoff, btype='low' if filter_type == "Lowpass" else 'high', analog=False)
                filtered_data = signal.filtfilt(b, a, selected_data)
            elif filter_type == "Moving Average":
                param_value = int(param_value or 3)  # Default window size 3
                filtered_data = np.convolve(selected_data, np.ones(param_value) / param_value, mode='same')
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")

            # Update temp_data with filtered data
            if self.is_conductance_mode:
                self.temp_data['conductance'][mask] = filtered_data
            else:
                self.temp_data['current'][mask] = filtered_data

            # Store filter history
            filter_name = f"{filter_type}_{start:.2f}-{end:.2f}_{len(self.filter_history) + 1}"
            self.filter_history.append((filter_name, selected_time, filtered_data, mask))

            # Add to list widget
            item = QListWidgetItem(filter_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.filter_list.addItem(item)

            # Update plot
            self.update_filter_plot()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying filter: {str(e)}")

    def update_filter_plot(self):
        # Clear previous plots
        self.ax_current.clear()

        # Plot original data
        if self.is_conductance_mode:
            self.ax_current.plot(self.data['time'], self.data['conductance'], color='blue', alpha=0.5,
                                 label='Original Data')
        else:
            self.ax_current.plot(self.data['time'], self.data['current'], color='blue', alpha=0.5,
                                 label='Original Data')

        # Plot filtered data
        if self.is_conductance_mode:
            self.ax_current.plot(self.data['time'], self.temp_data['conductance'], color='red', alpha=0.7,
                                 label='Filtered Data')
        else:
            self.ax_current.plot(self.data['time'], self.temp_data['current'], color='red', alpha=0.7,
                                 label='Filtered Data')

        # Reset labels and legend
        self.ax_current.set_ylabel('Conductance (nS)' if self.is_conductance_mode else 'Current (nA)')
        self.ax_current.set_xlabel('Time (s)')
        self.ax_current.legend()

        # Set x-axis limits to match data range
        x_min, x_max = self.data['time'][0], self.data['time'][-1]
        self.ax_current.set_xlim(x_min, x_max)

        # Use tight_layout to automatically adjust the layout
        self.figure_main.tight_layout()

        # Refresh canvas
        self.canvas_main.draw()

    def undo_filter(self):
        if self.filter_history:
            self.filter_history.pop()
            self.filter_list.takeItem(self.filter_list.count() - 1)
            # Reset temp_data to original data
            for key in self.data:
                self.temp_data[key] = self.data[key].copy() if self.data[key] is not None else None
            # Reapply remaining filters
            for filter_info in self.filter_history:
                self.apply_filter_from_info(filter_info)
            self.update_filter_plot()

    def apply_filter_from_info(self, filter_info):
        # This method should apply a filter based on the stored filter information
        # You'll need to implement this based on how you store filter parameters
        pass

    def accept_filter(self):
        # Move temp_data to central data storage
        for key in self.temp_data:
            if self.temp_data[key] is not None:
                self.data[key] = self.temp_data[key].copy()

        self.filter_history = []
        self.filter_list.clear()
        self.plot_data()  # Update main plot

    def save_filter(self):
        try:
            if self.data['time'] is None or (self.data['current'] is None and self.data['conductance'] is None):
                raise ValueError("No data available to save")

            current_item = self.file_tree.currentItem()
            if current_item is None:
                raise ValueError("No file selected")

            full_path = current_item.toolTip(0)
            if not full_path:
                raise ValueError("Unable to get file path")

            original_dir = os.path.dirname(full_path)
            file_name = os.path.basename(full_path)

            processing_dir = os.path.join(original_dir, 'ProcessingData')
            if not os.path.exists(processing_dir):
                os.makedirs(processing_dir)

            index = 1
            while True:
                new_file_name = f"{os.path.splitext(file_name)[0]}_Filtered_{index}.h5"
                new_full_path = os.path.join(processing_dir, new_file_name)
                if not os.path.exists(new_full_path):
                    break
                index += 1

            with h5py.File(new_full_path, 'w') as hf:
                for key in self.data:
                    if self.data[key] is not None:
                        hf.create_dataset(key, data=self.data[key], compression="gzip", compression_opts=9)

                filter_group = hf.create_group('filters')
                for i, (name, _, _, _) in enumerate(self.filter_history):
                    filter_group.create_dataset(f'filter_{i}', data=name)

            QMessageBox.information(self, "Success", f"Data saved as {new_file_name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving data: {str(e)}")

####################################Fit#####################################################

    def create_fitting_tab(self):
        fitting_tab = QWidget()
        main_layout = QVBoxLayout(fitting_tab)

        # Range group
        range_group = QGroupBox("Fitting Range")
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Start (s):"))
        self.fitting_start = QLineEdit()
        range_layout.addWidget(self.fitting_start)
        range_layout.addWidget(QLabel("End (s):"))
        self.fitting_end = QLineEdit()
        range_layout.addWidget(self.fitting_end)
        range_group.setLayout(range_layout)
        main_layout.addWidget(range_group)

        # Fitting function selection
        function_group = QGroupBox("Fitting Function")
        function_layout = QHBoxLayout()
        function_layout.addWidget(QLabel("Function:"))
        self.fitting_function = QComboBox()
        self.fitting_function.addItems(["Linear", "Polynomial", "Exponential", "Logarithmic", "Gaussian", "Step",
                                        "Gaussian Mixture", "Lorentzian", "Exponential Decay", "Spike Analysis"])
        function_layout.addWidget(self.fitting_function)
        function_group.setLayout(function_layout)
        main_layout.addWidget(function_group)

        # Buttons group
        button_group = QGroupBox("Actions")
        button_layout = QHBoxLayout()
        self.fitting_apply = QPushButton("Apply")
        self.fitting_undo = QPushButton("Undo")
        self.fitting_accept = QPushButton("Accept")
        self.fitting_save = QPushButton("Save")

        self.fitting_apply.clicked.connect(self.apply_fitting)
        self.fitting_undo.clicked.connect(self.undo_fitting)
        self.fitting_accept.clicked.connect(self.accept_fitting)
        self.fitting_save.clicked.connect(self.save_fitting)

        button_layout.addWidget(self.fitting_apply)
        button_layout.addWidget(self.fitting_undo)
        button_layout.addWidget(self.fitting_accept)
        button_layout.addWidget(self.fitting_save)
        button_group.setLayout(button_layout)
        main_layout.addWidget(button_group)

        # Fitting list
        list_group = QGroupBox("Applied Fittings")
        list_layout = QVBoxLayout()
        self.fitting_list = QListWidget()
        self.fitting_list.itemClicked.connect(self.display_fitting_results)
        list_layout.addWidget(self.fitting_list)
        list_group.setLayout(list_layout)
        main_layout.addWidget(list_group)

        # Fitting results display
        results_group = QGroupBox("Fitting Results")
        results_layout = QVBoxLayout()
        self.fitting_results_display = QTextEdit()
        self.fitting_results_display.setReadOnly(True)
        results_layout.addWidget(self.fitting_results_display)

        # Copy button
        copy_button = QPushButton("Copy Results")
        copy_button.clicked.connect(self.copy_fitting_results)
        results_layout.addWidget(copy_button)

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        self.tab_widget.addTab(fitting_tab, "Fitting")

    def display_fitting_results(self, function_type, popt, pcov, x_data, y_data, y_fit):
        results = f"Fitting Results for {function_type} Function:\n\n"

        # Parameters
        results += "Parameters:\n"
        param_names = {
            "Linear": ['a', 'b'],
            "Polynomial": ['a', 'b', 'c'],
            "Exponential": ['a', 'b', 'c'],
            "Logarithmic": ['a', 'b'],
            "Gaussian": ['a', 'b', 'c'],
            "Step": ['y0'] + [f'A{i}' for i in range(1, len(popt) // 2 + 1)] + [f'x{i}' for i in
                                                                                range(1, len(popt) // 2 + 1)]
        }
        for i, (param, name) in enumerate(zip(popt, param_names[function_type])):
            error = np.sqrt(pcov[i, i])
            results += f"{name} = {param:.6f} ± {error:.6f}\n"

        # Equation
        results += "\nFitted Equation:\n"
        if function_type == "Linear":
            results += f"y = {popt[0]:.6f}x + {popt[1]:.6f}\n"
        elif function_type == "Polynomial":
            results += f"y = {popt[0]:.6f}x^2 + {popt[1]:.6f}x + {popt[2]:.6f}\n"
        elif function_type == "Exponential":
            results += f"y = {popt[0]:.6f} * exp({popt[1]:.6f}x) + {popt[2]:.6f}\n"
        elif function_type == "Logarithmic":
            results += f"y = {popt[0]:.6f} * ln(x) + {popt[1]:.6f}\n"
        elif function_type == "Gaussian":
            results += f"y = {popt[0]:.6f} * exp(-(x - {popt[1]:.6f})^2 / (2 * {popt[2]:.6f}^2))\n"
        elif function_type == "Step":
            results += "y = " + " + ".join([f"{popt[i * 2 + 1]:.6f} * H(x - {popt[i * 2 + 2]:.6f})" for i in
                                            range(len(popt) // 2 - 1)]) + f" + {popt[0]:.6f}\n"
        elif function_type == "Gaussian Mixture":
            results += "Gaussian Mixture Components:\n"
            for i, (weight, mean, cov) in enumerate(zip(popt.weights_, popt.means_, popt.covariances_)):
                results += f"Component {i+1}:\n"
                results += f"  Weight: {weight[0]:.6f}\n"
                results += f"  Mean: {mean[0]:.6f}, {mean[1]:.6f}\n"
                results += f"  Covariance: {cov[0][0]:.6f}, {cov[0][1]:.6f}, {cov[1][0]:.6f}, {cov[1][1]:.6f}\n\n"
        elif function_type == "Lorentzian":
            results += "Parameters:\n"
            results += f"Amplitude: {popt[0]:.6f} ± {np.sqrt(pcov[0, 0]):.6f}\n"
            results += f"Center: {popt[1]:.6f} ± {np.sqrt(pcov[1, 1]):.6f}\n"
            results += f"Width: {popt[2]:.6f} ± {np.sqrt(pcov[2, 2]):.6f}\n\n"
            results += "Fitted Equation:\n"
            results += f"y = {popt[0]:.6f} * {popt[2]:.6f}^2 / ((x - {popt[1]:.6f})^2 + {popt[2]:.6f}^2)\n"
        elif function_type == "Exponential Decay":
            results += "Parameters:\n"
            results += f"A: {popt[0]:.6f} ± {np.sqrt(pcov[0, 0]):.6f}\n"
            results += f"tau: {popt[1]:.6f} ± {np.sqrt(pcov[1, 1]):.6f}\n"
            results += f"C: {popt[2]:.6f} ± {np.sqrt(pcov[2, 2]):.6f}\n\n"
            results += "Fitted Equation:\n"
            results += f"y = {popt[0]:.6f} * exp(-x/{popt[1]:.6f}) + {popt[2]:.6f}\n"
        elif function_type == "Spike Analysis":
            peaks, peak_heights, time_widths = popt
            results += "Spike Information:\n"
            for i, (peak, height, width) in enumerate(zip(peaks, peak_heights, time_widths)):
                results += f"Spike {i+1}:\n"
                results += f"  Time: {x_data[peak]:.3f}\n"
                results += f"  Height: {height:.2f}\n"
                results += f"  Width: {width:.3f}\n\n"

        # Goodness of fit
        residuals = y_data - y_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals ** 2))

        results += f"\nR-squared: {r_squared:.6f}\n"
        results += f"RMSE: {rmse:.6f}\n"

        self.fitting_results_display.setText(results)

    def copy_fitting_results(self):
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.fitting_results_display.toPlainText())
        QMessageBox.information(self, "Info", "Fitting results copied to clipboard")

    def linear_fit(self, x, a, b):
        return a * x + b

    def polynomial_fit(self, x, a, b, c):
        return a * x**2 + b * x + c

    def exponential_fit(self, x, a, b, c):
        return a * np.exp(b * x) + c

    def logarithmic_fit(self, x, a, b):
        return a * np.log(x) + b

    def gaussian_fit(self, x, a, b, c):
        return a * np.exp(-(x - b)**2 / (2 * c**2))

    def multi_step_fit(self, x, *params):
        y0 = params[0]
        steps = params[1:]
        y = np.full_like(x, y0)
        for i in range(0, len(steps), 2):
            A, x0 = steps[i:i+2]
            y += A * np.heaviside(x - x0, 1)
        return y

    def estimate_steps(self, y_data):
        diff = np.diff(y_data)
        threshold = np.std(diff) * 2  # Adjust this multiplier as needed
        change_points = np.where(np.abs(diff) > threshold)[0]
        return len(change_points)

    def gaussian_mixture_fit(self, x, y, n_components=2):
        try:
            X = np.column_stack((x, y))
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(X)
            return gmm
        except Exception as e:
            logging.error(f"Error in gaussian_mixture_fit: {str(e)}")
            raise

    def lorentzian_fit(self, x, amp, cen, wid):
        return amp * wid ** 2 / ((x - cen) ** 2 + wid ** 2)

    def exponential_decay_fit(self, x, A, tau, C):
        return A * np.exp(-x / tau) + C

    def analyze_spikes(self, x, y, height_threshold=0.5, distance=50):
        # Find peaks
        peaks, _ = find_peaks(y, height=height_threshold, distance=distance)

        # Calculate properties of the peaks
        peak_heights = y[peaks]
        widths, width_heights, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)

        # Convert width to time units
        time_widths = widths * (x[1] - x[0])

        return peaks, peak_heights, time_widths

    def simplified_fit_spikes(self, x, y, peaks, widths):
        y_fit = np.zeros_like(y)
        for peak, width in zip(peaks, widths):
            left = int(peak - width / 2)
            right = int(peak + width / 2)
            y_fit[left:right] = y[peak]
        return y_fit

    def apply_fitting(self):
        if self.data['time'] is None or (self.data['current'] is None and self.data['conductance'] is None):
            QMessageBox.warning(self, "Warning", "No data available for fitting")
            return

        # Get current display range
        current_xlim = self.ax_current.get_xlim()

        # Use current display range if input is empty
        start = float(self.fitting_start.text()) if self.fitting_start.text() else current_xlim[0]
        end = float(self.fitting_end.text()) if self.fitting_end.text() else current_xlim[1]

        mask = (self.data['time'] >= start) & (self.data['time'] <= end)
        x_data = self.data['time'][mask]
        y_data = self.data['conductance'][mask] if self.is_conductance_mode else self.data['current'][mask]

        # Remove any NaN or inf values
        finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[finite_mask]
        y_data = y_data[finite_mask]

        function = self.fitting_function.currentText()

        try:
            if function == "Linear":
                popt, pcov = curve_fit(self.linear_fit, x_data, y_data)
                y_fit = self.linear_fit(x_data, *popt)
                self.display_fitting_results("Linear", popt, pcov, x_data, y_data, y_fit)

            elif function == "Polynomial":
                popt, pcov = curve_fit(self.polynomial_fit, x_data, y_data)
                y_fit = self.polynomial_fit(x_data, *popt)
                self.display_fitting_results("Polynomial", popt, pcov, x_data, y_data, y_fit)

            elif function == "Exponential":
                popt, pcov = curve_fit(self.exponential_fit, x_data, y_data)
                y_fit = self.exponential_fit(x_data, *popt)
                self.display_fitting_results("Exponential", popt, pcov, x_data, y_data, y_fit)

            elif function == "Logarithmic":
                # Ensure all x values are positive for log fitting
                positive_mask = x_data > 0
                x_data = x_data[positive_mask]
                y_data = y_data[positive_mask]
                popt, pcov = curve_fit(self.logarithmic_fit, x_data, y_data)
                y_fit = self.logarithmic_fit(x_data, *popt)
                self.display_fitting_results("Logarithmic", popt, pcov, x_data, y_data, y_fit)

            elif function == "Gaussian":
                popt, pcov = curve_fit(self.gaussian_fit, x_data, y_data)
                y_fit = self.gaussian_fit(x_data, *popt)
                self.display_fitting_results("Gaussian", popt, pcov, x_data, y_data, y_fit)

            elif function == "Step":
                n_steps = self.estimate_steps(y_data)
                p0 = [np.min(y_data)] + [np.ptp(y_data) / n_steps, np.mean(x_data)] * n_steps
                popt, pcov = curve_fit(self.multi_step_fit, x_data, y_data, p0=p0)
                y_fit = self.multi_step_fit(x_data, *popt)
                self.display_fitting_results("Step", popt, pcov, x_data, y_data, y_fit)

            elif function == "Gaussian Mixture":
                n_components = 2  # 你可能想让这个可调
                gmm = self.gaussian_mixture_fit(x_data, y_data, n_components)

                # 使用更安全的方法计算y_fit
                y_fit = np.zeros_like(x_data)
                for w, m, c in zip(gmm.weights_, gmm.means_, gmm.covariances_):
                    y_fit += w * stats.multivariate_normal(mean=m, cov=c).pdf(np.column_stack((x_data, y_data)))

                self.display_fitting_results("Gaussian Mixture", gmm, None, x_data, y_data, y_fit)

            elif function == "Lorentzian":
                p0 = [np.max(y_data), np.mean(x_data), np.std(x_data)]  # 初始猜测
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        popt, pcov = curve_fit(self.lorentzian_fit, x_data, y_data, p0=p0)
                    except OptimizeWarning:
                        popt, pcov = curve_fit(self.lorentzian_fit, x_data, y_data, p0=p0, method='lm')
                y_fit = self.lorentzian_fit(x_data, *popt)
                self.display_fitting_results("Lorentzian", popt, pcov, x_data, y_data, y_fit)

            elif function == "Exponential Decay":
                p0 = [np.max(y_data), (x_data[-1] - x_data[0]) / 2, np.min(y_data)]  # Initial guess
                popt, pcov = curve_fit(self.exponential_decay_fit, x_data, y_data, p0=p0)
                y_fit = self.exponential_decay_fit(x_data, *popt)

            elif function == "Spike Analysis":
                height_threshold = 0.5  # You might want to make this adjustable
                distance = 50  # You might want to make this adjustable
                peaks, peak_heights, time_widths = self.analyze_spikes(x_data, y_data, height_threshold, distance)
                y_fit = self.simplified_fit_spikes(x_data, y_data, peaks, time_widths)
                self.display_fitting_results("Spike Analysis", (peaks, peak_heights, time_widths), None, x_data, y_data, y_fit)

            else:
                raise ValueError(f"Unknown fitting function: {function}")

            # Store fitting results
            fit_name = f"{os.path.splitext(self.file_tree.currentItem().text(0))[0]}_{function}_{start:.2f}-{end:.2f}_{len(self.fitting_history) + 1}"
            self.fitting_history.append((fit_name, x_data, y_fit))

            # Add to list widget
            item = QListWidgetItem(fit_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.fitting_list.addItem(item)

            # Update plot
            self.update_fitting_plot()

        except Exception as e:
            error_msg = f"Error applying fitting: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logging.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def update_fitting_plot(self):
        # Clear previous plots
        self.ax_current.clear()

        # Plot original data
        if self.is_conductance_mode:
            self.ax_current.plot(self.data['time'], self.data['conductance'], color='blue', alpha=0.5,
                                 label='Original Data')
        else:
            self.ax_current.plot(self.data['time'], self.data['current'], color='blue', alpha=0.5,
                                 label='Original Data')

        # Plot fitted data
        for fit_name, x_data, y_fit in self.fitting_history:
            if "Spike Analysis" in fit_name:
                self.ax_current.plot(x_data, y_fit, 'r-', linewidth=2, alpha=0.7, label=fit_name)
                peaks, peak_heights, _ = self.analyze_spikes(x_data, y_fit)
                self.ax_current.plot(x_data[peaks], peak_heights, "x", color='g', label='Detected Peaks')
            else:
                self.ax_current.plot(x_data, y_fit, 'r--', alpha=0.7, label=fit_name)

        # Reset labels and legend
        self.ax_current.set_ylabel('Conductance (nS)' if self.is_conductance_mode else 'Current (nA)')
        self.ax_current.set_xlabel('Time (s)')
        self.ax_current.legend()

        # Set x-axis limits to match data range
        x_min, x_max = self.data['time'][0], self.data['time'][-1]
        self.ax_current.set_xlim(x_min, x_max)

        # Use tight_layout to automatically adjust the layout
        self.figure_main.tight_layout()

        # Refresh canvas
        self.canvas_main.draw()

    def undo_fitting(self):
        if self.fitting_history:
            self.fitting_history.pop()
            self.fitting_list.takeItem(self.fitting_list.count() - 1)
            self.update_fitting_plot()

    def accept_fitting(self):
        # In this case, we don't modify the original data
        # We just keep the fitting results for visualization and saving
        self.update_fitting_plot()

    def save_fitting(self):
        try:
            checked_items = [self.fitting_list.item(i) for i in range(self.fitting_list.count())
                             if self.fitting_list.item(i).checkState() == Qt.CheckState.Checked]

            if not checked_items:
                QMessageBox.warning(self, "Warning", "No fitting data selected for saving")
                return

            current_item = self.file_tree.currentItem()
            if current_item is None:
                raise ValueError("No file selected")

            full_path = current_item.toolTip(0)
            if not full_path:
                raise ValueError("Unable to get file path")

            original_dir = os.path.dirname(full_path)
            processing_dir = os.path.join(original_dir, 'ProcessingData')
            if not os.path.exists(processing_dir):
                os.makedirs(processing_dir)

            for item in checked_items:
                fit_name = item.text()
                _, x_data, y_fit = next((data for data in self.fitting_history if data[0] == fit_name), (None, None, None))

                if x_data is None or y_fit is None:
                    continue

                file_name = f"{fit_name}.h5"
                save_path = os.path.join(processing_dir, file_name)

                with h5py.File(save_path, 'w') as hf:
                    hf.create_dataset('time', data=x_data)
                    hf.create_dataset('fitted_data', data=y_fit)

            QMessageBox.information(self, "Success", f"Fitting data saved to {processing_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save fitting data: {str(e)}")

###################################conductance###################################################

    def create_conductance_tab(self):
        conductance_tab = QWidget()
        main_layout = QVBoxLayout(conductance_tab)

        # Range group
        range_group = QGroupBox("Conductance Range")
        range_layout = QGridLayout()
        range_layout.addWidget(QLabel("Start (s):"), 0, 0)
        self.conductance_start = QLineEdit()
        range_layout.addWidget(self.conductance_start, 0, 1)
        range_layout.addWidget(QLabel("End (s):"), 1, 0)
        self.conductance_end = QLineEdit()
        range_layout.addWidget(self.conductance_end, 1, 1)
        range_group.setLayout(range_layout)
        main_layout.addWidget(range_group)

        # Buttons group
        button_group = QGroupBox("Actions")
        button_layout = QGridLayout()
        self.convert_button = QPushButton("Convert to Conductance")
        self.undo_conductance_button = QPushButton("Undo")
        self.accept_conductance_button = QPushButton("Accept")
        self.save_conductance_button = QPushButton("Save")

        self.convert_button.clicked.connect(self.convert_conductance)
        self.undo_conductance_button.clicked.connect(self.undo_conductance)
        self.accept_conductance_button.clicked.connect(self.accept_conductance)
        self.save_conductance_button.clicked.connect(self.save_conductance)

        button_layout.addWidget(self.convert_button, 0, 0, 1, 2)
        button_layout.addWidget(self.undo_conductance_button, 1, 0)
        button_layout.addWidget(self.accept_conductance_button, 1, 1)
        button_layout.addWidget(self.save_conductance_button, 2, 0, 1, 2)
        button_group.setLayout(button_layout)
        main_layout.addWidget(button_group)

        # Add some vertical space
        main_layout.addStretch(1)

        self.tab_widget.addTab(conductance_tab, "Conductance")

        # Initialize conductance state
        self.reset_conductance_state()

    def reset_conductance_state(self):
        self.is_conductance_mode = False
        self.data['conductance'] = None
        self.temp_data['conductance'] = None
        if hasattr(self, 'convert_button'):
            self.convert_button.setText("Convert to Conductance")

    def convert_conductance(self):
        logging.info(f"Converting data. Current mode: {self.is_conductance_mode}")
        if self.data['current'] is None or self.data['voltage'] is None:
            QMessageBox.warning(self, "Error", "No data available for conversion")
            return

        try:
            # Copy data to temp_data for conversion
            for key in self.data:
                self.temp_data[key] = self.data[key].copy() if self.data[key] is not None else None

            if not self.is_conductance_mode:
                # Convert current to conductance
                self.temp_data['conductance'] = self.temp_data['current'] / self.temp_data['voltage']
                self.is_conductance_mode = True
                logging.info("Converted to conductance mode")
            else:
                # Convert conductance back to current
                self.temp_data['current'] = self.temp_data['conductance'] * self.temp_data['voltage']
                self.temp_data['conductance'] = None
                self.is_conductance_mode = False
                logging.info("Converted to current mode")

            self.update_convert_button_text()
            self.update_conductance_plot()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Conversion failed: {str(e)}")
            logging.error(f"Conversion failed: {str(e)}", exc_info=True)

        
    def update_convert_button_text(self):
        if self.is_conductance_mode:
            self.convert_button.setText("Convert to Current")
        else:
            self.convert_button.setText("Convert to Conductance")

    def undo_conductance(self):
        # Restore original data
        for key in self.data:
            self.temp_data[key] = self.data[key].copy() if self.data[key] is not None else None

        self.is_conductance_mode = self.data['conductance'] is not None
        self.convert_button.setText("Convert to Current" if self.is_conductance_mode else "Convert to Conductance")

        self.update_conductance_plot()

    def accept_conductance(self):
        logging.info(f"Accepting conductance changes. Current mode: {self.is_conductance_mode}")
        
        # Move temp_data to central data storage
        for key in self.temp_data:
            if self.temp_data[key] is not None:
                self.data[key] = self.temp_data[key].copy()

        # Ensure conductance mode is correctly set
        if self.data['conductance'] is not None:
            self.is_conductance_mode = True
            # Store original current data if not already stored
            if self.original_current is None:
                self.original_current = self.data['current'].copy()
        else:
            self.is_conductance_mode = False
            # Restore original current data if exists
            if self.original_current is not None:
                self.data['current'] = self.original_current.copy()
            self.original_current = None

        logging.info(f"Mode after accepting changes: {self.is_conductance_mode}")

        # Update the main plot
        self.plot_data()

        # Update PSD plot if necessary
        self.update_psd_plot()

        # Update the convert button text
        self.update_convert_button_text()

        # Inform the user
        QMessageBox.information(self, "Success", "Changes have been accepted and applied.")

    def update_conductance_plot(self):
        logging.info(f"Updating conductance plot. Mode: {self.is_conductance_mode}")
        # Clear current plots
        self.ax_voltage.clear()
        self.ax_current.clear()

        # Determine the range to plot
        start = float(self.conductance_start.text()) if self.conductance_start.text() else self.temp_data['time'][0]
        end = float(self.conductance_end.text()) if self.conductance_end.text() else self.temp_data['time'][-1]

        mask = (self.temp_data['time'] >= start) & (self.temp_data['time'] <= end)
        plot_time = self.temp_data['time'][mask]

        # Plot voltage data
        plot_voltage = self.temp_data['voltage'][mask]
        self.ax_voltage.plot(plot_time, plot_voltage, color=self.show_settings['voltage_color'])
        self.ax_voltage.set_ylabel('Voltage (V)')

        # Plot current or conductance data
        if self.is_conductance_mode:
            plot_data = self.temp_data['conductance'][mask]
            self.ax_current.plot(plot_time, plot_data, color=self.show_settings['current_color'])
            self.ax_current.set_ylabel('Conductance (nS)')
            logging.info("Plotted conductance data")
        else:
            plot_data = self.temp_data['current'][mask]
            self.ax_current.plot(plot_time, plot_data, color=self.show_settings['current_color'])
            self.ax_current.set_ylabel('Current (nA)')
            logging.info("Plotted current data")

        self.ax_current.set_xlabel('Time (s)')

        # Set x-axis limits for both plots to exactly match the data range
        x_min, x_max = plot_time[0], plot_time[-1]
        self.ax_voltage.set_xlim(x_min, x_max)
        self.ax_current.set_xlim(x_min, x_max)

        # Ensure Voltage and Current plots don't have grid lines
        self.ax_voltage.grid(False)
        self.ax_current.grid(False)

        self.setup_axes()

        # Refresh main canvas
        self.canvas_main.draw()

        logging.info(f"Conductance plot updated. Final mode: {self.is_conductance_mode}")

    def save_conductance(self):
        try:
            if self.temp_data['conductance'] is None and not self.is_conductance_mode:
                raise ValueError("No conductance data available to save")

            current_item = self.file_tree.currentItem()
            if current_item is None:
                raise ValueError("No file selected")

            full_path = current_item.toolTip(0)
            if not full_path:
                raise ValueError("Unable to get file path")

            original_dir = os.path.dirname(full_path)
            file_name = os.path.basename(full_path)

            processing_dir = os.path.join(original_dir, 'ProcessingData')
            if not os.path.exists(processing_dir):
                os.makedirs(processing_dir)

            index = 1
            while True:
                new_file_name = f"{os.path.splitext(file_name)[0]}_Conductance_{index}.h5"
                new_full_path = os.path.join(processing_dir, new_file_name)
                if not os.path.exists(new_full_path):
                    break
                index += 1

            start = float(self.conductance_start.text()) if self.conductance_start.text() else self.temp_data['time'][0]
            end = float(self.conductance_end.text()) if self.conductance_end.text() else self.temp_data['time'][-1]
            mask = (self.temp_data['time'] >= start) & (self.temp_data['time'] <= end)

            with h5py.File(new_full_path, 'w') as hf:
                hf.create_dataset('time', data=self.temp_data['time'][mask])
                hf.create_dataset('voltage', data=self.temp_data['voltage'][mask])
                hf.create_dataset('current', data=self.temp_data['current'][mask])
                if self.is_conductance_mode:
                    hf.create_dataset('conductance', data=self.temp_data['conductance'][mask])
                else:
                    conductance = self.temp_data['current'][mask] / self.temp_data['voltage'][mask]
                    hf.create_dataset('conductance', data=conductance)

            QMessageBox.information(self, "Success", f"Data saved as {new_file_name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save conductance data: {str(e)}")

    def create_plot_tab(self):
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)

        # Trace Plot Group
        trace_group = QGroupBox("Plot Trace")
        trace_layout = QVBoxLayout(trace_group)
        self.plot_trace_button = QPushButton("Plot Trace")
        self.plot_trace_button.clicked.connect(self.plot_trace)
        trace_layout.addWidget(self.plot_trace_button)
        plot_layout.addWidget(trace_group)

        # PSD Plot Group
        psd_group = QGroupBox("Plot PSD")
        psd_layout = QVBoxLayout(psd_group)
        self.plot_psd_button = QPushButton("Plot PSD")
        self.plot_psd_button.clicked.connect(self.plot_psd)
        psd_layout.addWidget(self.plot_psd_button)
        plot_layout.addWidget(psd_group)

        self.tab_widget.addTab(plot_tab, "Plot")

    def plot_trace(self):
        if self.data['time'] is None or self.data['voltage'] is None or self.data['current'] is None:
            QMessageBox.warning(self, "Warning", "No data available for plotting")
            return

        self.plot_window = PlotCustomizationWindow(self, data=self.data)
        self.plot_window.show()
    
    def plot_psd(self):
        if not self.psd_data:
            QMessageBox.warning(self, "Warning", "No PSD data available for plotting")
            return

        self.plot_window = PSDPlotCustomizationWindow(self, psd_data=self.psd_data)
        self.plot_window.show()
    
    def closeEvent(self, event):
        logging.info("PreprocessWindow closing")
        self.closed.emit()
        super().closeEvent(event)