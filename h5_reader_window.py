import os
import logging
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QToolBar,
                             QFileDialog, QPushButton, QTreeView, QTextEdit, QSplitter,
                             QLabel, QHeaderView, QMessageBox, QSlider, QLineEdit, QListWidgetItem, QListWidget,
                             QFormLayout, QGroupBox, QScrollArea, QCheckBox, QApplication, QComboBox, QAbstractItemView)
from PyQt6.QtGui import QAction, QIcon, QFileSystemModel, QFont, QIntValidator, QDoubleValidator
from PyQt6.QtCore import Qt, QDir
import h5py
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
from scipy import stats



class H5ReaderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("H5 Reader")
        self.setGeometry(100, 100, 1600, 1000)

        # Initialize attributes
        self.current_file = None
        self.time_data = None
        self.voltage_data = None
        self.current_data = None
        self.conductance_data = None
        self.current_inverted = False
        self.highlight_rect = None
        self.plot_type = 'current'  # Default to current
        self.highlight_width = 0.1  # Default width of 0.1 seconds
        self.is_updating = False  # Add this line to initialize is_updating

        # Color scheme
        self.colors = {
            'background': '#FFFFFF',
            'voltage': '#0e38b1',
            'current': '#64af17',
            'conductance': '#fb6b05',
            'highlight': '#fdff50'
        }
        self.highlight_alpha = 0.5  # Define the alpha (transparency) for the highlight

        self.setup_ui()

    def setup_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left side (file browser, info, notes, and controls)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(350)  # Set a fixed width for the left side

        # Toolbar
        self.toolbar = QToolBar()
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Import action
        import_action = QAction(QIcon("path/to/import_icon.png"), "Import", self)
        import_action.triggered.connect(self.import_files)
        self.toolbar.addAction(import_action)

        # Current/Conductance Histogram action
        hist_action = QAction("Histogram", self)
        hist_action.triggered.connect(self.show_histogram)
        self.toolbar.addAction(hist_action)

        # Current/Conductance Derivative action
        detail_action = QAction("Derivative", self)
        detail_action.triggered.connect(self.show_derivative)
        self.toolbar.addAction(detail_action)

        # Add plot type selection
        plot_type_layout = QHBoxLayout()
        plot_type_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(['Current', 'Conductance'])
        self.plot_type_combo.currentTextChanged.connect(self.change_plot_type)
        plot_type_layout.addWidget(self.plot_type_combo)
        left_layout.addLayout(plot_type_layout)

        # File system model and view
        self.file_system_model = QFileSystemModel()
        self.file_system_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self.file_system_model.setNameFilters(["*.h5"])
        self.file_system_model.setNameFilterDisables(False)

        self.file_tree_view = QTreeView()
        self.file_tree_view.setModel(self.file_system_model)
        self.file_tree_view.clicked.connect(self.on_file_selected)
        
        # Show only the first column (file name) and hide others
        for i in range(1, self.file_system_model.columnCount()):
            self.file_tree_view.hideColumn(i)
        
        self.file_tree_view.setHeaderHidden(True)
        self.file_tree_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.file_tree_view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        
        # Ensure the column can be resized
        self.file_tree_view.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.file_tree_view.header().setStretchLastSection(False)
        
        left_layout.addWidget(self.file_tree_view)

        # File info text area
        self.file_info_text = QTextEdit()
        self.file_info_text.setReadOnly(True)
        self.file_info_text.setMaximumHeight(150)
        left_layout.addWidget(self.file_info_text)

        # Notes area
        notes_layout = QVBoxLayout()
        notes_layout.addWidget(QLabel("Notes:"))
        self.notes_text = QTextEdit()
        notes_layout.addWidget(self.notes_text)
        self.save_notes_button = QPushButton("Save Notes")
        self.save_notes_button.clicked.connect(self.save_notes)
        notes_layout.addWidget(self.save_notes_button)
        left_layout.addLayout(notes_layout)

        # Controls area
        self.setup_controls(left_layout)

        # Add highlight width control
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Highlight Width (s):"))
        self.highlight_width_input = QLineEdit(str(self.highlight_width))
        self.highlight_width_input.setValidator(QDoubleValidator(0, 100, 2))
        self.highlight_width_input.textChanged.connect(self.update_highlight_width)
        width_layout.addWidget(self.highlight_width_input)
        left_layout.addLayout(width_layout)

        # Add highlight slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Highlight Position:"))
        self.highlight_slider = QSlider(Qt.Orientation.Horizontal)
        self.highlight_slider.setMinimum(0)
        self.highlight_slider.setMaximum(1000)
        self.highlight_slider.valueChanged.connect(self.update_highlight_from_slider)
        slider_layout.addWidget(self.highlight_slider)
        left_layout.addLayout(slider_layout)

        # Right side (data visualization)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        # Matplotlib Figure
        self.figure = plt.figure(figsize=(12, 15), constrained_layout=True)
        gs = self.figure.add_gridspec(3, 2, height_ratios=[1, 1, 2], width_ratios=[5, 1])
        self.ax1 = self.figure.add_subplot(gs[0, :])
        self.ax2 = self.figure.add_subplot(gs[1, :], sharex=self.ax1)
        self.ax3 = self.figure.add_subplot(gs[2, 0])
        self.ax4 = self.figure.add_subplot(gs[2, 1], sharey=self.ax3)

        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # Matplotlib Toolbar
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.mpl_toolbar)

        # Add left and right widgets to main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([350, 1250])  # Adjust sizes to give more space to visualization
        main_layout.addWidget(splitter)

        # Set fonts
        chinese_font = QFont("Microsoft YaHei", 9)
        self.file_info_text.setFont(chinese_font)
        self.notes_text.setFont(chinese_font)

        # Set default directory
        self.set_default_directory()

    def setup_controls(self, layout):
        controls_layout = QVBoxLayout()

        # Time range input
        time_range_layout = QHBoxLayout()
        time_range_layout.addWidget(QLabel("Time Range:"))
        self.time_min_input = QLineEdit()
        self.time_max_input = QLineEdit()
        time_range_layout.addWidget(self.time_min_input)
        time_range_layout.addWidget(QLabel("to"))
        time_range_layout.addWidget(self.time_max_input)
        controls_layout.addLayout(time_range_layout)

        # Bins input
        bins_layout = QHBoxLayout()
        bins_layout.addWidget(QLabel("Histogram Bins:"))
        self.bins_input = QLineEdit()
        self.bins_input.setText("50")  # Default value
        bins_layout.addWidget(self.bins_input)
        controls_layout.addLayout(bins_layout)

        # Control buttons
        button_layout = QHBoxLayout()
        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_plots)
        button_layout.addWidget(self.update_button)
        self.invert_current_button = QPushButton("Invert Current")
        self.invert_current_button.clicked.connect(self.invert_current)
        button_layout.addWidget(self.invert_current_button)
        controls_layout.addLayout(button_layout)

        layout.addLayout(controls_layout)

    def set_default_directory(self):
        default_path = r"H:\Experiment"
        if os.path.exists(default_path) and os.path.isdir(default_path):
            self.file_system_model.setRootPath(default_path)
            self.file_tree_view.setRootIndex(self.file_system_model.index(default_path))
        else:
            logging.warning(f"Default directory '{default_path}' not found. User can import manually.")

    def import_files(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.file_system_model.setRootPath(directory)
            self.file_tree_view.setRootIndex(self.file_system_model.index(directory))

    def set_root_directory(self, directory):
        self.file_system_model.setRootPath(directory)
        self.file_tree_view.setRootIndex(self.file_system_model.index(directory))
        self.add_to_history(directory)

        # Adjust column width to fit content
        self.file_tree_view.resizeColumnToContents(0)

    def on_item_expanded_or_collapsed(self, index):
        # Save the current scroll position
        h_scroll = self.file_tree_view.horizontalScrollBar().value()
        v_scroll = self.file_tree_view.verticalScrollBar().value()

        # Force the view to update
        self.file_tree_view.updateGeometries()

        # Restore the scroll position
        self.file_tree_view.horizontalScrollBar().setValue(h_scroll)
        self.file_tree_view.verticalScrollBar().setValue(v_scroll)

    def on_file_selected(self, index):
        file_path = self.file_system_model.filePath(index)
        if file_path.endswith('.h5'):
            # Save current horizontal scroll position
            h_scroll = self.file_tree_view.horizontalScrollBar().value()

            self.current_file = file_path
            self.display_h5_info(file_path)

            # Reset visualization area
            self.reset_visualization()

            # Attempt to visualize data
            try:
                self.visualize_h5_data(file_path)
            except Exception as e:
                self.show_visualization_error(str(e))

            self.load_notes(file_path)
            self.current_inverted = False  # Reset inversion state for new file

            # Restore horizontal scroll position
            self.file_tree_view.horizontalScrollBar().setValue(h_scroll)


    def reset_visualization(self):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.set_facecolor(self.colors['background'])
        self.canvas.draw()
        self.time_data = None
        self.current_data = None
        self.voltage_data = None

    def show_visualization_error(self, error_message):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.set_facecolor(self.colors['background'])
            ax.text(0.5, 0.5, f"Error visualizing H5 file:\n{error_message}",
                    ha='center', va='center', wrap=True)
        self.canvas.draw()
        logging.error(f"Error in visualize_h5_data: {error_message}")

    def display_h5_info(self, file_path):
        try:
            with h5py.File(file_path, 'r') as f:
                info = f"File Path: {file_path}\n\n"
                info += "File Structure:\n"
                info += self.get_h5_structure(f)
                info += "\nAttributes:\n"
                for key, value in f.attrs.items():
                    info += f"  {key}: {value}\n"

                # Display notes if they exist
                if 'metadata/notes' in f:
                    notes = f['metadata/notes'][0]
                    # Decode if it's bytes, otherwise use as is
                    if isinstance(notes, bytes):
                        notes = notes.decode('utf-8')
                    info += "\nNotes:\n"
                    # Limit the display to first 100 characters
                    if len(notes) > 100:
                        info += f"{notes[:100]}...\n(Note truncated, full content in Notes section)"
                    else:
                        info += f"{notes}\n"

                self.file_info_text.setText(info)
        except Exception as e:
            self.file_info_text.setText(f"Error reading H5 file: {str(e)}")
            logging.error(f"Error in display_h5_info: {str(e)}")

    def get_h5_structure(self, group, prefix=''):
        info = ""
        for key, item in group.items():
            if key == 'metadata' and isinstance(item, h5py.Group):
                info += f"{prefix}{key}/ (Metadata Group)\n"
                info += self.get_h5_structure(item, prefix + '  ')
            elif isinstance(item, h5py.Group):
                info += f"{prefix}{key}/ (Group)\n"
                info += self.get_h5_structure(item, prefix + '  ')
            elif isinstance(item, h5py.Dataset):
                if key == 'notes' and isinstance(group, h5py.Group) and group.name == '/metadata':
                    info += f"{prefix}{key} (Notes Dataset)\n"
                else:
                    info += f"{prefix}{key} (Dataset: shape {item.shape}, dtype {item.dtype})\n"
        return info

    def change_plot_type(self, plot_type):
        self.plot_type = plot_type.lower()
        if self.plot_type == 'conductance' and self.conductance_data is None:
            QMessageBox.warning(self, "Warning", "Conductance data not available for this file.")
            self.plot_type_combo.setCurrentText('Current')
            self.plot_type = 'current'
        else:
            self.update_plots()

    def update_highlight_from_slider(self):
        if self.is_updating or self.time_data is None:
            return
        self.is_updating = True
        try:
            slider_value = self.highlight_slider.value() / 1000
            time_range = self.time_data[-1] - self.time_data[0]
            highlight_center = self.time_data[0] + time_range * slider_value
            t_min = max(self.time_data[0], highlight_center - self.highlight_width / 2)
            t_max = min(self.time_data[-1], highlight_center + self.highlight_width / 2)
            self.time_min_input.setText(f"{t_min:.2f}")
            self.time_max_input.setText(f"{t_max:.2f}")
            self.update_highlight(t_min, t_max)
            self.update_plot3(t_min, t_max)
            self.update_plot4(t_min, t_max)
            self.canvas.draw_idle()
        finally:
            self.is_updating = False

    def update_highlight_width(self):
            try:
                new_width = float(self.highlight_width_input.text())
                if new_width > 0:
                    self.highlight_width = new_width
                    self.update_highlight_from_slider()
            except ValueError:
                pass  # Ignore invalid input
    def update_highlight(self, t_min, t_max):
        if hasattr(self, 'highlight_rect') and self.highlight_rect:
            self.highlight_rect.remove()
        self.highlight_rect = Rectangle((t_min, self.ax2.get_ylim()[0]), t_max - t_min,
                                        self.ax2.get_ylim()[1] - self.ax2.get_ylim()[0],
                                        facecolor=self.colors['highlight'], alpha=self.highlight_alpha)
        self.ax2.add_patch(self.highlight_rect)
        self.canvas.draw_idle()

    def visualize_h5_data(self, file_path):
        try:
            with h5py.File(file_path, 'r') as f:
                # Read data
                self.time_data = f['time'][:]
                self.current_data = f['current'][:]
                self.voltage_data = f['voltage'][:]
                
                # Check for conductance data
                if 'conductance' in f:
                    self.conductance_data = f['conductance'][:]
                else:
                    self.conductance_data = None
                    if self.plot_type == 'conductance':
                        QMessageBox.warning(self, "Warning", "Conductance data not available. Switching to current plot.")
                        self.plot_type = 'current'
                        self.plot_type_combo.setCurrentText('Current')

            # Validate data
            if self.time_data is None or self.current_data is None or self.voltage_data is None:
                raise ValueError("Required data (time, current, or voltage) not found in the file.")

            # Ensure all data arrays have the same length
            min_length = min(len(self.time_data), len(self.current_data), len(self.voltage_data))
            self.time_data = self.time_data[:min_length]
            self.current_data = self.current_data[:min_length]
            self.voltage_data = self.voltage_data[:min_length]
            if self.conductance_data is not None:
                self.conductance_data = self.conductance_data[:min_length]

            # Set initial time range for zoom
            time_range = self.time_data[-1] - self.time_data[0]
            initial_zoom = time_range * 0.1  # 10% of total time range
            zoom_start = self.time_data[0] + time_range * 0.45
            zoom_end = zoom_start + initial_zoom

            self.time_min_input.setText(f"{zoom_start:.2f}")
            self.time_max_input.setText(f"{zoom_end:.2f}")

            # Set default number of bins
            self.bins_input.setText("50")

            # Initialize slider range
            self.highlight_slider.setMinimum(0)
            self.highlight_slider.setMaximum(1000)
            self.highlight_slider.setValue(500)  # Set to middle by default

            # Update file info
            self.display_h5_info(file_path)

            # Plot data
            self.plot_data()

            # Update highlight based on initial zoom
            self.update_highlight(zoom_start, zoom_end)

            # Load notes if any
            self.load_notes(file_path)

            # Reset inversion state for new file
            self.current_inverted = False
            self.invert_current_button.setChecked(False)

            logging.info(f"Successfully visualized data from {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize H5 file: {str(e)}")
            logging.error(f"Error in visualize_h5_data: {str(e)}")
    def plot_data(self):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.set_facecolor(self.colors['background'])

        # Plot voltage (plot1)
        self.ax1.plot(self.time_data, self.voltage_data, color=self.colors['voltage'], linewidth=1.5)
        self.ax1.set_ylabel('Voltage (V)', fontsize=14, fontweight='bold')
        self.ax1.tick_params(axis='both', which='major', labelsize=12)
        self.ax1.tick_params(axis='x', which='both', labelbottom=False, bottom=True)
        self.ax1.set_xlim(self.time_data[0], self.time_data[-1])

        # Plot current or conductance (plot2)
        if self.plot_type == 'current':
            data_to_plot = -self.current_data if self.current_inverted else self.current_data
            ylabel = 'Current (nA)'
            color = self.colors['current']
        else:
            data_to_plot = self.conductance_data
            ylabel = 'Conductance (nS)'
            color = self.colors['conductance']

        self.line2, = self.ax2.plot(self.time_data, data_to_plot, color=color, linewidth=1.5)
        self.ax2.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        self.ax2.tick_params(axis='both', which='major', labelsize=12)
        self.ax2.set_xlim(self.time_data[0], self.time_data[-1])

        # Set initial time range for zoom
        time_range = self.time_data[-1] - self.time_data[0]
        initial_zoom = time_range * 0.1  # 10% of total time range
        zoom_start = self.time_data[0] + time_range * 0.45
        zoom_end = zoom_start + initial_zoom

        self.time_min_input.setText(f"{zoom_start:.2f}")
        self.time_max_input.setText(f"{zoom_end:.2f}")

        self.update_plots()
    def get_current_data_to_plot(self):
            if self.plot_type == 'current':
                return -self.current_data if self.current_inverted else self.current_data
            else:  # conductance
                return self.conductance_data
    def update_plots(self):
        if self.is_updating or self.time_data is None:
            return
        self.is_updating = True
        try:
            t_min = float(self.time_min_input.text())
            t_max = float(self.time_max_input.text())

            # Clear previous plots
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()

            data_to_plot = self.get_current_data_to_plot()
            ylabel = 'Current (nA)' if self.plot_type == 'current' else 'Conductance (nS)'
            color = self.colors[self.plot_type]

            # Plot voltage (plot1)
            self.ax1.plot(self.time_data, self.voltage_data, color=self.colors['voltage'], linewidth=1.5)
            self.ax1.set_ylabel('Voltage (V)', fontsize=14, fontweight='bold')
            self.ax1.tick_params(axis='both', which='major', labelsize=12)
            self.ax1.tick_params(axis='x', which='both', labelbottom=False)

            # Plot current or conductance (plot2)
            self.ax2.plot(self.time_data, data_to_plot, color=color, linewidth=1.5)
            self.ax2.set_ylabel(ylabel, fontsize=14, fontweight='bold')
            self.ax2.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
            self.ax2.tick_params(axis='both', which='major', labelsize=12)

            # Set x-axis limits for plot1 and plot2
            self.ax1.set_xlim(self.time_data[0], self.time_data[-1])
            self.ax2.set_xlim(self.time_data[0], self.time_data[-1])

            # Update plot2 highlight
            if hasattr(self, 'highlight_rect') and self.highlight_rect:
                self.highlight_rect.remove()
            self.highlight_rect = Rectangle((t_min, self.ax2.get_ylim()[0]), t_max - t_min,
                                            self.ax2.get_ylim()[1] - self.ax2.get_ylim()[0],
                                            facecolor=self.colors['highlight'], alpha=self.highlight_alpha)
            self.ax2.add_patch(self.highlight_rect)

            # Update plot3 and plot4
            self.update_plot3(t_min, t_max)
            self.update_plot4(t_min, t_max)

            # Update the canvas
            self.canvas.draw_idle()

            # Update slider position
            highlight_center = (t_min + t_max) / 2
            slider_value = int(((highlight_center - self.time_data[0]) / (self.time_data[-1] - self.time_data[0])) * 1000)
            self.highlight_slider.blockSignals(True)
            self.highlight_slider.setValue(slider_value)
            self.highlight_slider.blockSignals(False)

        finally:
            self.is_updating = False

    def update_plot3(self, t_min, t_max):
        self.ax3.clear()
        mask = (self.time_data >= t_min) & (self.time_data <= t_max)
        data_to_plot = self.get_current_data_to_plot()
        
        self.ax3.plot(self.time_data[mask], data_to_plot[mask], color=self.colors[self.plot_type], linewidth=1.5)
        self.ax3.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        self.ax3.set_ylabel('Current (nA)' if self.plot_type == 'current' else 'Conductance (nS)', 
                            fontsize=14, fontweight='bold')
        self.ax3.tick_params(axis='both', which='major', labelsize=12)
        self.ax3.set_xlim(t_min, t_max)

    def update_plot4(self, t_min, t_max):
        self.ax4.clear()
        mask = (self.time_data >= t_min) & (self.time_data <= t_max)
        data_to_plot = self.get_current_data_to_plot()
        zoomed_data = data_to_plot[mask]
        
        bins = int(self.bins_input.text())
        counts, bin_edges, _ = self.ax4.hist(zoomed_data, bins=bins, orientation='horizontal',
                                             color=self.colors[self.plot_type], alpha=0.7)
        self.ax4.set_xlabel('Count', fontsize=14, fontweight='bold')
        self.ax4.tick_params(axis='both', which='major', labelsize=12)
        self.ax4.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        self.ax4.tick_params(axis='y', which='both', left=True, right=False, labelleft=False)

        max_count = max(counts) if len(counts) > 0 else 1
        self.ax4.set_xlim(0, max_count * 1.1)
        self.ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self.ax4.tick_params(axis='x', rotation=45)

        self.ax4.set_ylim(self.ax3.get_ylim())

    def invert_current(self):
        if self.current_data is not None:
            self.current_inverted = not self.current_inverted
            self.plot_data()

    def show_histogram(self):
        try:
            if self.plot_type == 'current' and self.current_data is not None:
                data = self.current_data
            elif self.plot_type == 'conductance' and self.conductance_data is not None:
                data = self.conductance_data
            else:
                QMessageBox.warning(self, "Warning", f"No {self.plot_type} data available.")
                return

            if not hasattr(self, 'hist_window') or not self.hist_window.isVisible():
                self.hist_window = HistogramWindow(self.time_data, data, self.current_inverted,
                                                          self.plot_type, self)
            else:
                self.hist_window.update_data(self.time_data, data, self.current_inverted, self.plot_type)
            self.hist_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while showing the histogram: {str(e)}")
    
    def load_notes(self, file_path):
        try:
            with h5py.File(file_path, 'r') as f:
                if 'metadata/notes' in f:
                    notes = f['metadata/notes'][0]
                    # Decode if it's bytes, otherwise use as is
                    if isinstance(notes, bytes):
                        notes = notes.decode('utf-8')
                    self.notes_text.setPlainText(notes)
                else:
                    self.notes_text.clear()
        except Exception as e:
            logging.error(f"Error loading notes from {file_path}: {str(e)}")
            self.notes_text.clear()

    def save_notes(self):
        if self.current_file:
            try:
                notes = self.notes_text.toPlainText()

                with h5py.File(self.current_file, 'r+') as f:
                    if 'metadata' not in f:
                        f.create_group('metadata')
                    if 'metadata/notes' in f:
                        del f['metadata/notes']

                    # Create a variable-length string dataset
                    dt = h5py.special_dtype(vlen=str)
                    dataset = f.create_dataset('metadata/notes', (1,), dtype=dt)
                    dataset[0] = notes

                logging.info(f"Notes saved for {self.current_file}")
                QMessageBox.information(self, "Success", "Notes saved successfully.")

                # Refresh the file info display
                self.display_h5_info(self.current_file)
            except Exception as e:
                logging.error(f"Error saving notes to {self.current_file}: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to save notes: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "No file selected. Please select an H5 file first.")

    def show_derivative(self):
        try:
            if self.plot_type == 'current':
                data = self.current_data
            else:  # conductance
                data = self.conductance_data

            if data is None:
                QMessageBox.warning(self, "Warning", f"No {self.plot_type} data available.")
                return

            self.derivative_window = DerivativeWindow(
                self.time_data, 
                self.voltage_data, 
                data,
                self.current_inverted,
                self.plot_type,
                self
            )
            self.derivative_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                f"An error occurred while showing the {self.plot_type.capitalize()} Derivative window: {str(e)}")

class HistogramWindow(QMainWindow):
    def __init__(self, time_data, data, is_inverted=False, plot_type='current', parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{plot_type.capitalize()} Analysis")
        self.setGeometry(100, 100, 1600, 900)

        self.time_data = time_data
        self.data = data
        self.is_inverted = is_inverted
        self.plot_type = plot_type
        self.filtered_time = None
        self.filtered_data = None
        self.peaks = []

        # Define color scheme and font sizes
        self.colors = {
            'background': '#FFFFFF',
            'voltage': '#0e38b1',
            'current': '#64af17',
            'conductance': '#fb6b05',
            'highlight': '#fdff50',
            'peak_line': '#0c00ff'  # New color for peak annotation lines
        }
        self.font_sizes = {
            'title': 18,
            'axis_label': 14,
            'tick_label': 12,
        }

        self.setup_ui()
        self.setup_logging()
        self.update_plot()

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        splitter.addWidget(left_scroll)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        splitter.addWidget(right_widget)

        splitter.setSizes([400, 1200])

        self.setup_left_controls(left_layout)
        self.setup_right_plots(right_layout)

        # Update labels based on plot_type
        if self.plot_type == 'current':
            self.setWindowTitle("Current Analysis")
            ylabel = 'Current (nA)'
        else:
            self.setWindowTitle("Conductance Analysis")
            ylabel = 'Conductance (nS)'

        # Update plot labels
        self.ax_hist.set_ylabel(ylabel, fontsize=self.font_sizes['axis_label'], fontweight='bold')

    def setup_left_controls(self, layout):
        # Time range controls
        time_group = QGroupBox("Time Range")
        time_layout = QFormLayout()
        self.min_time_input = QLineEdit()
        self.max_time_input = QLineEdit()
        time_layout.addRow("Min (s):", self.min_time_input)
        time_layout.addRow("Max (s):", self.max_time_input)
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # Histogram controls
        hist_group = QGroupBox("Histogram Settings")
        hist_layout = QVBoxLayout()
        bins_layout = QFormLayout()
        self.bins_input = QLineEdit("50")
        bins_layout.addRow("Bins:", self.bins_input)
        hist_layout.addLayout(bins_layout)
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.update_plot)
        hist_layout.addWidget(self.apply_button)
        hist_group.setLayout(hist_layout)
        layout.addWidget(hist_group)

        # Peak detection controls
        peak_group = QGroupBox("Peak Detection")
        peak_layout = QVBoxLayout()
        params_layout = QFormLayout()
        self.prominence_input = QLineEdit("0.1")
        self.width_input = QLineEdit("5")
        self.distance_input = QLineEdit("1")
        self.error_bar_input = QLineEdit("5")  # New input for error bar range
        params_layout.addRow("Prominence:", self.prominence_input)
        params_layout.addRow("Width:", self.width_input)
        params_layout.addRow("Distance:", self.distance_input)
        params_layout.addRow("Error Bar Range (%):", self.error_bar_input)
        peak_layout.addLayout(params_layout)
        self.auto_peak_button = QPushButton("Auto Find Peaks")
        self.auto_peak_button.clicked.connect(self.auto_find_peaks)
        peak_layout.addWidget(self.auto_peak_button)
        self.manual_peak_button = QPushButton("Manual Select Peaks")
        self.manual_peak_button.clicked.connect(self.manual_select_peaks)
        peak_layout.addWidget(self.manual_peak_button)

        # Create checkbox layout
        checkbox_layout = QHBoxLayout()
        self.use_click_value_checkbox = QCheckBox("Use clicked value as peak")
        self.use_click_value_checkbox.setChecked(True)  # Default checked
        checkbox_layout.addWidget(self.use_click_value_checkbox)

        # Add new checkbox for showing peak differences
        self.show_peak_diff_checkbox = QCheckBox("Show peak differences")
        self.show_peak_diff_checkbox.setChecked(True)  # Default checked
        self.show_peak_diff_checkbox.stateChanged.connect(self.update_plot)
        checkbox_layout.addWidget(self.show_peak_diff_checkbox)

        peak_layout.addLayout(checkbox_layout)
        peak_group.setLayout(peak_layout)
        layout.addWidget(peak_group)

        # Peak list
        list_group = QGroupBox("Detected Peaks")
        list_layout = QVBoxLayout()
        self.peak_list = QListWidget()
        self.peak_list.itemClicked.connect(self.on_peak_selected)
        list_layout.addWidget(self.peak_list)

        button_layout = QHBoxLayout()
        self.remove_peak_button = QPushButton("Remove Selected Peak")
        self.remove_peak_button.clicked.connect(self.remove_selected_peak)
        button_layout.addWidget(self.remove_peak_button)

        # Add button for copying to clipboard
        self.copy_peaks_button = QPushButton("Copy Peaks to Clipboard")
        self.copy_peaks_button.clicked.connect(self.copy_peaks_to_clipboard)
        button_layout.addWidget(self.copy_peaks_button)

        list_layout.addLayout(button_layout)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)

        layout.addStretch(1)

    def setup_right_plots(self, layout):
        self.figure = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create subplot with shared y-axis
        gs = self.figure.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.02)
        self.ax_current = self.figure.add_subplot(gs[0, 0])
        self.ax_hist = self.figure.add_subplot(gs[0, 1], sharey=self.ax_current)

        # Remove y-axis labels from histogram
        self.ax_hist.yaxis.set_ticklabels([])

        # Adjust the layout manually
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

        # Add Matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def setup_input_fields(self):
        min_time = np.min(self.time_data)
        max_time = np.max(self.time_data)
        min_current = np.min(self.current_data)
        max_current = np.max(self.current_data)

        self.min_time_input.setText(f"{min_time:.2f}")
        self.max_time_input.setText(f"{max_time:.2f}")
        self.min_current_input.setText(f"{min_current:.2f}")
        self.max_current_input.setText(f"{max_current:.2f}")

        # Use QDoubleValidator for all inputs
        double_validator = QDoubleValidator()
        self.min_time_input.setValidator(double_validator)
        self.max_time_input.setValidator(double_validator)
        self.min_current_input.setValidator(double_validator)
        self.max_current_input.setValidator(double_validator)

    def get_time_range(self):
        try:
            t_min = float(self.min_time_input.text()) if self.min_time_input.text() else np.min(self.time_data)
            t_max = float(self.max_time_input.text()) if self.max_time_input.text() else np.max(self.time_data)
            return t_min, t_max
        except ValueError:
            return np.min(self.time_data), np.max(self.time_data)

    def get_current_range(self):
        try:
            min_val = float(self.min_current_input.text())
            max_val = float(self.max_current_input.text())
            return min_val, max_val
        except ValueError:
            return np.min(self.current_data), np.max(self.current_data)

    def plot_current_and_histogram(self):
        try:
            self.ax_current.clear()
            self.ax_hist.clear()

            # Plot current data
            plot_color = self.colors['current'] if self.plot_type == 'current' else self.colors['conductance']
            self.ax_current.plot(self.time_data, self.current_data, color=plot_color, linewidth=1)
            self.ax_current.set_xlabel('Time (s)', fontsize=self.font_sizes['axis_label'], fontweight='bold')
            self.ax_current.set_ylabel('Current (nA)', fontsize=self.font_sizes['axis_label'], fontweight='bold')
            self.ax_current.tick_params(axis='both', which='major', labelsize=self.font_sizes['tick_label'])

            # Remove extra space on left and right of the current plot
            self.ax_current.set_xlim(self.time_data.min(), self.time_data.max())

            # Plot histogram
            bins = int(self.bins_input.text()) if self.bins_input.text() else 50
            self.counts, self.bins, _ = self.ax_hist.hist(self.current_data, bins=bins,
                                                          orientation='horizontal', color=plot_color, alpha=0.7)
            self.ax_hist.set_xlabel('Count', fontsize=self.font_sizes['axis_label'], fontweight='bold')
            self.ax_hist.tick_params(axis='x', which='major', labelsize=self.font_sizes['tick_label'])

            # Add ticks to y-axis of histogram without labels
            self.ax_hist.tick_params(axis='y', which='both', left=True, labelleft=False)

            # Ensure y-axis limits match the current plot
            self.ax_hist.set_ylim(self.ax_current.get_ylim())

            # Set background color
            self.ax_current.set_facecolor(self.colors['background'])
            self.ax_hist.set_facecolor(self.colors['background'])

            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            logging.error(f"Error in plot_current_and_histogram: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred while plotting: {str(e)}")

    def auto_find_peaks(self):
        try:
            prominence = float(self.prominence_input.text()) * np.max(self.counts)
            width = float(self.width_input.text())
            distance = float(self.distance_input.text())

            peak_indices, properties = find_peaks(self.counts,
                                                  prominence=prominence,
                                                  width=width,
                                                  distance=distance)

            self.peaks = []
            for i, peak_index in enumerate(peak_indices):
                peak_current = self.bins[peak_index]
                peak_height = properties['prominences'][i]
                peak_width = properties['widths'][i]
                # Calculate error based on peak width
                error = peak_width * (self.bins[1] - self.bins[0]) / 2
                self.peaks.append((peak_current, peak_height, peak_width, error))

            self.update_peak_list()
            self.plot_peaks()
        except Exception as e:
            logging.error(f"Error in auto_find_peaks: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred during peak finding: {str(e)}")

    def add_peak_detection_controls(self, parent_layout):
        peak_params_group = QGroupBox("Peak Detection Parameters")
        peak_params_layout = QFormLayout()

        self.prominence_input = QLineEdit("0.1")
        self.prominence_input.setValidator(QDoubleValidator(0, 1, 4))
        peak_params_layout.addRow("Prominence (0-1):", self.prominence_input)

        self.width_input = QLineEdit("5")
        self.width_input.setValidator(QDoubleValidator(0, 1000, 2))
        peak_params_layout.addRow("Width:", self.width_input)

        self.distance_input = QLineEdit("1")
        self.distance_input.setValidator(QDoubleValidator(0, 1000, 2))
        peak_params_layout.addRow("Distance:", self.distance_input)

        peak_params_group.setLayout(peak_params_layout)
        parent_layout.addWidget(peak_params_group)

    def update_peak_list(self):
            self.peak_list.clear()
            show_diff = self.show_peak_diff_checkbox.isChecked()
            unit = 'nA' if self.plot_type == 'current' else 'nS'

            for i, (peak, height, width, error) in enumerate(self.peaks):
                item_text = f"Peak {i + 1}: {peak:.4f} ± {error:.4f} {unit} (Height: {height:.2f}, Width: {width:.2f})"
                if show_diff and i < len(self.peaks) - 1:
                    diff = self.peaks[i + 1][0] - peak
                    item_text += f", Diff to next: {diff:.3f} {unit}"
                item = QListWidgetItem(item_text)
                self.peak_list.addItem(item)

    def on_peak_selected(self, item):
        index = self.peak_list.row(item)
        self.plot_peaks()  # Highlight the selected peak

    def remove_selected_peak(self):
        selected_items = self.peak_list.selectedItems()
        if selected_items:
            index = self.peak_list.row(selected_items[0])
            del self.peaks[index]
            self.update_peak_list()
            self.plot_peaks()

    def manual_select_peaks(self):
        if self.use_click_value_checkbox.isChecked():
            QMessageBox.information(self, "Manual Peak Selection",
                                    "Click on the histogram to select peak positions. The clicked value will be used directly as the peak. Press 'Enter' when finished.")
        else:
            QMessageBox.information(self, "Manual Peak Selection",
                                    "Click on the histogram to select peak positions. Custom parameters will be used for peak fitting. Press 'Enter' when finished.")
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('key_press_event', self.on_key)

    def on_click(self, event):
        if event.inaxes == self.ax_hist:
            selected_current = event.ydata
            if self.use_click_value_checkbox.isChecked():
                self.add_peak_directly(selected_current)
            else:
                self.analyze_peak(selected_current)

    def add_peak_directly(self, peak_value):
        # 找到选定峰值落在哪两个已知峰之间
        left_peak = None
        right_peak = None
        for peak in self.peaks:
            if peak[0] < peak_value:
                left_peak = peak
            elif peak[0] > peak_value:
                right_peak = peak
                break

        # 计算L1和L2
        if left_peak:
            L1 = left_peak[0] + left_peak[3]  # 峰值 + error bar
        else:
            L1 = self.bins[0]

        if right_peak:
            L2 = right_peak[0] - right_peak[3]  # 峰值 - error bar
        else:
            L2 = self.bins[-1]

        print(f"L1 (lower limit): {L1:.4f}")
        print(f"L2 (upper limit): {L2:.4f}")

        # 在L1和L2之间选择直方图数据
        mask = (self.bins[:-1] >= L1) & (self.bins[:-1] <= L2)
        x_data = self.bins[:-1][mask]
        y_data = self.counts[mask]

        if len(x_data) < 2:
            print("Not enough data points between L1 and L2.")
            return

        # 计算峰值高度
        peak_height = np.interp(peak_value, x_data, y_data)

        # 使用用户输入的 error bar 范围
        try:
            error_percentage = float(self.error_bar_input.text()) / 100
            error = (L2 - L1) * error_percentage
        except ValueError:
            print("Invalid error bar range input. Using default.")
            error = (L2 - L1) * 0.05  # 默认使用5%

        new_peak = (peak_value, peak_height, error * 2, error)
        self.peaks.append(new_peak)
        self.peaks.sort(key=lambda x: x[0])
        unit = 'nA' if self.plot_type == 'current' else 'nS'
        print(f"Added new peak directly: {peak_value:.4f} ± {error:.4f} {unit}")

        self.update_peak_list()
        self.plot_peaks()

        # 绘制新添加的峰
        self.ax_hist.axhline(y=peak_value, color='r', linestyle='--', linewidth=1)
        self.ax_hist.errorbar(peak_height, peak_value, xerr=None, yerr=error, fmt='r.', capsize=5)
        self.canvas.draw()

    def analyze_peak(self, selected_current):
        # 找到选定峰值落在哪两个已知峰之间
        left_peak = None
        right_peak = None
        left_index = right_index = -1
        for i, peak in enumerate(self.peaks):
            if peak[0] < selected_current:
                left_peak = peak
                left_index = i
            elif peak[0] > selected_current:
                right_peak = peak
                right_index = i
                break

        print(f"Selected value falls between peaks {left_index + 1} and {right_index + 1}")

        # 计算拟合范围
        if left_peak:
            L1 = left_peak[0] + left_peak[3]  # 峰值 + error bar
            print(f"L1 (lower limit): {L1:.4f}")
        else:
            L1 = self.bins[0]
            print(f"L1 (lower limit): {L1:.4f} (using minimum bin value)")

        if right_peak:
            L2 = right_peak[0] - right_peak[3]  # 峰值 - error bar
            print(f"L2 (upper limit): {L2:.4f}")
        else:
            L2 = self.bins[-1]
            print(f"L2 (upper limit): {L2:.4f} (using maximum bin value)")

        # 在L1和L2之间选择直方图数据
        mask = (self.bins[:-1] >= L1) & (self.bins[:-1] <= L2)
        x_data = self.bins[:-1][mask]
        y_data = self.counts[mask]

        # 找出L1和L2之间的最高点
        max_index = np.argmax(y_data)
        max_current = x_data[max_index]
        max_count = y_data[max_index]
        print(f"Highest point between L1 and L2: Current = {max_current:.4f} nA, Count = {max_count}")

        # 使用高斯函数进行拟合
        def gaussian(x, amp, cen, wid):
            return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

        try:
            # 使用peak detection的参数
            prominence = float(self.prominence_input.text()) * np.max(self.counts)
            width = float(self.width_input.text())

            # 设置初始参数
            initial_amp = max_count
            initial_cen = max_current
            initial_wid = min((L2 - L1) / 4, width)  # 使用较小的值作为初始宽度

            # 设置参数边界
            lower_bounds = [0, L1, 0]  # 振幅>0, 中心>L1, 宽度>0
            upper_bounds = [np.inf, L2, L2 - L1]  # 振幅无上限, 中心<L2, 宽度<范围

            # 尝试不同的初始参数和边界条件
            try_fits = [
                (initial_amp, initial_cen, initial_wid),
                (max_count / 2, (L1 + L2) / 2, (L2 - L1) / 4),
                (max_count * 2, selected_current, (L2 - L1) / 8)
            ]

            best_fit = None
            best_error = np.inf

            for init_params in try_fits:
                try:
                    popt, pcov = curve_fit(gaussian, x_data, y_data,
                                           p0=init_params,
                                           bounds=(lower_bounds, upper_bounds))

                    amp, cen, wid = popt
                    errors = np.sqrt(np.diag(pcov))
                    error = errors[1]  # 中心位置的误差

                    if error < best_error and L1 <= cen <= L2:
                        best_fit = (popt, pcov)
                        best_error = error

                except RuntimeError:
                    continue

            if best_fit is None:
                print("Failed to find a suitable fit.")
                return

            popt, pcov = best_fit
            amp, cen, wid = popt
            errors = np.sqrt(np.diag(pcov))
            error = errors[1]  # 中心位置的误差

            new_peak = (cen, amp, wid, error)
            print(f"New peak found: {cen:.4f} ± {error:.4f} nA")

            # 检查是否与现有峰重叠
            overlapping_peak = next((p for p in self.peaks if abs(p[0] - cen) < max(p[3], error)), None)
            if overlapping_peak:
                self.peaks[self.peaks.index(overlapping_peak)] = new_peak
                print("Updated existing overlapping peak")
            else:
                self.peaks.append(new_peak)
                print("Added new peak")

            self.peaks.sort(key=lambda x: x[0])  # 按电流值排序

            # 绘制拟合结果
            x_fit = np.linspace(L1, L2, 100)
            y_fit = gaussian(x_fit, *popt)
            self.ax_hist.plot(y_fit, x_fit, 'r--', linewidth=2)
            self.canvas.draw()

            self.update_peak_list()
            self.plot_peaks()

        except Exception as e:
            print(f"Failed to fit peak: {str(e)}")
            logging.error(f"Failed to fit peak at {selected_current}: {str(e)}")

    def on_key(self, event):
        if event.key == 'enter':
            self.canvas.mpl_disconnect('button_press_event')
            self.canvas.mpl_disconnect('key_press_event')
            QMessageBox.information(self, "Manual Peak Selection", "Peak selection completed.")

    def plot_peaks(self):
        self.update_plot()  # 重绘主图
        show_diff = self.show_peak_diff_checkbox.isChecked()
        unit = 'nA' if self.plot_type == 'current' else 'nS'

        for i, (peak, height, width, error) in enumerate(self.peaks):
            self.ax_hist.axhline(y=peak, color=self.colors['peak_line'], linestyle='--', linewidth=1)
            self.ax_hist.text(0, peak, f'{i + 1}', color='r', ha='right', va='center')
            # 添加误差条
            self.ax_hist.errorbar(height, peak, xerr=None, yerr=error, fmt='r.', capsize=5)

            # 在相邻peak之间添加差值标记
            if show_diff and i < len(self.peaks) - 1:
                next_peak = self.peaks[i + 1][0]
                mid_point = (peak + next_peak) / 2
                diff = next_peak - peak
                # 在两个peak之间的中点添加文本标记
                self.ax_hist.text(height, mid_point, f'{diff:.3f} {unit}', color='blue',
                                  ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        self.canvas.draw()

    def fit_peaks(self):
        # Fit Gaussian mixture to the peaks
        def gaussian(x, amp, cen, wid):
            return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

        def multi_gaussian(x, *params):
            return np.sum([gaussian(x, *params[i:i + 3]) for i in range(0, len(params), 3)], axis=0)

        initial_guess = []
        for peak in self.peaks:
            initial_guess.extend([np.max(self.counts), peak, 1])

        try:
            popt, _ = curve_fit(multi_gaussian, self.bins[:-1], self.counts, p0=initial_guess)

            x = np.linspace(np.min(self.bins), np.max(self.bins), 1000)
            self.ax_hist.plot(multi_gaussian(x, *popt), x, 'r-', linewidth=2)

            # Display fitted peak positions
            for i in range(0, len(popt), 3):
                self.ax_hist.axhline(y=popt[i + 1], color='g', linestyle=':')
                self.ax_hist.text(0, popt[i + 1], f'{popt[i + 1]:.2f} nA',
                                  verticalalignment='center', horizontalalignment='left')

            self.canvas.draw()

            QMessageBox.information(self, "Peak Fitting", "Peak fitting completed successfully.")
        except RuntimeError:
            QMessageBox.warning(self, "Peak Fitting", "Failed to fit peaks. Try selecting different initial positions.")

    def copy_peaks_to_clipboard(self):
        if not self.peaks:
            QMessageBox.information(self, "Copy Peaks", "No peaks to copy.")
            return

        unit = 'nA' if self.plot_type == 'current' else 'nS'
        # 创建一个字符串来存储所有的peak信息
        peaks_text = f"Peak\t{self.plot_type.capitalize()} ({unit})\tHeight\tWidth\tError\n"
        for i, (peak, height, width, error) in enumerate(self.peaks):
            peaks_text += f"{i + 1}\t{peak:.4f}\t{height:.2f}\t{width:.2f}\t{error:.4f}\n"

        # 如果有多于一个peak,添加相邻peak之间的差值
        if len(self.peaks) > 1:
            peaks_text += f"\nDifferences between adjacent peaks ({unit}):\n"
            for i in range(len(self.peaks) - 1):
                diff = self.peaks[i + 1][0] - self.peaks[i][0]
                peaks_text += f"Peak {i + 1} to Peak {i + 2}: {diff:.4f} {unit}\n"

        # 复制到剪贴板
        clipboard = QApplication.clipboard()
        clipboard.setText(peaks_text)

        QMessageBox.information(self, "Copy Peaks", "Peak information has been copied to the clipboard.")

    def update_plot(self):
        try:
            self.ax_current.clear()
            self.ax_hist.clear()

            data_to_plot = -self.data if self.is_inverted and self.plot_type == 'current' else self.data

            t_min, t_max = self.get_time_range()
            mask = (self.time_data >= t_min) & (self.time_data <= t_max)
            self.filtered_time = self.time_data[mask]
            self.filtered_data = data_to_plot[mask]

            # Plot data
            plot_color = self.colors['current'] if self.plot_type == 'current' else self.colors['conductance']
            self.ax_current.plot(self.filtered_time, self.filtered_data, color=plot_color, linewidth=1)
            self.ax_current.set_xlabel('Time (s)', fontsize=self.font_sizes['axis_label'], fontweight='bold')
            if self.plot_type == 'current':
                self.ax_current.set_ylabel('Current (nA)', fontsize=self.font_sizes['axis_label'], fontweight='bold')
            else:
                self.ax_current.set_ylabel('Conductance (nS)', fontsize=self.font_sizes['axis_label'], fontweight='bold')
            self.ax_current.tick_params(axis='both', which='major', labelsize=self.font_sizes['tick_label'])

            # Remove extra space on left and right of the current plot
            self.ax_current.set_xlim(self.filtered_time.min(), self.filtered_time.max())

            # Plot histogram
            bins = int(self.bins_input.text()) if self.bins_input.text() else 50
            self.counts, self.bins, _ = self.ax_hist.hist(self.filtered_data, bins=bins,
                                                          orientation='horizontal', color=plot_color,
                                                          alpha=0.7)
            self.ax_hist.set_xlabel('Count', fontsize=self.font_sizes['axis_label'], fontweight='bold')
            self.ax_hist.tick_params(axis='x', which='major', labelsize=self.font_sizes['tick_label'])

            # Add ticks to y-axis of histogram without labels
            self.ax_hist.tick_params(axis='y', which='both', left=True, right=True, labelleft=False)

            # Set background color
            self.ax_current.set_facecolor(self.colors['background'])
            self.ax_hist.set_facecolor(self.colors['background'])

            # Draw the canvas
            self.canvas.draw()

        except Exception as e:
            logging.error(f"Error in update_plot: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred while plotting: {str(e)}")

    def update_data(self, time_data, data, is_inverted, plot_type):
        self.time_data = time_data
        self.data = data
        self.is_inverted = is_inverted
        self.plot_type = plot_type
        self.update_plot()

    def closeEvent(self, event):
        # Override the close event to hide the window instead of closing it
        event.ignore()
        self.hide()

class DerivativeWindow(QMainWindow):
    def __init__(self, time_data, voltage_data, data, is_inverted, plot_type, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{plot_type.capitalize()} Derivative")
        self.setGeometry(100, 100, 1200, 800)

        self.time_data, self.voltage_data, self.data = self.preprocess_data(time_data, voltage_data, data)
        self.is_inverted = is_inverted
        self.plot_type = plot_type
        self.fitted_data = None
        self.derivatives = None
        self.segment_times = None
        self.plot3_ymin = None
        self.plot3_ymax = None

        self.colors = {
            'voltage': '#0e38b1',
            'current': '#64af17',
            'conductance': '#fb6b05',
            'fitted': '#e54b4b',
            'derivative': '#b7e3e4',
            'moving_avg': '#f03f35',
            'fitted_conductance': '#053161'  # 新增这一行
        }

        self.setup_ui()
        self.plot_initial_data()


    def preprocess_data(self, time_data, voltage_data, data):
        min_length = min(len(time_data), len(voltage_data), len(data))
        time_data = time_data[:min_length]
        voltage_data = voltage_data[:min_length]
        data = data[:min_length]

        if not np.all(np.diff(time_data) > 0):
            sorted_indices = np.argsort(time_data)
            time_data = time_data[sorted_indices]
            voltage_data = voltage_data[sorted_indices]
            data = data[sorted_indices]

        return time_data, voltage_data, data

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setCentralWidget(main_widget)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_widget.setFixedWidth(300)

        self.add_left_controls(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)

        # Adjust subplot parameters to remove empty space
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, hspace=0.3)

        # Update labels based on plot_type
        if self.plot_type == 'current':
            self.setWindowTitle("Current Derivative")
            ylabel = 'Current (nA)'
        else:
            self.setWindowTitle("Conductance Derivative")
            ylabel = 'Conductance (nS)'

        # Update plot labels
        self.ax2.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        self.ax3.set_ylabel(f'|{self.plot_type.capitalize()} Derivative| ({ylabel[:-4]}/s)', fontsize=12, fontweight='bold')

    def add_left_controls(self, layout):
        # Plot Time Range
        plot_group = QGroupBox("Plot Time Range")
        plot_layout = QFormLayout(plot_group)
        self.plot_min = QLineEdit(str(self.time_data[0]))
        self.plot_max = QLineEdit(str(self.time_data[-1]))
        plot_layout.addRow("Min:", self.plot_min)
        plot_layout.addRow("Max:", self.plot_max)
        layout.addWidget(plot_group)

        # Fit Interval
        fit_group = QGroupBox("Fit Interval")
        fit_layout = QFormLayout(fit_group)
        self.fit_interval = QLineEdit("0.05")  # Default to 50 ms
        fit_layout.addRow("Interval (s):", self.fit_interval)
        layout.addWidget(fit_group)

        # 添加多项式阶数控件
        poly_group = QGroupBox("Polynomial Fitting")
        poly_layout = QFormLayout(poly_group)
        self.poly_degree = QLineEdit("3")  # 默认为3阶多项式
        poly_layout.addRow("Degree:", self.poly_degree)
        layout.addWidget(poly_group)

        # Moving Average
        ma_group = QGroupBox("Moving Average")
        ma_layout = QFormLayout(ma_group)
        self.ma_window_input = QLineEdit("10")  # Default window size
        ma_layout.addRow("Window Size:", self.ma_window_input)
        layout.addWidget(ma_group)

        # Plot3 Y-Axis Range Control
        y_axis_group = QGroupBox("Plot3 Y-Axis Range")
        y_axis_layout = QFormLayout(y_axis_group)
        self.y_min_input = QLineEdit()
        self.y_max_input = QLineEdit()
        y_axis_layout.addRow("Min:", self.y_min_input)
        y_axis_layout.addRow("Max:", self.y_max_input)
        layout.addWidget(y_axis_group)

        # Invert Current Checkbox
        self.invert_checkbox = QCheckBox("Invert Current")
        self.invert_checkbox.setChecked(self.is_inverted)
        self.invert_checkbox.stateChanged.connect(self.toggle_invert_current)
        layout.addWidget(self.invert_checkbox)

        # Buttons
        self.fit_button = QPushButton("Fit and Update")
        self.fit_button.clicked.connect(self.perform_fit)
        layout.addWidget(self.fit_button)

        self.apply_y_range_button = QPushButton("Apply Y-Axis Range")
        self.apply_y_range_button.clicked.connect(self.apply_y_axis_range)
        layout.addWidget(self.apply_y_range_button)

        layout.addStretch(1)

    def toggle_invert_current(self, state):
        self.is_inverted = (state == Qt.CheckState.Checked.value)
        self.plot_initial_data()

    def plot_initial_data(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax1.plot(self.time_data, self.voltage_data, color=self.colors['voltage'])
        self.ax1.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')

        data_to_plot = -self.data if self.is_inverted and self.plot_type == 'current' else self.data
        data_color = self.colors['current'] if self.plot_type == 'current' else self.colors['conductance']
        self.ax2.plot(self.time_data, data_to_plot, color=data_color)
        if self.plot_type == 'current':
            self.ax2.set_ylabel('Current (nA)', fontsize=12, fontweight='bold')
        else:
            self.ax2.set_ylabel('Conductance (nS)', fontsize=12, fontweight='bold')

        if self.plot_type == 'current':
            self.ax3.set_ylabel('|Current Derivative| (nA/s)', fontsize=12, fontweight='bold')
        else:
            self.ax3.set_ylabel('|Conductance Derivative| (nS/s)', fontsize=12, fontweight='bold')
        self.ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')

        # Set x-axis limits to remove empty space
        self.ax1.set_xlim(self.time_data[0], self.time_data[-1])

        # Remove grid lines
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.grid(False)

        self.canvas.draw()

    def apply_y_axis_range(self):
        try:
            self.plot3_ymin = float(self.y_min_input.text()) if self.y_min_input.text() else None
            self.plot3_ymax = float(self.y_max_input.text()) if self.y_max_input.text() else None
            self.update_plot3()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for Y-axis range.")

    def update_plot3(self):
        if self.derivatives is None:
            return

        self.ax3.clear()

        valid_mask = ~np.isnan(self.derivatives)
        self.ax3.plot(np.array(self.segment_times)[valid_mask], np.array(self.derivatives)[valid_mask],
                      color=self.colors['derivative'], label='|Derivative|')

        if hasattr(self, 'ma_derivatives') and hasattr(self, 'ma_times'):
            self.ax3.plot(self.ma_times, self.ma_derivatives, color=self.colors['moving_avg'],
                          label=f'Moving Avg (window={self.ma_window})')

        if self.plot_type == 'current':
            self.ax3.set_ylabel('|Current Derivative| (nA/s)', fontsize=12, fontweight='bold')
        else:
            self.ax3.set_ylabel('|Conductance Derivative| (nS/s)', fontsize=12, fontweight='bold')
        self.ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        self.ax3.legend(fontsize=10)

        if self.plot3_ymin is not None and self.plot3_ymax is not None:
            self.ax3.set_ylim(self.plot3_ymin, self.plot3_ymax)
        else:
            self.ax3.autoscale(axis='y')

        # Set x-axis limits to remove empty space
        self.ax3.set_xlim(min(self.segment_times), max(self.segment_times))

        # Remove grid lines
        self.ax3.grid(False)

        self.canvas.draw()

    def perform_fit(self):
        try:
            interval = float(self.fit_interval.text())
            t_min = float(self.plot_min.text())
            t_max = float(self.plot_max.text())
            ma_window = int(self.ma_window_input.text())
            poly_degree = int(self.poly_degree.text())

            mask = (self.time_data >= t_min) & (self.time_data <= t_max)
            time = self.time_data[mask]
            data = -self.data[mask] if self.is_inverted and self.plot_type == 'current' else self.data[mask]

            segments = np.arange(t_min, t_max + interval, interval)
            self.fitted_data = np.zeros_like(data)
            self.derivatives = []
            self.segment_times = []

            for i in range(len(segments) - 1):
                start, end = segments[i], segments[i + 1]
                seg_mask = (time >= start) & (time < end)

                if np.sum(seg_mask) > poly_degree:  # 确保有足够的点进行拟合
                    seg_time = time[seg_mask]
                    seg_data = data[seg_mask]

                    # 多项式拟合
                    coeffs = np.polyfit(seg_time, seg_data, poly_degree)
                    poly = np.poly1d(coeffs)

                    # 计算拟合值
                    self.fitted_data[seg_mask] = poly(seg_time)

                    # 计算导数
                    deriv = np.polyder(poly)
                    mid_point = (start + end) / 2
                    self.derivatives.append(abs(deriv(mid_point)))  # 取绝对值
                    self.segment_times.append(mid_point)
                else:
                    self.derivatives.append(np.nan)
                    self.segment_times.append((start + end) / 2)

            self.update_plots(time, data, ma_window)

        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Error in fitting: {str(e)}")
            print(f"Error in fitting: {str(e)}")

    def update_plots(self, time, data, ma_window):
        self.ax1.clear()
        self.ax2.clear()

        mask = (self.time_data >= time[0]) & (self.time_data <= time[-1])
        self.ax1.plot(self.time_data[mask], self.voltage_data[mask], color=self.colors['voltage'])
        self.ax1.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')

        data_color = self.colors['current'] if self.plot_type == 'current' else self.colors['conductance']
        self.ax2.plot(time, data, color=data_color, label='Original')
        
        # 根据 plot_type 选择 fitted curve 的颜色
        if self.plot_type == 'conductance':
            fitted_color = self.colors['fitted_conductance']
        else:
            fitted_color = self.colors['fitted']
        
        self.ax2.plot(time, self.fitted_data, color=fitted_color, label='Fitted')
        
        if self.plot_type == 'current':
            self.ax2.set_ylabel('Current (nA)', fontsize=12, fontweight='bold')
        else:
            self.ax2.set_ylabel('Conductance (nS)', fontsize=12, fontweight='bold')
        self.ax2.legend(fontsize=10)

        valid_mask = ~np.isnan(self.derivatives)
        self.ma_window = ma_window
        self.ma_derivatives = np.convolve(np.array(self.derivatives)[valid_mask], np.ones(ma_window), 'valid') / ma_window
        self.ma_times = np.array(self.segment_times)[valid_mask][ma_window - 1:]

        # Set x-axis limits to remove empty space
        self.ax1.set_xlim(time[0], time[-1])
        self.ax2.set_xlim(time[0], time[-1])

        # Remove grid lines
        self.ax1.grid(False)
        self.ax2.grid(False)

        self.update_plot3()

        self.canvas.draw()

    
