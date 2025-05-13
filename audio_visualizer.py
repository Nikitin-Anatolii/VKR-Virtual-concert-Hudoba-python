import sys
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QFileDialog, QLabel, QHBoxLayout)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class AudioVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Analysis Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create top controls layout
        controls_layout = QHBoxLayout()
        
        # Create file selection button
        self.file_button = QPushButton("Select Analysis JSON File")
        self.file_button.clicked.connect(self.load_json_file)
        controls_layout.addWidget(self.file_button)
        
        # Create BPM label
        self.bpm_label = QLabel("BPM: --")
        self.bpm_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        controls_layout.addWidget(self.bpm_label)
        
        main_layout.addLayout(controls_layout)
        
        # Create status label
        self.status_label = QLabel("No file loaded")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)
        
        # Initialize subplots
        self.ax1 = self.figure.add_subplot(311)  # Spectral Flux
        self.ax2 = self.figure.add_subplot(312)  # Energy
        self.ax3 = self.figure.add_subplot(313)  # Spectral Centroid
        
        self.figure.tight_layout(pad=3.0)
        
        # Store data
        self.data = None
        
    def load_json_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select JSON File",
            "",
            "JSON Files (*.json)"
        )
        
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    self.data = json.load(f)
                self.status_label.setText(f"Loaded: {file_name}")
                self.bpm_label.setText(f"BPM: {self.data.get('bpm', '--')}")
                self.plot_data()
            except Exception as e:
                self.status_label.setText(f"Error loading file: {str(e)}")
    
    def plot_data(self):
        if not self.data or 'frames' not in self.data:
            return
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Extract data
        frames = self.data['frames']
        time = np.arange(len(frames))
        spectral_flux = [frame['spectral_flux'] for frame in frames]
        energy = [frame['energy'] for frame in frames]
        spectral = [frame['spectral_centroid'] for frame in frames]
        
        # Plot Spectral Flux
        self.ax1.plot(time, spectral_flux, 'b-', label='Spectral Flux')
        self.ax1.set_title('Spectral Flux Over Time')
        self.ax1.set_ylabel('Normalized Flux')
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Plot Energy
        self.ax2.plot(time, energy, 'r-', label='Energy')
        self.ax2.set_title('Energy Over Time')
        self.ax2.set_ylabel('Normalized Energy')
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Plot Spectral Centroid
        self.ax3.plot(time, spectral, 'g-', label='Spectral Centroid')
        self.ax3.set_title('Spectral Centroid Over Time')
        self.ax3.set_xlabel('Time (seconds)')
        self.ax3.set_ylabel('Normalized Spectral Centroid')
        self.ax3.grid(True)
        self.ax3.legend()
        
        # Update canvas
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = AudioVisualizer()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 