# Imports
import os
import napari
from qtpy import uic
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QWidget,
)

# Main Plugin class that is connected from outside at napari plugin entry point
class Lab(QWidget):
    def __init__(self, napari_viewer):
        # Initializing
        super().__init__()
        self.viewer = napari_viewer

        # Load the UI file - Main window
        script_dir = os.path.dirname(__file__)
        ui_file_name = "simple.ui"
        abs_file_path = os.path.join(
            script_dir, "..", "ui", ui_file_name
        )
        uic.loadUi(abs_file_path, self)