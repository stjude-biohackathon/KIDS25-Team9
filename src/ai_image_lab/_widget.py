import os
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from ui.styles import STYLE_SHEET
from ui.main_tab import ProjectSetupTab
from ui.data_processing_tab import DataProcessingTab
from ui.train_tab import TrainTab
from ui.inference_tab import InferenceTab

class Lab(QWidget):
    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self.setObjectName("Root")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(STYLE_SHEET)
        self._build_ui()
        self._wire_events()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("ModernTabs")
        self.tabs.setDocumentMode(True)  # already present, keep it
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)  # pill tabs feel cleaner fixed
        self.tabs.setTabsClosable(False)  # no close buttons needed
        self.tabs.tabBar().setElideMode(QtCore.Qt.ElideNone)  # tidy long labels
        self.tabs.setIconSize(QtCore.QSize(16, 16))  # if you add icons later
        self.tabs.setUsesScrollButtons(True)  # show arrows if overflow
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.tabBar().setExpanding(False)

        # Tabs
        self.tab_project = ProjectSetupTab()
        self.tab_processing = DataProcessingTab()
        self.tab_train = TrainTab()
        self.tab_infer = InferenceTab()

        self.tabs.addTab(self.tab_project, "Project")
        self.tabs.addTab(self.tab_processing, "Data Processing")
        self.tabs.addTab(self.tab_train, "Train / Fine-tune")
        self.tabs.addTab(self.tab_infer, "Inference")

        layout.addWidget(self.tabs)

        # Overall preferred size hint to keep under ~50% of viewer width
        self.setMinimumWidth(380)
        self.setMaximumWidth(560)

    def _wire_events(self):
        # Tab navigation flow
        self.tab_project.continued.connect(self._on_project_continue)
        self.tab_processing.requested_run.connect(self._on_processing_continue)
        self.tab_train.start_training.connect(self._on_train_start)
        self.tab_infer.run_inference.connect(self._on_infer_run)

    # ---- Handlers connecting UI to your backend logic ---- #
    def _on_project_continue(self, payload: dict):
        # Example: you might initialize a project object here
        # and pass it to other tabs as needed
        images = payload.get("images"); labels = payload.get("labels"); task = payload.get("task")
        QtWidgets.QMessageBox.information(self, "Project Setup", f"Images: {images}\nLabels: {labels}\nTask: {task}")
        self.tabs.setCurrentWidget(self.tab_processing)

    def _on_processing_continue(self, payload: dict):
        # Hook into your preprocessing pipeline here
        # Example: self.pipeline.run_preprocessing(**payload)
        self.tabs.setCurrentWidget(self.tab_train)

    def _on_train_start(self, payload: dict):
        # Hook into your trainer here, potentially async in your app
        # Example: self.trainer.start(mode=..., ...)
        self.tabs.setCurrentWidget(self.tab_infer)

    def _on_infer_run(self, payload: dict):
        # Hook into your inference engine here
        # Example: results = self.inferencer.run(**payload)
        pass

    # Helpful for napari when it asks for a preferred size
    def sizeHint(self):
        return QtCore.QSize(520, 720)
