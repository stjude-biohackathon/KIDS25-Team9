import os
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QWidget, QVBoxLayout, QStackedLayout, QTabWidget

from ui.styles import STYLE_SHEET
from ui.main_tab import ProjectSetupTab
from ui.annotation_tab import AnnotationTab
from ui.data_processing_tab import DataProcessingTab
from ui.train_tab import TrainTab
from ui.inference_tab import InferenceTab

MODEL_BUILDING_TASKS = {"semantic-2d", "semantic-3d", "instance", "fine-tune"}
INFERENCE_TASK = "inference"


class ModelBuilderWidget(QtWidgets.QWidget):
    """
    Container that shows tabs AFTER a model-building task is chosen:
      Tabs: Annotation, Data Processing, Training
    """
    back_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("ModernTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setElideMode(QtCore.Qt.ElideNone)
        self.tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # tabs
        self.tab_annotation = AnnotationTab()
        self.tab_processing = DataProcessingTab()
        self.tab_train = TrainTab()

        self.tabs.addTab(self.tab_annotation, "   Annotation    ")
        self.tabs.addTab(self.tab_processing, "    Data Processing    ")
        self.tabs.addTab(self.tab_train, "    Training    ")

        # Back button row
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_back = QtWidgets.QPushButton("Back to Tasks")
        btn_row.addWidget(self.btn_back)

        lay.addWidget(self.tabs, 1)  # give vertical stretch to consume space
        lay.addLayout(btn_row)

        self.btn_back.clicked.connect(self.back_requested.emit)


class Lab(QWidget):
    """
    Main container:
      - Stage 0: ProjectSetupTab (task selection + dataset paths)
      - Stage 1: ModelBuilderWidget (Annotation, Data Processing, Training tabs)
      - Stage 2: Inference page (as a task, not a tab)
    """
    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self.setObjectName("Root")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(STYLE_SHEET)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(0, 0)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Stacked stages live inside a wrapper widget so expansion persists after reopen
        self.stage = QStackedLayout()

        self.project_page = ProjectSetupTab()
        self.builder_page = ModelBuilderWidget()
        self.infer_page = InferenceTab()

        wrapper = QtWidgets.QWidget()
        wrapper.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        wrapper.setLayout(self.stage)
        root.addWidget(wrapper, 1)

        self.stage.addWidget(self.project_page)
        self.stage.addWidget(self.builder_page)
        self.stage.addWidget(self.infer_page)

        # width guard (~50% of viewer)
        self.setMinimumWidth(400)
        #self.setMaximumWidth(560)

        # Wiring
        self.project_page.continued.connect(self._on_project_continue)
        self.builder_page.back_requested.connect(self._go_back_to_project)
        self.infer_page.back_requested.connect(self._go_back_to_project)

    # ---------- Navigation ---------- #
    def _on_project_continue(self, payload: dict):
        task = payload.get("task", "").strip()
        if task in MODEL_BUILDING_TASKS:
            self.stage.setCurrentWidget(self.builder_page)
        elif task == INFERENCE_TASK:
            self.stage.setCurrentWidget(self.infer_page)
        else:
            QtWidgets.QMessageBox.warning(self, "Choose a task", "Please select a valid task.")

    def _go_back_to_project(self):
        self.stage.setCurrentWidget(self.project_page)

    def sizeHint(self):
        return QtCore.QSize(520, 720)