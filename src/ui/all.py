# ===========================
# FILE: ai_image_lab/__init__.py
# ===========================
from .lab import Lab

try:
    from napari_plugin_engine import napari_hook_implementation

    @napari_hook_implementation
    def napari_experimental_provide_dock_widget():
        return Lab
except Exception:  # pragma: no cover
    # Allow importing this package outside napari without the plugin engine
    pass


# ===========================
# FILE: ai_image_lab/styles.py
# ===========================
from qtpy import QtCore

STYLE_SHEET = """
#Root { background: #0e1116; }

/* Hero */
#Hero { 
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
        stop:0 #1b2330, stop:1 #151a22);
    border: 1px solid #1f2835;
    border-radius: 16px;
}

QLabel#H1 {
    color: #e6edf3;
    font-size: 22px;
    font-weight: 700;
}
QLabel#SubH1 {
    color: #9fb0c1;
    font-size: 13px;
}

QLabel#H2 {
    color: #cbd5e1;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.2px;
}

QLabel#FieldLabel {
    color: #9fb0c1;
    font-size: 12px;
}

/* General text fields */
QLineEdit {
    background: #0b0f14;
    color: #d6e2f0;
    border: 1px solid #243042;
    border-radius: 12px;
    padding: 10px 12px;
    selection-background-color: #385375;
}
QLineEdit:focus {
    border: 1px solid #3a84ff;
}
QLineEdit[placeholderText] { color: #6b7d91; }

/* Buttons */
QPushButton { 
    border-radius: 12px; 
    padding: 10px 14px; 
    font-weight: 600; 
}
QPushButton#PrimaryBtn { 
    background: #1f7aec; color: white; border: 1px solid #1b65c2; 
}
QPushButton#PrimaryBtn:hover { background: #2a86ff; }

QPushButton#SecondaryBtn {
    background: #141a23; color: #cbd5e1; border: 1px solid #2b3a4d;
}
QPushButton#SecondaryBtn:hover { border-color: #3a84ff; color: #e6edf3; }

QPushButton#GhostBtn { 
    background: transparent; color: #93a6bb; border: 1px solid #2b3a4d; 
}
QPushButton#GhostBtn:hover { color: #cbd5e1; border-color: #3a84ff; }

QPushButton#CTA { 
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
        stop:0 #2b9bff, stop:1 #6a5cff);
    color: white; border: none; 
    padding: 12px 18px; font-size: 14px; 
}
QPushButton#CTA:disabled { background: #1a2230; color: #5f728a; }
QPushButton#CTA:hover:!disabled { filter: brightness(1.05); }

/* Cards */
#Card, #SelectableCard {
    background: #0b0f14;
    border: 1px solid #1c2634;
    border-radius: 16px;
}
#Card { box-shadow: 0 8px 24px rgba(0,0,0,0.28); }

/* SelectableCard states */
#SelectableCard[active="true"] {
    border: 1px solid #3a84ff;
    box-shadow: 0 0 0 3px rgba(58,132,255,0.18);
}

QLabel#CardTitle { color: #e6edf3; font-weight: 600; }
QLabel#CardSubtitle { color: #9fb0c1; font-size: 12px; }
"""

# Optional small helper for spacing policies across the app
DEFAULT_CONTENT_MARGINS = (12, 12, 12, 12)
DEFAULT_SPACING = 12


# ===========================
# FILE: ai_image_lab/widgets/common.py
# ===========================
import os
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFrame, QRadioButton, QSizePolicy
)

class DropLineEdit(QLineEdit):
    """A QLineEdit that accepts folder drops and shows a placeholder like a web dropzone."""

    pathChanged = QtCore.Signal(str)

    def __init__(self, placeholder="Drop a folder here or click Browse…", parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText(placeholder)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                local_path = urls[0].toLocalFile()
                if os.path.isdir(local_path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            local_path = urls[0].toLocalFile()
            if os.path.isdir(local_path):
                self.setText(local_path)
                self.pathChanged.emit(local_path)
                event.acceptProposedAction()
                return
        event.ignore()


class Card(QFrame):
    """A rounded, shadowed card container for a modern web-like feel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)


class SelectableCard(QFrame):
    """Clickable card tied to a hidden QRadioButton for model selection."""

    clicked = QtCore.Signal()

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("SelectableCard")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._radio = QRadioButton(self)
        self._radio.setVisible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(6)

        self.title = QLabel(title)
        self.title.setObjectName("CardTitle")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("CardSubtitle")
        self.subtitle.setWordWrap(True)

        outer.addWidget(self.title)
        outer.addWidget(self.subtitle)
        outer.addStretch(1)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.clicked.emit()
        super().mousePressEvent(e)

    def setChecked(self, value: bool):
        self._radio.setChecked(value)
        self.setProperty("active", value)
        self.style().unpolish(self)
        self.style().polish(self)

    def isChecked(self) -> bool:
        return self._radio.isChecked()

    def radio(self) -> QRadioButton:
        return self._radio


def labeled_row(label_text: str, field: QLineEdit, button: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(8)
    label = QLabel(label_text)
    label.setObjectName("FieldLabel")
    label.setMinimumWidth(110)
    label.setWordWrap(True)
    row.addWidget(label)
    row.addWidget(field, 1)
    row.addWidget(button)
    return row


# ===========================
# FILE: ai_image_lab/tabs/main_tab.py
# ===========================
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QButtonGroup

from ..styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ..widgets.common import Card, DropLineEdit, SelectableCard, labeled_row

class ProjectSetupTab(QWidget):
    continued = QtCore.Signal(dict)  # emits {images, labels, task}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        # Header / Hero
        hero = QFrame()
        hero.setObjectName("Hero")
        hero_lay = QVBoxLayout(hero)
        hero_lay.setContentsMargins(18, 20, 18, 20)
        hero_lay.setSpacing(6)

        title = QLabel("Model Builder")
        title.setObjectName("H1")
        subtitle = QLabel("Create training projects from your images & labels. Choose your task to get tailored defaults.")
        subtitle.setObjectName("SubH1")
        subtitle.setWordWrap(True)
        hero_lay.addWidget(title)
        hero_lay.addWidget(subtitle)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2, 2, 2, 2)
        body_lay.setSpacing(DEFAULT_SPACING)

        # Card: Data sources
        data_card = Card()
        data_lay = data_card.layout()
        data_title = QLabel("Datasets")
        data_title.setObjectName("H2")
        data_lay.addWidget(data_title)

        # Image folder
        self.image_path = DropLineEdit("Drop your *image* folder or click Browse…")
        btn_img = QPushButton("Browse")
        btn_img.setObjectName("PrimaryBtn")
        data_lay.addLayout(labeled_row("Image folder", self.image_path, btn_img))

        # Label folder
        self.label_path = DropLineEdit("Drop your *label* folder or click Browse…")
        btn_lab = QPushButton("Browse")
        btn_lab.setObjectName("SecondaryBtn")
        data_lay.addLayout(labeled_row("Label folder", self.label_path, btn_lab))

        body_lay.addWidget(data_card)

        # Card: Task selection
        task_card = Card()
        task_lay = task_card.layout()
        task_title = QLabel("Task")
        task_title.setObjectName("H2")
        task_lay.addWidget(task_title)

        grid = QHBoxLayout()
        grid.setSpacing(12)

        self.task_group = QButtonGroup(self)
        self.task_group.setExclusive(True)

        self.card_sem2d = SelectableCard(
            "Semantic Segmentation (2D)",
            "Pixel-wise classification on 2D images. Great for masks from histology/light microscopy.",
        )
        self.card_sem3d = SelectableCard(
            "Semantic Segmentation (3D)",
            "Voxel-wise classification on volumes. Works with isotropic or anisotropic stacks.",
        )
        self.card_inst = SelectableCard(
            "Instance Segmentation",
            "Detect and separate individual objects (nuclei, cells, synapses, etc.).",
        )
        self.card_sam = SelectableCard(
            "Fine-tune SAM",
            "Adapt Segment Anything to your domain with your own annotations.",
        )

        for i, card in enumerate([self.card_sem2d, self.card_sem3d, self.card_inst, self.card_sam]):
            self.task_group.addButton(card.radio(), i)
            grid.addWidget(card)

        task_lay.addLayout(grid)
        body_lay.addWidget(task_card)

        # Actions
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_cancel = QPushButton("Clear")
        self.btn_cancel.setObjectName("GhostBtn")
        self.btn_next = QPushButton("Continue")
        self.btn_next.setObjectName("CTA")
        self.btn_next.setEnabled(False)
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_next)

        body_lay.addLayout(actions)
        scroll.setWidget(body)

        root.addWidget(hero)
        root.addWidget(scroll)

        # Keep references
        self._btn_img = btn_img
        self._btn_lab = btn_lab

        # Size hinting: keep it under ~50% width of typical viewer
        self.setMinimumWidth(380)
        self.setMaximumWidth(560)

        # Wire events
        self._btn_img.clicked.connect(lambda: self._choose_dir(self.image_path))
        self._btn_lab.clicked.connect(lambda: self._choose_dir(self.label_path))
        self.image_path.textChanged.connect(self._validate_ready)
        self.label_path.textChanged.connect(self._validate_ready)
        for card in [self.card_sem2d, self.card_sem3d, self.card_inst, self.card_sam]:
            card.clicked.connect(lambda c=card: self._on_card_clicked(c))
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_next.clicked.connect(self._on_continue)

    # --------------- Behavior --------------- #
    def _choose_dir(self, line_edit: DropLineEdit):
        start = line_edit.text().strip() or os.path.expanduser("~")
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if dirname:
            line_edit.setText(dirname)

    def _on_card_clicked(self, card: SelectableCard):
        for c in [self.card_sem2d, self.card_sem3d, self.card_inst, self.card_sam]:
            c.setChecked(c is card)
        self._validate_ready()

    def _selected_task(self) -> str:
        mapping = {
            self.card_sem2d: "semantic-2d",
            self.card_sem3d: "semantic-3d",
            self.card_inst: "instance",
            self.card_sam: "finetune-sam",
        }
        for card, key in mapping.items():
            if card.isChecked():
                return key
        return ""

    def _validate_ready(self):
        ok = bool(self.image_path.text().strip()) and bool(self.label_path.text().strip()) and bool(self._selected_task())
        self.btn_next.setEnabled(ok)

    def _on_cancel(self):
        self.image_path.clear()
        self.label_path.clear()
        for c in [self.card_sem2d, self.card_sem3d, self.card_inst, self.card_sam]:
            c.setChecked(False)
        self._validate_ready()

    def _on_continue(self):
        payload = {
            "images": self.image_path.text().strip(),
            "labels": self.label_path.text().strip(),
            "task": self._selected_task(),
        }
        self.continued.emit(payload)


# ===========================
# FILE: ai_image_lab/tabs/data_processing_tab.py
# ===========================
import os
from qtpy import QtCore
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QHBoxLayout, QCheckBox, QSpinBox, QComboBox
)
from ..widgets.common import Card
from ..styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING

class DataProcessingTab(QWidget):
    requested_run = QtCore.Signal(dict)  # emits {augmentations: {...}, normalization: {...}, annotate: bool}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2, 2, 2, 2)
        body_lay.setSpacing(DEFAULT_SPACING)

        # Augmentations Card (placeholder controls; you will flesh out later)
        aug_card = Card()
        aug_lay = aug_card.layout()
        lbl = QLabel("Data Augmentation")
        lbl.setObjectName("H2")
        aug_lay.addWidget(lbl)

        self.cb_flip = QCheckBox("Random flips")
        self.cb_rot = QCheckBox("Random rotations")
        self.cb_noise = QCheckBox("Gaussian noise")
        aug_lay.addWidget(self.cb_flip)
        aug_lay.addWidget(self.cb_rot)
        aug_lay.addWidget(self.cb_noise)

        # Normalization Card
        norm_card = Card()
        norm_lay = norm_card.layout()
        lbl2 = QLabel("Preprocessing / Normalization")
        lbl2.setObjectName("H2")
        norm_lay.addWidget(lbl2)

        row_norm = QHBoxLayout()
        self.norm_method = QComboBox()
        self.norm_method.addItems(["none", "z-score", "min-max", "percentile-clip+z"])
        self.norm_clip = QSpinBox(); self.norm_clip.setRange(0, 20); self.norm_clip.setValue(2)
        row_norm.addWidget(QLabel("Method"))
        row_norm.addWidget(self.norm_method)
        row_norm.addWidget(QLabel("Clip %"))
        row_norm.addWidget(self.norm_clip)
        norm_lay.addLayout(row_norm)

        # Annotation Card
        ann_card = Card()
        ann_lay = ann_card.layout()
        lbl3 = QLabel("Additional Annotation")
        lbl3.setObjectName("H2")
        ann_lay.addWidget(lbl3)
        self.cb_annotate = QCheckBox("Open annotation tools to refine labels")
        ann_lay.addWidget(self.cb_annotate)

        # Actions
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_back = QPushButton("Back")
        self.btn_run = QPushButton("Apply & Continue")
        self.btn_run.setObjectName("CTA")
        actions.addWidget(self.btn_back)
        actions.addWidget(self.btn_run)

        body_lay.addWidget(aug_card)
        body_lay.addWidget(norm_card)
        body_lay.addWidget(ann_card)
        body_lay.addLayout(actions)
        scroll.setWidget(body)
        root.addWidget(scroll)

        self.setMinimumWidth(380)
        self.setMaximumWidth(560)

        # Events
        self.btn_run.clicked.connect(self._emit_request)

    def _emit_request(self):
        payload = {
            "augmentations": {
                "flip": self.cb_flip.isChecked(),
                "rotation": self.cb_rot.isChecked(),
                "noise": self.cb_noise.isChecked(),
            },
            "normalization": {
                "method": self.norm_method.currentText(),
                "clip": self.norm_clip.value(),
            },
            "annotate": self.cb_annotate.isChecked(),
        }
        self.requested_run.emit(payload)


# ===========================
# FILE: ai_image_lab/tabs/train_tab.py
# ===========================
from qtpy import QtCore
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QHBoxLayout, QCheckBox, QSpinBox, QComboBox
)
from ..widgets.common import Card
from ..styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING

class TrainTab(QWidget):
    start_training = QtCore.Signal(dict)  # emits {mode: "train"|"finetune", epochs:int, batch:int}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget(); body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2,2,2,2); body_lay.setSpacing(DEFAULT_SPACING)

        card = Card(); lay = card.layout()
        title = QLabel("Train or Fine-tune")
        title.setObjectName("H2"); lay.addWidget(title)

        self.cb_train = QCheckBox("Train from scratch")
        self.cb_finetune = QCheckBox("Fine-tune existing model")
        self.cb_train.stateChanged.connect(lambda _: self._sync_mode(self.cb_train))
        self.cb_finetune.stateChanged.connect(lambda _: self._sync_mode(self.cb_finetune))
        lay.addWidget(self.cb_train)
        lay.addWidget(self.cb_finetune)

        # Simple hyperparams
        hp_row = QHBoxLayout()
        self.epochs = QSpinBox(); self.epochs.setRange(1, 1000); self.epochs.setValue(50)
        self.batch = QSpinBox(); self.batch.setRange(1, 64); self.batch.setValue(4)
        hp_row.addWidget(QLabel("Epochs")); hp_row.addWidget(self.epochs)
        hp_row.addWidget(QLabel("Batch")); hp_row.addWidget(self.batch)
        lay.addLayout(hp_row)

        actions = QHBoxLayout(); actions.addStretch(1)
        self.btn_back = QPushButton("Back")
        self.btn_start = QPushButton("Start Training")
        self.btn_start.setObjectName("CTA")
        actions.addWidget(self.btn_back)
        actions.addWidget(self.btn_start)

        body_lay.addWidget(card)
        body_lay.addLayout(actions)
        scroll.setWidget(body)
        root.addWidget(scroll)

        self.setMinimumWidth(380)
        self.setMaximumWidth(560)

        self.btn_start.clicked.connect(self._emit_start)

    def _sync_mode(self, source):
        # make checkboxes mutually exclusive
        if source is self.cb_train and self.cb_train.isChecked():
            self.cb_finetune.setChecked(False)
        elif source is self.cb_finetune and self.cb_finetune.isChecked():
            self.cb_train.setChecked(False)

    def _emit_start(self):
        mode = "train" if self.cb_train.isChecked() else ("finetune" if self.cb_finetune.isChecked() else "")
        payload = {"mode": mode, "epochs": self.epochs.value(), "batch": self.batch.value()}
        self.start_training.emit(payload)


# ===========================
# FILE: ai_image_lab/tabs/inference_tab.py
# ===========================
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QHBoxLayout, QLineEdit, QFileDialog, QCheckBox
)
from ..widgets.common import Card, DropLineEdit, labeled_row
from ..styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING

class InferenceTab(QWidget):
    run_inference = QtCore.Signal(dict)  # emits {input_path: str, is_folder: bool, save_path: str, overlay: bool}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget(); body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2,2,2,2); body_lay.setSpacing(DEFAULT_SPACING)

        # Input card
        input_card = Card(); il = input_card.layout()
        lbl = QLabel("Inference Input")
        lbl.setObjectName("H2"); il.addWidget(lbl)

        self.input_path = DropLineEdit("Drop a folder or single image…")
        btn_browse_in = QPushButton("Browse"); btn_browse_in.setObjectName("PrimaryBtn")
        il.addLayout(labeled_row("Input", self.input_path, btn_browse_in))

        # Output save
        out_card = Card(); ol = out_card.layout()
        lbl2 = QLabel("Output Settings")
        lbl2.setObjectName("H2"); ol.addWidget(lbl2)

        self.save_path = DropLineEdit("Drop a folder to save results…")
        btn_browse_out = QPushButton("Browse"); btn_browse_out.setObjectName("SecondaryBtn")
        ol.addLayout(labeled_row("Save to", self.save_path, btn_browse_out))

        self.cb_overlay = QCheckBox("Add layer overlay to current viewer after inference")
        ol.addWidget(self.cb_overlay)

        # Actions
        actions = QHBoxLayout(); actions.addStretch(1)
        self.btn_back = QPushButton("Back")
        self.btn_run = QPushButton("Run Inference"); self.btn_run.setObjectName("CTA")
        actions.addWidget(self.btn_back); actions.addWidget(self.btn_run)

        body_lay.addWidget(input_card)
        body_lay.addWidget(out_card)
        body_lay.addLayout(actions)
        scroll.setWidget(body)
        root.addWidget(scroll)

        self.setMinimumWidth(380)
        self.setMaximumWidth(560)

        # Events
        btn_browse_in.clicked.connect(lambda: self._choose_path(self.input_path, file_ok=True))
        btn_browse_out.clicked.connect(lambda: self._choose_path(self.save_path, dir_only=True))
        self.btn_run.clicked.connect(self._emit_run)

    def _choose_path(self, dest: DropLineEdit, dir_only=False, file_ok=False):
        if dir_only:
            dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", "")
            if dirname:
                dest.setText(dirname)
        elif file_ok:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image or Folder", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All (*.*)")
            if path:
                dest.setText(path)

    def _emit_run(self):
        path = self.input_path.text().strip()
        is_folder = QtCore.QFileInfo(path).isDir()
        payload = {
            "input_path": path,
            "is_folder": is_folder,
            "save_path": self.save_path.text().strip(),
            "overlay": self.cb_overlay.isChecked(),
        }
        self.run_inference.emit(payload)


# ===========================
# FILE: ai_image_lab/lab.py
# ===========================

from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ui.common import Card, DropLineEdit, SelectableCard, labeled_row

import os
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from .styles import STYLE_SHEET
from .tabs.main_tab import ProjectSetupTab
from .tabs.data_processing_tab import DataProcessingTab
from .tabs.train_tab import TrainTab
from .tabs.inference_tab import InferenceTab

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
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.North)

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
