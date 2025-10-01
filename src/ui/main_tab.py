# FILE: ui/main_tab.py
import os
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QButtonGroup,
    QScrollArea
)

from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ui.common import Card, DropLineEdit, SelectableCard, labeled_row
from ui.state import state


class ProjectSetupTab(QWidget):
    """
    Main/project page (no tabs here):
      - Dataset pickers
      - Task cards in a vertically scrollable area
      - Emits `continued` with {images, labels, task}
    """
    continued = QtCore.Signal(dict)  # emits {images, labels, task}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        # Hero
        hero = QFrame(); hero.setObjectName("Hero")
        hero_lay = QVBoxLayout(hero)
        hero_lay.setContentsMargins(16, 16, 16, 16)
        hero_lay.setSpacing(4)

        title = QLabel("AI-Image-Lab"); title.setObjectName("H1")
        subtitle = QLabel("Select data and choose a task to continue.")
        subtitle.setObjectName("SubH1"); subtitle.setWordWrap(True)
        hero_lay.addWidget(title); hero_lay.addWidget(subtitle)

        # Data sources
        data_card = Card()
        data_lay = data_card.layout()
        data_lay.setContentsMargins(14, 14, 14, 14)

        data_title = QLabel("Datasets"); data_title.setObjectName("H2")
        data_lay.addWidget(data_title)

        self.image_path = DropLineEdit("Drop your *image* folder or click Browse…")
        btn_img = QPushButton("Browse"); btn_img.setObjectName("PrimaryBtn")
        data_lay.addLayout(labeled_row("Image folder", self.image_path, btn_img))

        self.label_path = DropLineEdit("Drop your *label* folder or click Browse…")
        btn_lab = QPushButton("Browse"); btn_lab.setObjectName("PrimaryBtn")
        data_lay.addLayout(labeled_row("Label folder", self.label_path, btn_lab))

        # Task selection (VERTICAL SCROLL AREA)
        task_card = Card()
        task_lay = task_card.layout()
        task_lay.setContentsMargins(14, 14, 14, 14)

        task_title = QLabel("Task"); task_title.setObjectName("H2")
        task_lay.addWidget(task_title)

        self.task_group = QButtonGroup(self)
        self.task_group.setExclusive(True)

        self.tasks_scroll = QScrollArea()
        self.tasks_scroll.setWidgetResizable(True)
        self.tasks_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tasks_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tasks_scroll.setFrameShape(QFrame.NoFrame)
        self.tasks_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        tasks_container = QWidget()
        tasks_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        tasks_col = QVBoxLayout(tasks_container)
        tasks_col.setContentsMargins(0, 0, 0, 0)
        tasks_col.setSpacing(10)

        # Define tasks
        self.card_sem2d = SelectableCard("Semantic Segmentation (2D)", "Pixel-wise classification on 2D images.")
        self.card_sem3d = SelectableCard("Semantic Segmentation (3D)", "Voxel-wise classification on volumes.")
        self.card_inst  = SelectableCard("Instance Segmentation", "Detect and separate individual objects.")
        self.card_fine  = SelectableCard("Fine-tune", "Start from a pre-trained model and adapt to your data.")
        self.card_samft = SelectableCard("Fine Tune SAM", "Fine-tune Segment Anything on your dataset.")
        self.card_infer = SelectableCard("Inference", "Run a trained model on images or a folder.")

        self._task_cards = [
            (self.card_sem2d, "semantic-2d"),
            (self.card_sem3d, "semantic-3d"),
            (self.card_inst,  "instance"),
            (self.card_fine,  "fine-tune"),
            (self.card_samft, "fine-tune-sam"),
            (self.card_infer, "inference"),
        ]

        for card, _key in self._task_cards:
            card.setMinimumHeight(92)
            card.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            tasks_col.addWidget(card)

        tasks_col.addStretch(1)
        self.tasks_scroll.setWidget(tasks_container)
        task_lay.addWidget(self.tasks_scroll)

        # Actions
        actions = QHBoxLayout(); actions.addStretch(1)
        self.btn_cancel = QPushButton("Clear"); self.btn_cancel.setObjectName("GhostBtn")
        self.btn_next = QPushButton("Continue"); self.btn_next.setObjectName("CTA"); self.btn_next.setEnabled(False)
        actions.addWidget(self.btn_cancel); actions.addWidget(self.btn_next)

        # Assemble; give the task card vertical stretch so it “eats” space
        root.addWidget(hero)
        root.addWidget(data_card)
        task_card.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        root.addWidget(task_card, 1)
        root.addLayout(actions)

        # Keep references
        self._btn_img = btn_img
        self._btn_lab = btn_lab

        # width constraints
        self.setMinimumWidth(400)

        # Events
        self._btn_img.clicked.connect(lambda: self._choose_dir(self.image_path))
        self._btn_lab.clicked.connect(lambda: self._choose_dir(self.label_path))
        self.image_path.textChanged.connect(self._validate_ready)
        self.label_path.textChanged.connect(self._validate_ready)

        for card, key in self._task_cards:
            self.task_group.addButton(card.radio())
            card.clicked.connect(lambda c=card: self._on_card_clicked(c))

        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_next.clicked.connect(self._on_continue)

    # ----------- Behavior ----------- #
    def _choose_dir(self, line_edit: DropLineEdit):
        start = line_edit.text().strip() or os.path.expanduser("~")
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if dirname:
            line_edit.setText(dirname)

    def _on_card_clicked(self, card: 'SelectableCard'):
        for c, _k in self._task_cards:
            c.setChecked(c is card)
        self._validate_ready()

    def _selected_task(self) -> str:
        mapping = {self.card_sem2d: "semantic-2d",
                   self.card_sem3d: "semantic-3d",
                   self.card_inst:  "instance",
                   self.card_fine:  "fine-tune",
                   self.card_samft: "fine-tune-sam",
                   self.card_infer: "inference"}
        for card, key in mapping.items():
            if card.isChecked():
                return key
        return ""

    def _validate_ready(self):
        task = self._selected_task()
        has_images = bool(self.image_path.text().strip())
        has_labels = bool(self.label_path.text().strip())

        # Inference: allow continue with NO datasets selected
        if task == "inference":
            ok = True
        else:
            # All other tasks require both images & labels
            ok = bool(task) and has_images and has_labels

        self.btn_next.setEnabled(ok)

    def _on_cancel(self):
        self.image_path.clear()
        self.label_path.clear()
        for c, _k in self._task_cards:
            c.setChecked(False)
        self._validate_ready()

    def _on_continue(self):
        state.task = self._selected_task()
        state.input_img_dir = self.image_path.text().strip()
        state.input_lbl_dir = self.label_path.text().strip()

        payload = {
            "images": state.input_img_dir,
            "labels": state.input_lbl_dir,
            "task": state.task,
        }
        self.continued.emit(payload)
