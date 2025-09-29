# FILE: ai_image_lab/tabs/main_tab.py

import os
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QButtonGroup
)

from ui.common import Card, DropLineEdit, SelectableCard, labeled_row


class ProjectSetupTab(QWidget):
    """
    Main/project tab:
    - No scroll area; everything visible in one view.
    - Task cards are horizontally stacked in a single row.
    - Emits `continued` with {images, labels, task}.
    """
    continued = QtCore.Signal(dict)  # emits {images, labels, task}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Hero
        hero = QFrame()
        hero.setObjectName("Hero")
        hero_lay = QVBoxLayout(hero)
        hero_lay.setContentsMargins(16, 16, 16, 16)
        hero_lay.setSpacing(4)

        title = QLabel("Model Builder")
        title.setObjectName("H1")
        subtitle = QLabel(
            "Create training projects from your images & labels. Choose your task to get tailored defaults."
        )
        subtitle.setObjectName("SubH1")
        subtitle.setWordWrap(True)
        hero_lay.addWidget(title)
        hero_lay.addWidget(subtitle)

        # Card: Data sources
        data_card = Card()
        data_lay = data_card.layout()
        data_lay.setContentsMargins(14, 14, 14, 14)

        data_title = QLabel("Datasets")
        data_title.setObjectName("H2")
        data_lay.addWidget(data_title)

        self.image_path = DropLineEdit("Drop your *image* folder or click Browse…")
        btn_img = QPushButton("Browse")
        btn_img.setObjectName("PrimaryBtn")
        data_lay.addLayout(labeled_row("Image folder", self.image_path, btn_img))

        self.label_path = DropLineEdit("Drop your *label* folder or click Browse…")
        btn_lab = QPushButton("Browse")
        btn_lab.setObjectName("SecondaryBtn")
        data_lay.addLayout(labeled_row("Label folder", self.label_path, btn_lab))

        # Card: Task selection (HORIZONTAL row)
        task_card = Card()
        task_lay = task_card.layout()
        task_lay.setContentsMargins(14, 14, 14, 14)

        task_title = QLabel("Task")
        task_title.setObjectName("H2")
        task_lay.addWidget(task_title)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.task_group = QButtonGroup(self)
        self.task_group.setExclusive(True)

        self.card_sem2d = SelectableCard(
            "Semantic Segmentation (2D)",
            "Pixel-wise classification on 2D images."
        )
        self.card_sem3d = SelectableCard(
            "Semantic Segmentation (3D)",
            "Voxel-wise classification on volumes."
        )
        self.card_inst = SelectableCard(
            "Instance Segmentation",
            "Detect and separate individual objects."
        )
        self.card_sam = SelectableCard(
            "Fine-tune SAM",
            "Adapt Segment Anything to your domain."
        )

        # Make them compact so 4 fit side-by-side without scrolling
        for c in [self.card_sem2d, self.card_sem3d, self.card_inst, self.card_sam]:
            c.setMinimumWidth(120)
            c.setMaximumWidth(9999)
            c.setMinimumHeight(88)
            c.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)

        # Add to group and row; give equal stretch so they share width nicely
        for i, card in enumerate([self.card_sem2d, self.card_sem3d, self.card_inst, self.card_sam]):
            self.task_group.addButton(card.radio(), i)
            row.addWidget(card, 1)

        task_lay.addLayout(row)

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

        # Assemble (no scroll area)
        root.addWidget(hero)
        root.addWidget(data_card)
        root.addWidget(task_card)
        root.addLayout(actions)

        # Keep references
        self._btn_img = btn_img
        self._btn_lab = btn_lab

        # Keep under ~50% of viewer width (soft)
        self.setMinimumWidth(420)
        self.setMaximumWidth(560)

        # Events
        self._btn_img.clicked.connect(lambda: self._choose_dir(self.image_path))
        self._btn_lab.clicked.connect(lambda: self._choose_dir(self.label_path))
        self.image_path.textChanged.connect(self._validate_ready)
        self.label_path.textChanged.connect(self._validate_ready)
        for card in [self.card_sem2d, self.card_sem3d, self.card_inst, self.card_sam]:
            card.clicked.connect(lambda c=card: self._on_card_clicked(c))
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_next.clicked.connect(self._on_continue)

    # ----------- Behavior ----------- #
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
