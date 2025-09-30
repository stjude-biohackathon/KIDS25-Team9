from qtpy import QtCore
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QHBoxLayout, QCheckBox, QSpinBox, QComboBox
)

from ui.common import Card
from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING


class DataProcessingTab(QWidget):
    requested_run = QtCore.Signal(dict)  # emits {augmentations:{}, normalization:{}, annotate:bool}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        # building the scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(scroll.sizePolicy().Expanding, scroll.sizePolicy().Expanding)
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2, 2, 2, 2)
        body_lay.setSpacing(DEFAULT_SPACING)

        # Augmentations Card
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

        # Actions
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_run = QPushButton("Apply")
        self.btn_run.setObjectName("CTA")
        actions.addWidget(self.btn_run)

        body_lay.addWidget(aug_card)
        body_lay.addWidget(norm_card)
        body_lay.addLayout(actions)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        self.setMinimumWidth(380)
        self.setMaximumWidth(560)

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
        }
        self.requested_run.emit(payload)
