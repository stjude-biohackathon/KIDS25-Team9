from qtpy import QtCore
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QHBoxLayout, QCheckBox, QSpinBox, QComboBox
)
from ui.common import Card
from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING

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