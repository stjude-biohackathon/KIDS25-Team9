from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QHBoxLayout
from ui.common import Card, DropLineEdit, labeled_row
from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING


class InferenceTab(QWidget):
    run_inference = QtCore.Signal(dict)  # {input_path, is_folder, save_path, overlay}
    back_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.setSizePolicy(self.sizePolicy().Expanding, self.sizePolicy().Expanding)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(scroll.sizePolicy().Expanding, scroll.sizePolicy().Expanding)

        body = QWidget()
        body.setSizePolicy(body.sizePolicy().Expanding, body.sizePolicy().Expanding)
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2,2,2,2)
        body_lay.setSpacing(DEFAULT_SPACING)

        # Input card
        input_card = Card(); il = input_card.layout()
        lbl = QLabel("Inference"); lbl.setObjectName("H2"); il.addWidget(lbl)

        self.input_path = DropLineEdit("Drop a folder or single image…")
        btn_browse_in = QPushButton("Browse"); btn_browse_in.setObjectName("PrimaryBtn")
        il.addLayout(labeled_row("Input", self.input_path, btn_browse_in))

        # Output save
        out_card = Card(); ol = out_card.layout()
        lbl2 = QLabel("Output Settings"); lbl2.setObjectName("H2"); ol.addWidget(lbl2)

        self.save_path = DropLineEdit("Drop a folder to save results…")
        btn_browse_out = QPushButton("Browse"); btn_browse_out.setObjectName("SecondaryBtn")
        ol.addLayout(labeled_row("Save to", self.save_path, btn_browse_out))

        self.cb_overlay = QtWidgets.QCheckBox("Add layer overlay to current viewer after inference")
        ol.addWidget(self.cb_overlay)

        # Actions
        actions = QHBoxLayout(); actions.addStretch(1)
        self.btn_back = QPushButton("Back to Tasks")
        self.btn_run = QPushButton("Run Inference"); self.btn_run.setObjectName("CTA")
        actions.addWidget(self.btn_back); actions.addWidget(self.btn_run)

        body_lay.addWidget(input_card)
        body_lay.addWidget(out_card)
        body_lay.addLayout(actions)

        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        self.setMinimumWidth(380)
        self.setMaximumWidth(560)

        # Events
        btn_browse_in.clicked.connect(lambda: self._choose_path(self.input_path, file_ok=True))
        btn_browse_out.clicked.connect(lambda: self._choose_path(self.save_path, dir_only=True))
        self.btn_run.clicked.connect(self._emit_run)
        self.btn_back.clicked.connect(self.back_requested.emit)

    def _choose_path(self, dest: DropLineEdit, dir_only=False, file_ok=False):
        if dir_only:
            dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", "")
            if dirname:
                dest.setText(dirname)
        elif file_ok:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Image", "",
                "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All (*.*)"
            )
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
