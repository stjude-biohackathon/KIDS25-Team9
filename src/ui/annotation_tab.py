from qtpy import QtCore
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ui.common import Card


class AnnotationTab(QWidget):
    proceed = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(self.sizePolicy().Expanding, self.sizePolicy().Expanding)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        card = Card()
        lay = card.layout()
        title = QLabel("Annotation"); title.setObjectName("H2")
        lay.addWidget(title)

        desc = QLabel("Use Napari tools to refine labels. (Hook your annotation workflow here.)")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        # actions at bottom
        actions = QHBoxLayout()
        actions.addStretch(1)
        btn = QPushButton("Save Annotations")
        actions.addWidget(btn)

        card.setSizePolicy(card.sizePolicy().Expanding, card.sizePolicy().Expanding)
        root.addWidget(card, 1)   # stretch to fill vertical space
        root.addLayout(actions)
