from qtpy import QtCore
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ui.common import Card


class AnnotationTab(QWidget):
    """
    Placeholder annotation tab.
    You can later add your Napari layer tools, class pickers, ROI helpers, etc.
    """
    proceed = QtCore.Signal(dict)  # if you need to signal outward

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        card = Card()
        lay = card.layout()
        title = QLabel("Annotation")
        title.setObjectName("H2")
        lay.addWidget(title)

        desc = QLabel("Use Napari tools to refine labels. (Hook your annotation workflow here.)")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        # optional actions
        actions = QHBoxLayout()
        actions.addStretch(1)
        btn = QPushButton("Save Annotations")
        actions.addWidget(btn)

        card.setSizePolicy(card.sizePolicy().Expanding, card.sizePolicy().Expanding)
        root.addWidget(card, 1)
        root.addLayout(actions)
