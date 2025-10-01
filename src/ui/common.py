import os
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFrame, QRadioButton, QSizePolicy
)


class DropLineEdit(QLineEdit):
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
    clicked = QtCore.Signal()  # emits no args

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
