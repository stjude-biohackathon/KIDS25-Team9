# FILE: ui/huggingface_model_dialog.py

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QTextEdit, QSplitter, QMessageBox,
    QProgressBar, QComboBox
)
from qtpy.QtCore import Qt, QThread, Signal

from utilities.huggingface_client import HuggingFaceClient


class FetchModelsThread(QThread):
    """Background thread for fetching models from Hugging Face API."""
    
    finished = Signal(list)  # emits list of models
    error = Signal(str)      # emits error message
    
    def __init__(self, client: HuggingFaceClient, search: str = "", task: str = ""):
        super().__init__()
        self.client = client
        self.search = search
        self.task = task
    
    def run(self):
        try:
            models = self.client.list_models(
                search=self.search if self.search else None,
                task=self.task if self.task else None,
                limit=50
            )
            self.finished.emit(models)
        except Exception as e:
            self.error.emit(str(e))


class HuggingFaceModelDialog(QDialog):
    """
    Dialog for browsing and selecting models from Hugging Face Hub.
    """
    
    # Task filter options relevant for image analysis
    TASK_FILTERS = [
        ("All Tasks", ""),
        ("Image Segmentation", "image-segmentation"),
        ("Object Detection", "object-detection"),
        ("Image Classification", "image-classification"),
        ("Depth Estimation", "depth-estimation"),
        ("Zero-Shot Image Classification", "zero-shot-image-classification"),
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Browse Hugging Face Models")
        self.setModal(True)
        self.resize(800, 600)
        
        self.client = HuggingFaceClient()
        self.selected_model_id = None
        self.fetch_thread = None
        
        self._build_ui()
        self._load_models()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # Search and filter row
        search_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search models...")
        self.search_input.returnPressed.connect(self._on_search)
        
        self.task_filter = QComboBox()
        for label, _ in self.TASK_FILTERS:
            self.task_filter.addItem(label)
        self.task_filter.currentIndexChanged.connect(self._on_search)
        
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._on_search)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._load_models)
        
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_input, 1)
        search_layout.addWidget(QLabel("Task:"))
        search_layout.addWidget(self.task_filter)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.refresh_button)
        
        layout.addLayout(search_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Splitter with model list and details
        splitter = QSplitter(Qt.Horizontal)
        
        # Model list
        list_widget = QtWidgets.QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        
        list_label = QLabel("Models")
        list_label.setStyleSheet("font-weight: bold;")
        list_layout.addWidget(list_label)
        
        self.model_list = QListWidget()
        self.model_list.itemSelectionChanged.connect(self._on_model_selected)
        self.model_list.itemDoubleClicked.connect(self._on_accept)
        list_layout.addWidget(self.model_list)
        
        splitter.addWidget(list_widget)
        
        # Model details
        details_widget = QtWidgets.QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        
        details_label = QLabel("Model Details")
        details_label.setStyleSheet("font-weight: bold;")
        details_layout.addWidget(details_label)
        
        self.model_details = QTextEdit()
        self.model_details.setReadOnly(True)
        details_layout.addWidget(self.model_details)
        
        splitter.addWidget(details_widget)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter, 1)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.select_button = QPushButton("Select")
        self.select_button.setEnabled(False)
        self.select_button.clicked.connect(self._on_accept)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def _load_models(self):
        """Load popular models from Hugging Face."""
        self._fetch_models("", "")
    
    def _on_search(self):
        """Handle search button click or Enter key."""
        search_text = self.search_input.text().strip()
        task_idx = self.task_filter.currentIndex()
        task_value = self.TASK_FILTERS[task_idx][1]
        self._fetch_models(search_text, task_value)
    
    def _fetch_models(self, search: str, task: str):
        """Fetch models in background thread."""
        if self.fetch_thread and self.fetch_thread.isRunning():
            return
        
        self.progress_bar.setVisible(True)
        self.search_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.model_list.clear()
        self.model_details.clear()
        
        self.fetch_thread = FetchModelsThread(self.client, search, task)
        self.fetch_thread.finished.connect(self._on_models_loaded)
        self.fetch_thread.error.connect(self._on_fetch_error)
        self.fetch_thread.start()
    
    def _on_models_loaded(self, models):
        """Handle successful model fetch."""
        self.progress_bar.setVisible(False)
        self.search_button.setEnabled(True)
        self.refresh_button.setEnabled(True)
        
        self.model_list.clear()
        for model in models:
            model_id = model.get("id", "")
            downloads = model.get("downloads", 0)
            likes = model.get("likes", 0)
            
            item = QListWidgetItem(f"{model_id} (↓{downloads:,} ♥{likes})")
            item.setData(Qt.UserRole, model)
            self.model_list.addItem(item)
        
        if not models:
            self.model_details.setPlainText("No models found. Try a different search query.")
    
    def _on_fetch_error(self, error_msg):
        """Handle fetch error."""
        self.progress_bar.setVisible(False)
        self.search_button.setEnabled(True)
        self.refresh_button.setEnabled(True)
        
        QMessageBox.warning(
            self,
            "Error",
            f"Failed to fetch models from Hugging Face:\n{error_msg}"
        )
    
    def _on_model_selected(self):
        """Handle model selection in the list."""
        items = self.model_list.selectedItems()
        if not items:
            self.select_button.setEnabled(False)
            self.model_details.clear()
            return
        
        item = items[0]
        model_data = item.data(Qt.UserRole)
        
        self.select_button.setEnabled(True)
        self.selected_model_id = model_data.get("id", "")
        
        # Display model details
        details_text = self._format_model_details(model_data)
        self.model_details.setPlainText(details_text)
    
    def _format_model_details(self, model: dict) -> str:
        """Format model information for display."""
        lines = []
        lines.append(f"Model ID: {model.get('id', 'N/A')}")
        lines.append(f"Author: {model.get('author', 'N/A')}")
        lines.append(f"Downloads: {model.get('downloads', 0):,}")
        lines.append(f"Likes: {model.get('likes', 0):,}")
        
        if "pipeline_tag" in model:
            lines.append(f"Task: {model['pipeline_tag']}")
        
        if "tags" in model:
            tags = ", ".join(model["tags"][:10])  # show first 10 tags
            lines.append(f"Tags: {tags}")
        
        if "lastModified" in model:
            lines.append(f"Last Modified: {model['lastModified']}")
        
        return "\n".join(lines)
    
    def _on_accept(self):
        """Handle accept (Select button or double-click)."""
        if self.selected_model_id:
            self.accept()
    
    def get_selected_model_id(self) -> str:
        """Get the selected model ID."""
        return self.selected_model_id or ""
