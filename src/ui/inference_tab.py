# FILE: ui/inference_tab.py

import os
import importlib

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QHBoxLayout,
    QComboBox, QLineEdit, QFileDialog, QMenu, QMessageBox, QSizePolicy
)

from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ui.common import Card, DropLineEdit, labeled_row
from ui.huggingface_model_dialog import HuggingFaceModelDialog


class InferenceTab(QWidget):
    """
    Single-card inference UI (no dependency on state.task):

    - Input: file or folder (Browse menu: File / Folder)
    - Model source: Hugging Face | Bioimage IO | CBI Model Zoo | Local
        * Non-Local => URL field
        * Local     => model file picker + "Model type" dropdown (Semantic 2D / Semantic 3D / Instance / General)
          The selected model type is used to choose the inferencer class. (We do not fetch from state.)
    - Save to: folder
    - Run Inference:
        * if input is file   -> run_inference_single(...)
        * if input is folder -> run_inference(...)
      (Inferencer class resolved dynamically from selection)
    - Results viewer:
        * "Load Results", "Previous", "Next" to add label layer(s) on top of the
          corresponding input images in the EXISTING napari viewer.
    """
    run_inference = QtCore.Signal(dict)  # optional: emits payload on run
    back_requested = QtCore.Signal()

    _IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

    # ---- Customize to your project ----
    INFERENCER_CLASSES = {
        "semantic-2d": "inferencers.semantic2d.Semantic2DInferencer",
        "semantic-3d": "inferencers.semantic3d.Semantic3DInferencer",
        "instance":    "inferencers.instance.InstanceInferencer",
        "general":     "inferencers.general.GeneralInferencer",
    }
    # -----------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # navigation state for results
        self._nav_pairs = []   # list of (image_path, label_path)
        self._nav_index = -1

        self._build_ui()

    # ---------------- UI ---------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        card = Card()
        lay = card.layout()
        title = QLabel("Inference"); title.setObjectName("H2")
        lay.addWidget(title)

        # --- Input row (file or folder) ---
        in_row = QHBoxLayout(); in_row.setSpacing(8)
        in_row.addWidget(QLabel("Input"), 0)
        self.input_path_edit = QLineEdit(); self.input_path_edit.setReadOnly(True)
        self.input_path_edit.setPlaceholderText("Choose an image file or a folder…")
        self.btn_browse_in = QPushButton("Browse"); self.btn_browse_in.setObjectName("PrimaryBtn")
        # Browse menu: File / Folder
        menu = QMenu(self.btn_browse_in)
        act_file = menu.addAction("Choose File…")
        act_dir  = menu.addAction("Choose Folder…")
        self.btn_browse_in.setMenu(menu)
        act_file.triggered.connect(self._choose_input_file)
        act_dir.triggered.connect(self._choose_input_folder)
        in_row.addWidget(self.input_path_edit, 1)
        in_row.addWidget(self.btn_browse_in, 0)
        lay.addLayout(in_row)

        # --- Model source drop-down ---
        src_row = QHBoxLayout(); src_row.setSpacing(8)
        src_row.addWidget(QLabel("Model source"), 0)
        self.model_source = QComboBox()
        self.model_source.addItems(["Hugging Face", "Bioimage IO", "CBI Model Zoo", "Local"])
        src_row.addWidget(self.model_source, 1)
        lay.addLayout(src_row)

        # --- Model selector: URL (for web sources) OR local file browse ---
        self.model_stack = QtWidgets.QStackedWidget()
        # URL page (with Browse button for Hugging Face)
        url_page = QWidget(); url_lay = QHBoxLayout(url_page); url_lay.setContentsMargins(0, 0, 0, 0); url_lay.setSpacing(8)
        self.model_url_edit = QLineEdit(); self.model_url_edit.setPlaceholderText("Enter model URL or identifier…")
        self.btn_hf_browse = QPushButton("Browse"); self.btn_hf_browse.setObjectName("SecondaryBtn")
        self.btn_hf_browse.clicked.connect(self._browse_huggingface_models)
        url_lay.addWidget(QLabel("Model")); url_lay.addWidget(self.model_url_edit, 1); url_lay.addWidget(self.btn_hf_browse)
        # Local file page
        file_page = QWidget(); file_lay = QHBoxLayout(file_page); file_lay.setContentsMargins(0, 0, 0, 0); file_lay.setSpacing(8)
        self.model_file_edit = QLineEdit(); self.model_file_edit.setReadOnly(True); self.model_file_edit.setPlaceholderText("Choose a local model file…")
        self.btn_model_browse = QPushButton("Browse"); self.btn_model_browse.setObjectName("SecondaryBtn")
        self.btn_model_browse.clicked.connect(self._choose_model_file)
        file_lay.addWidget(QLabel("Model")); file_lay.addWidget(self.model_file_edit, 1); file_lay.addWidget(self.btn_model_browse)

        self.model_stack.addWidget(url_page)   # index 0
        self.model_stack.addWidget(file_page)  # index 1
        lay.addWidget(self.model_stack)

        # --- Local-only: Model type (class resolution) ---
        self.local_type_row = QHBoxLayout(); self.local_type_row.setSpacing(8)
        self.local_type_row.addWidget(QLabel("Model type"), 0)
        self.local_task_type = QComboBox()
        self.local_task_type.addItems(["Semantic 2D", "Semantic 3D", "Instance", "General"])
        self.local_type_row.addWidget(self.local_task_type, 1)
        lay.addLayout(self.local_type_row)

        self.model_source.currentTextChanged.connect(self._on_model_source_changed)
        self._on_model_source_changed(self.model_source.currentText())

        # --- Save to folder ---
        self.save_path = DropLineEdit("Drop or browse a folder to save results…")
        btn_browse_out = QPushButton("Browse"); btn_browse_out.setObjectName("SecondaryBtn")
        lay.addLayout(labeled_row("Save to", self.save_path, btn_browse_out))
        btn_browse_out.clicked.connect(lambda: self._choose_dir(self.save_path))

        # --- Run + Result navigation ---
        # --- Actions row 1: Back + Run ---
        row1 = QHBoxLayout();
        row1.setSpacing(8)

        self.btn_back = QPushButton("Back to Tasks")
        row1.addWidget(self.btn_back)
        self.btn_back.clicked.connect(self.back_requested.emit)

        self.btn_run = QPushButton("Run Inference");
        self.btn_run.setObjectName("CTA")
        row1.addWidget(self.btn_run)
        row1.addStretch(1)

        lay.addLayout(row1)

        # --- Actions row 2: Results navigation ---
        row2 = QHBoxLayout();
        row2.setSpacing(8)

        self.btn_load = QPushButton("Load Results")
        self.btn_prev = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.btn_prev.setEnabled(False);
        self.btn_next.setEnabled(False)

        row2.addWidget(self.btn_load)
        row2.addWidget(self.btn_prev)
        row2.addWidget(self.btn_next)
        row2.addStretch(1)

        lay.addLayout(row2)

        scroll.setWidget(card)
        root.addWidget(scroll, 1)

        # Events
        self.btn_run.clicked.connect(self._on_run_inference)
        self.btn_load.clicked.connect(self._on_load_results)
        self.btn_prev.clicked.connect(lambda: self._step_nav(-1))
        self.btn_next.clicked.connect(lambda: self._step_nav(+1))

    # ---------------- Events / Browsing ---------------- #
    def _choose_input_file(self):
        start = os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", start,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All (*.*)"
        )
        if path:
            self.input_path_edit.setText(path)

    def _choose_input_folder(self):
        start = os.path.expanduser("~")
        dirname = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if dirname:
            self.input_path_edit.setText(dirname)

    def _choose_dir(self, dest: DropLineEdit):
        start = dest.text().strip() or os.path.expanduser("~")
        dirname = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if dirname:
            dest.setText(dirname)

    def _choose_model_file(self):
        start = self.model_file_edit.text().strip() or os.path.expanduser("~")
        fname, _ = QFileDialog.getOpenFileName(self, "Select Model File", start, "All files (*.*)")
        if fname:
            self.model_file_edit.setText(fname)
    
    def _browse_huggingface_models(self):
        """Open dialog to browse and select models from Hugging Face Hub."""
        try:
            dialog = HuggingFaceModelDialog(self)
            if dialog.exec_():
                model_id = dialog.get_selected_model_id()
                if model_id:
                    self.model_url_edit.setText(model_id)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to open Hugging Face model browser:\n{type(e).__name__}: {e}"
            )

    def _on_model_source_changed(self, text: str):
        """
        Non-Local -> URL entry (no local type row)
        Local     -> file browse + show local task type dropdown
        Hugging Face -> URL entry with Browse button enabled
        """
        is_local = text.lower().strip() == "local"
        is_huggingface = text.lower().strip() == "hugging face"
        
        self.model_stack.setCurrentIndex(1 if is_local else 0)
        
        # show/hide the model type chooser only for Local
        for i in range(self.local_type_row.count()):
            item = self.local_type_row.itemAt(i)
            if item.widget():
                item.widget().setVisible(is_local)
        
        # Show/hide Browse button for Hugging Face
        if hasattr(self, 'btn_hf_browse'):
            self.btn_hf_browse.setVisible(is_huggingface)

    # ---------------- Run Inference ---------------- #
    def _on_run_inference(self):
        input_path = self.input_path_edit.text().strip()
        save_dir = self.save_path.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Missing input", "Please choose an input file or folder.")
            return
        if not save_dir:
            QMessageBox.warning(self, "Missing output folder", "Please choose a folder to save results.")
            return
        os.makedirs(save_dir, exist_ok=True)

        model_id = self._get_model_identifier()  # URL or local file path
        infer_cls = self._resolve_inferencer_class_from_selection()
        if infer_cls is None:
            QMessageBox.critical(self, "Inferencer not found",
                                 "Could not resolve an inferencer class for the current selection.")
            return

        try:
            inferencer = infer_cls()
            if os.path.isdir(input_path):
                self._run_inference_folder(inferencer, input_path, model_id, save_dir)
            else:
                self._run_inference_single_image(inferencer, input_path, model_id, save_dir)

            # optional emit for listeners
            self.run_inference.emit({
                "input_path": input_path,
                "is_folder": os.path.isdir(input_path),
                "model": model_id,
                "save_path": save_dir,
                "source": self.model_source.currentText(),
                "local_model_type": self.local_task_type.currentText() if self.model_source.currentText().lower() == "local" else "",
            })

            # refresh nav list so user can inspect outputs
            self._build_nav_pairs(input_path, save_dir)
            if self._nav_pairs:
                self._nav_index = 0
                self._update_nav_buttons()
                self._show_nav_index()
            else:
                QMessageBox.information(self, "No outputs found",
                                        "Inference completed, but no output images were found in the save folder.")
        except Exception as e:
            QMessageBox.critical(self, "Inference failed", f"{type(e).__name__}: {e}")

    def _get_model_identifier(self) -> str:
        if self.model_source.currentText().lower() == "local":
            return self.model_file_edit.text().strip()
        return self.model_url_edit.text().strip()

    def _resolve_inferencer_class_from_selection(self):
        """
        For Local: map the selected model type to a class via INFERENCER_CLASSES.
        For non-Local: default to 'general' inferencer.
        """
        src = self.model_source.currentText().lower().strip()
        key = "general"
        if src == "local":
            choice = self.local_task_type.currentText().lower().strip()
            if "semantic 2d" in choice:
                key = "semantic-2d"
            elif "semantic 3d" in choice:
                key = "semantic-3d"
            elif "instance" in choice:
                key = "instance"
            else:
                key = "general"
        # import the class
        path = self.INFERENCER_CLASSES.get(key)
        if not path:
            return None
        try:
            mod_name, cls_name = path.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name, None)
        except Exception:
            return None

    # Separate functions (single vs folder), call class methods
    def _run_inference_single_image(self, inferencer, image_path: str, model_id: str, save_dir: str):
        """
        Call the inferencer's single-image API.
        Expected: inferencer.run_inference_single(image_path=..., model=..., save_dir=...)
        """
        if hasattr(inferencer, "run_inference_single"):
            inferencer.run_inference_single(image_path=image_path, model=model_id, save_dir=save_dir)
        else:
            # Fallback naming
            inferencer.run_single(image_path=image_path, model=model_id, save_dir=save_dir)

    def _run_inference_folder(self, inferencer, input_dir: str, model_id: str, save_dir: str):
        """
        Call the inferencer's folder API.
        Expected: inferencer.run_inference(input_dir=..., model=..., save_dir=...)
        """
        if hasattr(inferencer, "run_inference"):
            inferencer.run_inference(input_dir=input_dir, model=model_id, save_dir=save_dir)
        else:
            # Fallback naming
            inferencer.run_folder(input_dir=input_dir, model=model_id, save_dir=save_dir)

    # ---------------- Results navigation / Preview ---------------- #
    def _on_load_results(self):
        """
        Initialize (or refresh) navigation list from current input/save paths
        and show the first result.
        """
        input_path = self.input_path_edit.text().strip()
        save_dir = self.save_path.text().strip()
        if not input_path or not save_dir:
            QMessageBox.information(self, "Choose paths", "Please set input and save folder first.")
            return

        self._build_nav_pairs(input_path, save_dir)
        if not self._nav_pairs:
            QMessageBox.information(self, "Nothing to show", "No output files found in the save folder.")
            return

        self._nav_index = 0
        self._update_nav_buttons()
        self._show_nav_index()

    def _build_nav_pairs(self, input_path: str, save_dir: str):
        """
        Build [(input_image_path, output_label_path)] pairs by matching stems.
        """
        self._nav_pairs = []
        if os.path.isdir(input_path):
            images = self._list_images(input_path)
            outputs = self._list_images(save_dir)
            out_by_stem = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in outputs}
            for img in images:
                stem = os.path.splitext(os.path.basename(img))[0].lower()
                if stem in out_by_stem:
                    self._nav_pairs.append((img, out_by_stem[stem]))
        else:
            # single file input: try to find a single matching output by stem
            img = input_path
            stem = os.path.splitext(os.path.basename(img))[0].lower()
            outputs = self._list_images(save_dir)
            for out in outputs:
                if os.path.splitext(os.path.basename(out))[0].lower() == stem:
                    self._nav_pairs.append((img, out))
                    break

    def _list_images(self, folder: str):
        files = []
        try:
            for name in sorted(os.listdir(folder)):
                if name.lower().endswith(self._IMG_EXTS):
                    files.append(os.path.join(folder, name))
        except Exception:
            pass
        return files

    def _step_nav(self, delta: int):
        if not self._nav_pairs:
            return
        self._nav_index = max(0, min(len(self._nav_pairs) - 1, self._nav_index + delta))
        self._update_nav_buttons()
        self._show_nav_index()

    def _update_nav_buttons(self):
        n = len(self._nav_pairs)
        i = self._nav_index
        self.btn_prev.setEnabled(n > 0 and i > 0)
        self.btn_next.setEnabled(n > 0 and i < n - 1)

    def _show_nav_index(self):
        if not self._nav_pairs or self._nav_index < 0:
            return
        img_path, lbl_path = self._nav_pairs[self._nav_index]
        # load into current napari viewer
        viewer = self._find_parent_viewer()
        if viewer is None:
            QMessageBox.warning(self, "No viewer found",
                                "Couldn't locate a napari.Viewer on the parent chain.")
            return

        img = self._read_image(img_path)
        lbl = self._read_image(lbl_path)

        # replace previous preview layers
        self._remove_layer_if_exists(viewer, "Infer-Image")
        self._remove_layer_if_exists(viewer, "Infer-Labels")
        viewer.add_image(img, name="Infer-Image")
        viewer.add_labels(lbl, name="Infer-Labels")
        try:
            viewer.layers.selection = {viewer.layers["Infer-Labels"]}
        except Exception:
            pass

    # ---------------- Viewer helpers ---------------- #
    def _find_parent_viewer(self):
        w = self
        while w is not None:
            if hasattr(w, "viewer"):
                return getattr(w, "viewer")
            w = w.parent()
        return None

    def _remove_layer_if_exists(self, viewer, name: str):
        try:
            if name in viewer.layers:
                del viewer.layers[name]
        except Exception:
            pass

    def _read_image(self, path: str):
        import imageio.v3 as iio
        return iio.imread(path)
