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
from torch.utils.data import DataLoader
from utilities.unet_2d_dataset_builder import SegmentationDataset
from PIL import Image
from torchvision import transforms
from models.unet_2d import UNet
import torch
import tifffile
from pathlib import Path
import time

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
        self.model_source.addItems(["Local","Hugging Face", "Bioimage IO", "CBI Model Zoo"])
        src_row.addWidget(self.model_source, 1)
        lay.addLayout(src_row)

        # --- Model selector: URL (for web sources) OR local file browse ---
        self.model_stack = QtWidgets.QStackedWidget()
        # URL page
        url_page = QWidget(); url_lay = QHBoxLayout(url_page); url_lay.setContentsMargins(0, 0, 0, 0); url_lay.setSpacing(8)
        self.model_url_edit = QLineEdit(); self.model_url_edit.setPlaceholderText("Enter model URL or identifier…")
        url_lay.addWidget(QLabel("Model")); url_lay.addWidget(self.model_url_edit, 1)
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

    def _on_model_source_changed(self, text: str):
        """
        Non-Local -> URL entry (no local type row)
        Local     -> file browse + show local task type dropdown
        """
        is_local = text.lower().strip() == "local"
        self.model_stack.setCurrentIndex(1 if is_local else 0)
        # show/hide the model type chooser only for Local
        for i in range(self.local_type_row.count()):
            item = self.local_type_row.itemAt(i)
            if item.widget():
                item.widget().setVisible(is_local)

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

        if os.path.isdir(input_path):
            self._run_inference_folder(None, input_path, model_id, save_dir)
        else:
            self._run_inference_single_image(None, input_path, model_id, save_dir)


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

        if hasattr(inferencer, "run_inference_single"):
            inferencer.run_inference_single(image_path=image_path, model=model_id, save_dir=save_dir)
        else:
            # Fallback naming
            inferencer.run_single(image_path=image_path, model=model_id, save_dir=save_dir)

        """
        print("I am here to infer - single")
        choice = self.local_task_type.currentText().lower().strip()
        if "semantic 2d" in choice:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image = Image.open(image_path).convert("RGB")
            img_transform = transforms.ToTensor()
            image = img_transform(image)
            image = image.unsqueeze(0)
            model = UNet(3, 1)
            model.load_state_dict(torch.load(model_id, map_location=device))
            model.to(device)
            self.eval()
            with torch.no_grad():
                logits = self.forward(image.to(device))
                tifffile.imwrite(torch.argmax(logits, dim=1),save_dir+os.sep+"output.tif")

        elif "semantic 3d" in choice:
            key = "semantic-3d"
        elif "instance" in choice:
            key = "instance"
        else:
            key = "general"


    def _run_inference_folder(self, inferencer, input_dir: str, model_id: str, save_dir: str):
        """
        Call the inferencer's folder API.
        Expected: inferencer.run_inference(input_dir=..., model=..., save_dir=...)

        if hasattr(inferencer, "run_inference"):
            inferencer.run_inference(input_dir=input_dir, model=model_id, save_dir=save_dir)
        else:
            # Fallback naming
            inferencer.run_folder(input_dir=input_dir, model=model_id, save_dir=save_dir)
        """
        print("I am here to infer all")
        choice = self.local_task_type.currentText().lower().strip()
        if "semantic 2d" in choice:
            num_channels = 3
            num_classes = 2
            model = UNet(num_channels, num_classes)
            dataset = SegmentationDataset(input_dir,save_dir)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for i, images in enumerate(dataloader):
                infer_data = {
                    "weights_path": model_id,
                    "images": images,
                    "device": device
                }
                infer_masks = model.infer(infer_data)

                # mask_np = infer_masks[0].cpu().numpy().astype("uint8") * 255
                mask_np = infer_masks[0].cpu().numpy().astype("uint16") * 65535
                out_path = Path(save_dir) / f"infer_mask_{i}.tif"
                Image.fromarray(mask_np).save(out_path)

        elif "semantic 3d" in choice:
            key = "semantic-3d"
            time.sleep(10)

        elif "instance" in choice:
            key = "instance"
            time.sleep(10)
        else:
            key = "general"

        QMessageBox.information(self, "Inference", "Completed inference.")

    # ---------------- Results navigation / Preview ---------------- #
    def _maybe_handle_instance_preview(self, input_path: str, save_dir: str) -> bool:
        """
        If the current task type is 'instance', build navigation pairs where each item
        holds (image_path, {class_name: class_label_path}) based on subfolders.
        Return True if instance mode was detected and prepared, so callers can 'return' early.
        """
        choice = self.local_task_type.currentText().lower().strip()
        if "instance" not in choice:
            self._instance_mode = False
            return False

        self._instance_mode = True
        self._nav_pairs = []

        # collect class subfolders inside `save_dir`
        if not os.path.isdir(save_dir):
            return True  # nothing to do, but we handled the 'instance' branch

        class_dirs = [
            d for d in sorted(os.listdir(save_dir))
            if os.path.isdir(os.path.join(save_dir, d))
        ]
        if not class_dirs:
            return True

        # Helper to map: stem -> {class_name: label_path}
        def _collect_class_maps_for_folder(images_list):
            # For each class, build a stem->path table
            per_class_maps = {}
            for cls in class_dirs:
                cls_folder = os.path.join(save_dir, cls)
                cls_outputs = self._list_images(cls_folder)  # flat files per class
                per_class_maps[cls] = {
                    os.path.splitext(os.path.basename(p))[0].lower(): p
                    for p in cls_outputs
                }

            for img in images_list:
                stem = os.path.splitext(os.path.basename(img))[0].lower()
                class_map = {}
                for cls, stem_map in per_class_maps.items():
                    if stem in stem_map:
                        class_map[cls] = stem_map[stem]
                if class_map:
                    self._nav_pairs.append((img, class_map))

        if os.path.isdir(input_path):
            images = self._list_images(input_path)
            _collect_class_maps_for_folder(images)
        else:
            # Single-file input: match labels of same stem across all class subfolders
            img = input_path
            stem = os.path.splitext(os.path.basename(img))[0].lower()
            class_map = {}
            for cls in class_dirs:
                cls_folder = os.path.join(save_dir, cls)
                for out in self._list_images(cls_folder):
                    if os.path.splitext(os.path.basename(out))[0].lower() == stem:
                        class_map[cls] = out
                        break
            if class_map:
                self._nav_pairs.append((img, class_map))

        return True  # instance mode handled (even if no pairs found)


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

        # NEW: branch early for instance mode
        if self._maybe_handle_instance_preview(input_path, save_dir):
            if not getattr(self, "_nav_pairs", None):
                QMessageBox.information(self, "Nothing to show", "No output files found in the save folder.")
                return
            self._nav_index = 0
            self._update_nav_buttons()
            self._show_nav_index()  # will know how to display multiple class labels
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

        # Either (img_path, lbl_path) OR (img_path, {class_name: lbl_path})
        img_path, lbl_info = self._nav_pairs[self._nav_index]

        viewer = self._find_parent_viewer()
        if viewer is None:
            QMessageBox.warning(self, "No viewer found", "Couldn't locate a napari.Viewer on the parent chain.")
            return

        img = self._read_image(img_path)

        # Always replace previous preview image layer
        self._remove_layer_if_exists(viewer, "Infer-Image")
        viewer.add_image(img, name="Infer-Image")

        if getattr(self, "_instance_mode", False) and isinstance(lbl_info, dict):
            # Remove any previously created instance label layers
            self._remove_layers_with_prefix(viewer, "Infer-Instance-")

            # Add each class map as its own label layer
            last_layer = None
            for cls_name, lbl_path in sorted(lbl_info.items()):
                lbl = self._read_image(lbl_path)
                layer_name = f"Infer-Instance-{cls_name}"
                self._remove_layer_if_exists(viewer, layer_name)
                last_layer = viewer.add_labels(lbl, name=layer_name)

            # Select the last-added label layer if available
            try:
                if last_layer is not None:
                    viewer.layers.selection = {last_layer}
            except Exception:
                pass

        else:
            # Original single-label behavior
            lbl_path = lbl_info
            lbl = self._read_image(lbl_path)
            self._remove_layer_if_exists(viewer, "Infer-Labels")
            viewer.add_labels(lbl, name="Infer-Labels")
            try:
                viewer.layers.selection = {viewer.layers["Infer-Labels"]}
            except Exception:
                pass

    def _remove_layers_with_prefix(self, viewer, prefix: str):
        # Make a copy of names since we'll modify layers
        names = [lyr.name for lyr in list(viewer.layers)]
        for nm in names:
            if nm.startswith(prefix):
                self._remove_layer_if_exists(viewer, nm)
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
