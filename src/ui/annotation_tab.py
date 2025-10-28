# FILE: ui/annotation_tab.py

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QListWidget, QListWidgetItem, QComboBox, QMessageBox, QSizePolicy
)

from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ui.common import Card
from ui.state import state
from data_annotation.lasso_tool import LassoAnnotationTool


class AnnotationTab(QWidget):
    """
    Lists input images, lets you open/create labels in the existing napari viewer,
    enables quick 'advanced' actions (stubs), and can navigate to Data Processing.

    Buttons:
      - Create/Load Labels: open selected image + its label (or empty label) in viewer
      - Save Label: save current 'Anno-Labels' layer to the labels folder
      - Enable/Disable Lasso: toggle (dummy) lasso tool hooks
      - Interpolate 3D: run stub; has a dropdown to pick 'Method 1' or 'Method 2'
      - SAM: run stub
      - Skip: navigate to 'Data Processing'
      - Continue: navigate to 'Data Processing'
    """
    proceed = QtCore.Signal(dict)  # emits when Continue/Skip pressed

    _IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._selected_filename = ""       # just the file name
        self._last_label_save_path = ""    # full path where labels will be saved

        # cache last-known dirs to avoid pointless repopulates (optional)
        self._last_seen_img_dir = None
        self._last_seen_lbl_dir = None
        
        # Lasso annotation tool instance
        self._lasso_tool = None

        self._build_ui()
        # Initial populate (may be empty if state not set yet; showEvent will refresh again)
        self._populate_list(force=True)

    # --- ensures we refresh the listing whenever this tab becomes visible (Fix #1) ---
    def showEvent(self, e: QtGui.QShowEvent) -> None:
        super().showEvent(e)
        self._populate_list()  # re-read state.* each time we show

    # ------------------------ UI ------------------------ #
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        # --- Filters / controls above the list ---
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.chk_filter_unlabeled = QCheckBox("Filter images without any labels")
        self.chk_filter_unlabeled.setChecked(False)
        top_row.addWidget(self.chk_filter_unlabeled)
        top_row.addStretch(1)

        root.addLayout(top_row)

        # --- Card: image list (scrollable via QListWidget's own scroll) ---
        self.list_card = Card()
        list_lay = self.list_card.layout()
        title = QLabel("Images"); title.setObjectName("H2")
        list_lay.addWidget(title)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.SingleSelection)
        self.image_list.setUniformItemSizes(True)
        self.image_list.setAlternatingRowColors(False)
        self.image_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        list_lay.addWidget(self.image_list, 1)

        # Row: Create/Load and Save
        row_actions = QHBoxLayout()
        row_actions.setSpacing(8)
        self.btn_open = QPushButton("Create/Load Labels")
        self.btn_open.setObjectName("PrimaryBtn")
        self.btn_save = QPushButton("Save Label")
        self.btn_save.setObjectName("SecondaryBtn")
        row_actions.addWidget(self.btn_open)
        row_actions.addWidget(self.btn_save)
        row_actions.addStretch(1)
        list_lay.addLayout(row_actions)

        self.list_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.list_card, 1)

        # --- Advanced Annotation Features ---
        adv = Card()
        adv_lay = adv.layout()
        adv_title = QLabel("Advanced Annotation Features"); adv_title.setObjectName("H2")
        adv_lay.addWidget(adv_title)

        # Lasso toggle
        lasso_row = QHBoxLayout()
        lasso_row.setSpacing(8)
        self.btn_lasso = QPushButton("Enable Lasso")
        self.btn_lasso.setCheckable(True)
        lasso_row.addWidget(self.btn_lasso)
        lasso_row.addStretch(1)
        adv_lay.addLayout(lasso_row)

        # Interpolate 3D FIRST, then methods dropdown (Fix #2)
        interp_row = QHBoxLayout()
        interp_row.setSpacing(8)
        self.btn_interp = QPushButton("Interpolate 3D (Method 1)")
        self.cmb_interp = QComboBox()
        self.cmb_interp.addItems(["Method 1", "Method 2"])

        # order: button -> label -> combo -> stretch
        interp_row.addWidget(self.btn_interp)
        interp_row.addWidget(QLabel("Method"))
        interp_row.addWidget(self.cmb_interp)
        interp_row.addStretch(1)
        adv_lay.addLayout(interp_row)

        # SAM stub
        sam_row = QHBoxLayout()
        sam_row.setSpacing(8)
        self.btn_sam = QPushButton("SAM")
        sam_row.addWidget(self.btn_sam)
        sam_row.addStretch(1)
        adv_lay.addLayout(sam_row)

        root.addWidget(adv)

        # --- Bottom actions: Skip / Continue ---
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_skip = QPushButton("Skip")          # goes to Data Processing
        self.btn_skip.setObjectName("SecondaryBtn")
        self.btn_continue = QPushButton("Continue")  # goes to Data Processing
        self.btn_continue.setObjectName("CTA")
        bottom.addWidget(self.btn_skip)
        bottom.addWidget(self.btn_continue)
        root.addLayout(bottom)

        # events
        self.chk_filter_unlabeled.toggled.connect(lambda _: self._populate_list(force=True))
        self.image_list.currentItemChanged.connect(self._on_list_selection)
        self.btn_open.clicked.connect(self._on_create_or_load)
        self.btn_save.clicked.connect(self._on_save_label)

        self.btn_lasso.clicked.connect(self._on_toggle_lasso)
        self.cmb_interp.currentTextChanged.connect(self._on_interp_method_changed)
        self.btn_interp.clicked.connect(self._on_interpolate_3d)
        self.btn_sam.clicked.connect(self._on_sam)

        self.btn_skip.clicked.connect(self._on_skip)
        self.btn_continue.clicked.connect(self._on_continue)

    # ---------------------- Data / Listing ---------------------- #
    def _populate_list(self, force: bool = False):
        """
        List images from state.input_img_dir.
        If filter is checked, only include images that DO NOT have a matching label in state.input_lbl_dir.
        """
        import os

        img_dir = (state.input_img_dir or "").strip()
        lbl_dir = (state.input_lbl_dir or "").strip()

        # avoid redundant work unless forced or dirs changed
        if not force and img_dir == self._last_seen_img_dir and lbl_dir == self._last_seen_lbl_dir:
            return
        self._last_seen_img_dir, self._last_seen_lbl_dir = img_dir, lbl_dir

        self.image_list.clear()

        if not img_dir or not os.path.isdir(img_dir):
            self.image_list.addItem(QListWidgetItem("⚠ No valid input image folder set"))
            self.image_list.setEnabled(False)
            return

        self.image_list.setEnabled(True)

        # collect image files
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(self._IMG_EXTS)]
        files.sort()

        show_unlabeled_only = self.chk_filter_unlabeled.isChecked() and lbl_dir and os.path.isdir(lbl_dir)

        count = 0
        for f in files:
            if show_unlabeled_only:
                if self._label_path_if_exists(f) is not None:
                    continue  # has a label, skip
            item = QListWidgetItem(f)
            item.setData(Qt.UserRole, f)
            self.image_list.addItem(item)
            count += 1

        if count == 0:
            note = "No images found."
            if show_unlabeled_only:
                note = "No unlabeled images found."
            self.image_list.addItem(QListWidgetItem(f"ℹ {note}"))

    def _on_list_selection(self, current: QListWidgetItem, previous: QListWidgetItem):
        self._selected_filename = current.data(Qt.UserRole) if current else ""

    def _label_path_if_exists(self, image_filename: str):
        """
        Return full path to label if found (same basename, any supported ext), else None.
        """
        import os
        lbl_dir = (state.input_lbl_dir or "").strip()
        if not lbl_dir or not os.path.isdir(lbl_dir):
            return None
        base = os.path.splitext(image_filename)[0].lower()
        for ext in self._IMG_EXTS:
            p = os.path.join(lbl_dir, base + ext)
            if os.path.isfile(p):
                return p
        # also try exact filename as-is
        p2 = os.path.join(lbl_dir, image_filename)
        if os.path.isfile(p2):
            return p2
        return None

    # ---------------------- Viewer helpers ---------------------- #
    def _find_parent_viewer(self):
        """
        Walk up the QWidget parent chain and return the first object that has a `viewer` attr.
        (Your top-level `Lab` keeps `self.viewer`, so this finds the napari.Viewer.)
        """
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

    # ---------------------- I/O helpers ---------------------- #
    def _read_image(self, path: str):
        import imageio.v3 as iio
        return iio.imread(path)

    def _ensure_empty_label_for(self, img):
        """
        Create an empty labels array (zeros) matching the spatial shape of `img`.
        - If img is (H,W,C), labels -> (H,W).
        - If img is (Z,H,W) or (Z,H,W,C), labels -> (Z,H,W).
        """
        import numpy as np
        if img.ndim == 2:       # H, W
            shape = img.shape
        elif img.ndim == 3:
            if img.shape[-1] in (3, 4):  # H, W, C
                shape = img.shape[:2]
            else:                         # Z, H, W
                shape = img.shape
        elif img.ndim == 4:               # Z, H, W, C
            shape = img.shape[:3]
        else:
            shape = img.shape[:-1] if img.shape[-1] <= 4 else img.shape
        return np.zeros(shape, dtype="uint16")

    def _default_label_save_path(self, image_filename: str):
        """
        Suggest a label save path in the labels folder with .tif extension.
        """
        import os
        lbl_dir = (state.input_lbl_dir or "").strip()
        base = os.path.splitext(image_filename)[0]
        return os.path.join(lbl_dir, base + ".tif") if lbl_dir else ""

    # ---------------------- Actions: Load/Save ---------------------- #
    def _on_create_or_load(self):
        """
        Open selected image; add image layer and labels layer (existing or empty) in current viewer.
        """
        import os
        if not self._selected_filename:
            QMessageBox.information(self, "Select an image", "Please select an image from the list.")
            return

        img_dir = (state.input_img_dir or "").strip()
        lbl_dir = (state.input_lbl_dir or "").strip()
        if not img_dir or not os.path.isdir(img_dir):
            QMessageBox.warning(self, "Missing image folder", "Input image folder is not set.")
            return

        img_path = os.path.join(img_dir, self._selected_filename)
        if not os.path.isfile(img_path):
            QMessageBox.warning(self, "File missing", f"Image not found:\n{img_path}")
            return

        viewer = self._find_parent_viewer()
        if viewer is None:
            QMessageBox.warning(self, "No viewer found",
                                "Couldn't locate a napari.Viewer on the parent chain.")
            return

        # read image & label (or create empty)
        img = self._read_image(img_path)
        lbl_path = self._label_path_if_exists(self._selected_filename)
        if lbl_path is not None:
            lbl = self._read_image(lbl_path)
            self._last_label_save_path = lbl_path
        else:
            lbl = self._ensure_empty_label_for(img)
            self._last_label_save_path = self._default_label_save_path(self._selected_filename)

        # clear previous anno layers
        self._remove_layer_if_exists(viewer, "Anno-Image")
        self._remove_layer_if_exists(viewer, "Anno-Labels")
        # add layers
        viewer.add_image(img, name="Anno-Image")
        viewer.add_labels(lbl, name="Anno-Labels")

        # focus on labels for immediate editing
        try:
            viewer.layers.selection = {viewer.layers["Anno-Labels"]}
        except Exception:
            pass

        print(f"[AnnotationTab] Opened image: {img_path}")
        print(f"[AnnotationTab] Labels path: {self._last_label_save_path} (existing={lbl_path is not None})")

    def _on_save_label(self):
        """
        Save current 'Anno-Labels' layer to the labels folder with matching name.
        """
        import os
        import imageio.v3 as iio
        from numpy import asanyarray

        viewer = self._find_parent_viewer()
        if viewer is None:
            QMessageBox.warning(self, "No viewer found",
                                "Couldn't locate a napari.Viewer on the parent chain.")
            return

        if "Anno-Labels" not in viewer.layers:
            QMessageBox.information(self, "No label layer", "There's no 'Anno-Labels' layer to save.")
            return

        if not self._selected_filename:
            QMessageBox.information(self, "Select an image", "Please select an image to determine save name.")
            return

        lbl_dir = (state.input_lbl_dir or "").strip()
        if not lbl_dir:
            QMessageBox.warning(self, "Missing labels folder",
                                "Input label folder is not set. Please set it in the Project page.")
            return
        os.makedirs(lbl_dir, exist_ok=True)

        data = asanyarray(viewer.layers["Anno-Labels"].data)
        save_path = self._last_label_save_path or self._default_label_save_path(self._selected_filename)
        try:
            iio.imwrite(save_path, data)
            QMessageBox.information(self, "Saved", f"Label saved to:\n{save_path}")
            print(f"[AnnotationTab] Saved label: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")

    # ---------------------- Advanced stubs ---------------------- #
    def _on_toggle_lasso(self):
        """
        Toggle lasso enable/disable. Replace with your real hooks.
        """
        if self.btn_lasso.isChecked():
            self.btn_lasso.setText("Disable Lasso")
            self._enable_lasso()
        else:
            self.btn_lasso.setText("Enable Lasso")
            self._disable_lasso()

    def _enable_lasso(self):
        """Enable the lasso annotation tool."""
        viewer = self._find_parent_viewer()
        if viewer is None:
            QMessageBox.warning(self, "No viewer found",
                                "Couldn't locate a napari.Viewer. Please load an image first.")
            self.btn_lasso.setChecked(False)
            return
            
        # Check if image and labels layers exist
        if "Anno-Image" not in viewer.layers:
            QMessageBox.warning(self, "No image loaded",
                                "Please load an image first using 'Create/Load Labels' button.")
            self.btn_lasso.setChecked(False)
            return
            
        if "Anno-Labels" not in viewer.layers:
            QMessageBox.warning(self, "No labels layer",
                                "Please load an image first using 'Create/Load Labels' button.")
            self.btn_lasso.setChecked(False)
            return
        
        try:
            # Create and activate lasso tool
            if self._lasso_tool is None:
                self._lasso_tool = LassoAnnotationTool()
                
            self._lasso_tool.activate(viewer=viewer, 
                                     image_layer_name="Anno-Image",
                                     labels_layer_name="Anno-Labels")
            
            # Get the current selected label value from the labels layer
            labels_layer = viewer.layers["Anno-Labels"]
            if hasattr(labels_layer, 'selected_label'):
                self._lasso_tool.set_label_value(labels_layer.selected_label)
                
            print("[AnnotationTab] Lasso ENABLED - Draw a polygon around the object")
            QMessageBox.information(self, "Lasso Tool Activated",
                                  "Draw a polygon around the object you want to annotate.\n"
                                  "The tool will automatically snap to nearby edges.\n\n"
                                  "Tips:\n"
                                  "- Draw a rough shape around the object\n"
                                  "- The shape will be refined to match object boundaries\n"
                                  "- Press ESC to cancel the current polygon\n"
                                  "- Disable the lasso tool when done")
        except Exception as e:
            QMessageBox.critical(self, "Lasso activation failed", 
                               f"Failed to activate lasso tool: {type(e).__name__}: {e}")
            self.btn_lasso.setChecked(False)
            print(f"[AnnotationTab] Lasso activation error: {e}")

    def _disable_lasso(self):
        """Disable the lasso annotation tool."""
        if self._lasso_tool is not None and self._lasso_tool.is_active:
            try:
                self._lasso_tool.deactivate()
                print("[AnnotationTab] Lasso DISABLED")
            except Exception as e:
                print(f"[AnnotationTab] Error disabling lasso: {e}")
        else:
            print("[AnnotationTab] Lasso was not active")

    def _on_interp_method_changed(self, text: str):
        # reflect the choice in the button label
        self.btn_interp.setText(f"Interpolate 3D ({text})")

    def _on_interpolate_3d(self):
        method = self.cmb_interp.currentText()
        print(f"[AnnotationTab] Interpolate 3D invoked with {method}")

    def _on_sam(self):
        print("[AnnotationTab] SAM invoked")

    # ---------------------- Navigation ---------------------- #
    def _on_skip(self):
        """
        Go to Data Processing tab without doing anything.
        """
        self._goto_dataproc_tab()
        self.proceed.emit({"action": "skip"})

    def _on_continue(self):
        """
        Go to Data Processing tab without doing anything.
        """
        self._goto_dataproc_tab()
        self.proceed.emit({"action": "continue"})

    def _goto_dataproc_tab(self) -> bool:
        """
        Find parent QTabWidget and select 'Data Processing' tab by name.
        """
        w = self.parent()
        while w is not None:
            if isinstance(w, QtWidgets.QTabWidget):
                for i in range(w.count()):
                    if w.tabText(i).strip().lower() == "data processing":
                        w.setCurrentIndex(i)
                        return True
                # Fallback to index 1 if tab order is [Annotation, Data Processing, Training]
                if w.count() >= 2:
                    w.setCurrentIndex(1)
                    return True
                return False
            w = w.parent()
        return False
