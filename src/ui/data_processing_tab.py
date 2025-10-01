from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QHBoxLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy, QFormLayout, QLineEdit, QMessageBox
)

from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ui.common import Card, DropLineEdit, labeled_row
from ui.state import state


# --------------------- Small input helpers --------------------- #
class RangeField(QWidget):
    """Two-number range field: (min, max)."""
    changed = QtCore.Signal()

    def __init__(self, minimum=0.0, maximum=1.0, step=0.01, decimals=3, default=(0.0, 1.0), parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.min = QDoubleSpinBox()
        self.max = QDoubleSpinBox()
        for sb in (self.min, self.max):
            sb.setDecimals(decimals)
            sb.setRange(minimum, maximum)
            sb.setSingleStep(step)
            sb.setButtonSymbols(QDoubleSpinBox.NoButtons)
            sb.setFixedWidth(90)
            sb.valueChanged.connect(self.changed.emit)
        self.min.setValue(default[0])
        self.max.setValue(default[1])
        lay.addWidget(QLabel("min"))
        lay.addWidget(self.min)
        lay.addWidget(QLabel("max"))
        lay.addWidget(self.max)
        lay.addStretch(1)

    def get(self):
        return (float(self.min.value()), float(self.max.value()))

    def set(self, t):
        self.min.setValue(float(t[0]))
        self.max.setValue(float(t[1]))


class TripleField(QWidget):
    """Three-number field: (a, b, c). For mean/std triples."""
    changed = QtCore.Signal()

    def __init__(self, minimum=0.0, maximum=5.0, step=0.01, decimals=3, default=(0.0, 0.0, 0.0), parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.a = QDoubleSpinBox()
        self.b = QDoubleSpinBox()
        self.c = QDoubleSpinBox()
        for sb in (self.a, self.b, self.c):
            sb.setDecimals(decimals)
            sb.setRange(minimum, maximum)
            sb.setSingleStep(step)
            sb.setButtonSymbols(QDoubleSpinBox.NoButtons)
            sb.setFixedWidth(70)
            sb.valueChanged.connect(self.changed.emit)
        self.set(default)
        lay.addWidget(QLabel("x"))
        lay.addWidget(self.a)
        lay.addWidget(QLabel("y"))
        lay.addWidget(self.b)
        lay.addWidget(QLabel("z"))
        lay.addWidget(self.c)
        lay.addStretch(1)

    def get(self):
        return (float(self.a.value()), float(self.b.value()), float(self.c.value()))

    def set(self, t):
        a, b, c = t
        self.a.setValue(float(a))
        self.b.setValue(float(b))
        self.c.setValue(float(c))


class TransformSection(Card):
    """
    A Card with a checkbox title that enables/disables its parameter area.
    """
    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._lay = self.layout()
        self.chk = QCheckBox(title)
        self.chk.setChecked(False)
        self._lay.insertWidget(0, self.chk)
        self._params = QWidget()
        self._params_lay = QFormLayout(self._params)
        self._params_lay.setContentsMargins(0, 6, 0, 0)
        self._params_lay.setSpacing(8)
        self._lay.addWidget(self._params)
        self._set_enabled(False)
        self.chk.toggled.connect(self._set_enabled)
        self.chk.toggled.connect(self.toggled.emit)

    def _set_enabled(self, on: bool):
        self._params.setEnabled(on)
        self.setProperty("active", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def add_row(self, label: str, widget: QWidget):
        self._params_lay.addRow(QLabel(label), widget)

    def is_enabled(self) -> bool:
        return self.chk.isChecked()


# --------------------- Main Tab --------------------- #
class DataProcessingTab(QWidget):
    """
    UI for building an augmentation config + quick test actions.

    NOTE:
      - Input *image/label* directories are NOT shown here. They come from ui.state.
      - Output *image/label* directories ARE configurable here.

    Signals:
        requested_run(dict): emits full config for running the whole pipeline (still available).
        proceed_to_train(): emitted when Skip or Apply & Continue want to navigate to the Training tab.
    """
    requested_run = QtCore.Signal(dict)
    proceed_to_train = QtCore.Signal()

    _IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Read paths/task directly from global state (requirement #1)
        self._input_img_dir = state.input_img_dir
        self._input_lbl_dir = state.input_lbl_dir
        self._task = (state.task or "").strip()

        self._build_ui()
        self._apply_task_rules()  # enforce 3D limitations for specific transforms

    # ---------------- UI ---------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        # Scrollable body to hold all cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)  # allow horizontal scroll when needed

        body = QWidget()
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2, 2, 2, 2)
        body_lay.setSpacing(DEFAULT_SPACING)

        # ---- Outputs & iterations (inputs come from state) ----
        io_card = Card()
        io_lay = io_card.layout()
        title = QLabel("Outputs & Iterations"); title.setObjectName("H2")
        io_lay.addWidget(title)

        # Output dirs are editable here
        self.output_img_dir = DropLineEdit("Drop or browse output image folder…")
        btn_out_img = QPushButton("Browse"); btn_out_img.setObjectName("PrimaryBtn")
        io_lay.addLayout(labeled_row("Output images", self.output_img_dir, btn_out_img))

        self.output_lbl_dir = DropLineEdit("Drop or browse output label folder…")
        btn_out_lbl = QPushButton("Browse"); btn_out_lbl.setObjectName("SecondaryBtn")
        io_lay.addLayout(labeled_row("Output labels", self.output_lbl_dir, btn_out_lbl))

        iter_row = QHBoxLayout()
        iter_row.setSpacing(8)
        iter_row.addWidget(QLabel("Iterations"))
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 100000)
        self.iterations.setValue(2)
        self.iterations.setFixedWidth(90)
        iter_row.addWidget(self.iterations)
        iter_row.addStretch(1)
        io_lay.addLayout(iter_row)

        body_lay.addWidget(io_card)

        # ---- Transform sections ----
        xforms_title = QLabel("Transforms"); xforms_title.setObjectName("H2")
        body_lay.addWidget(xforms_title)

        # Rotation
        self.sec_rotation = TransformSection("Rotation")
        self.rotation_degrees = RangeField(minimum=-180, maximum=180, step=1, decimals=0, default=(-60, 60))
        self.sec_rotation.add_row("Degrees (min,max)", self.rotation_degrees)
        body_lay.addWidget(self.sec_rotation)

        # Translation
        self.sec_translation = TransformSection("Translation")
        self.translation_shift = RangeField(minimum=-1.0, maximum=1.0, step=0.01, decimals=3, default=(-0.1, 0.1))
        self.sec_translation.add_row("Shift (min,max) fraction of size", self.translation_shift)
        body_lay.addWidget(self.sec_translation)

        # Elastic (2D-only per requirement #3)
        self.sec_elastic = TransformSection("Elastic")
        self.elastic_alpha = QDoubleSpinBox(); self.elastic_alpha.setRange(0.0, 10.0); self.elastic_alpha.setSingleStep(0.1); self.elastic_alpha.setValue(1.0); self.elastic_alpha.setFixedWidth(90)
        self.elastic_sigma = QDoubleSpinBox(); self.elastic_sigma.setRange(0.0, 500.0); self.elastic_sigma.setSingleStep(1.0); self.elastic_sigma.setValue(50.0); self.elastic_sigma.setFixedWidth(90)
        self.sec_elastic.add_row("alpha", self.elastic_alpha)
        self.sec_elastic.add_row("sigma", self.elastic_sigma)
        body_lay.addWidget(self.sec_elastic)

        # Erasing (2D-only per requirement #3)
        self.sec_erasing = TransformSection("Erasing")
        self.erasing_scale = RangeField(minimum=0.0, maximum=1.0, step=0.01, decimals=3, default=(0.02, 0.33))
        self.erasing_ratio = RangeField(minimum=0.1, maximum=10.0, step=0.1, decimals=2, default=(0.3, 3.3))
        self.erasing_p = QDoubleSpinBox(); self.erasing_p.setRange(0.0, 1.0); self.erasing_p.setSingleStep(0.05); self.erasing_p.setValue(1.0); self.erasing_p.setFixedWidth(90)
        self.sec_erasing.add_row("Scale (min,max)", self.erasing_scale)
        self.sec_erasing.add_row("Ratio (min,max)", self.erasing_ratio)
        self.sec_erasing.add_row("p", self.erasing_p)
        body_lay.addWidget(self.sec_erasing)

        # Scaling
        self.sec_scaling = TransformSection("Scaling")
        self.scaling_scale = RangeField(minimum=0.1, maximum=3.0, step=0.01, decimals=3, default=(0.8, 1.2))
        self.sec_scaling.add_row("Scale (min,max)", self.scaling_scale)
        body_lay.addWidget(self.sec_scaling)

        # Normalize
        self.sec_normalize = TransformSection("Normalize")
        self.norm_mean = TripleField(minimum=0.0, maximum=1.0, step=0.01, decimals=3, default=(0.0, 0.0, 0.0))
        self.norm_std  = TripleField(minimum=0.0, maximum=5.0, step=0.01, decimals=3, default=(1.0, 1.0, 1.0))
        self.norm_max  = QDoubleSpinBox(); self.norm_max.setRange(1.0, 65535.0); self.norm_max.setSingleStep(1.0); self.norm_max.setValue(255.0); self.norm_max.setFixedWidth(110)
        self.norm_p    = QDoubleSpinBox(); self.norm_p.setRange(0.0, 1.0); self.norm_p.setSingleStep(0.05); self.norm_p.setValue(1.0); self.norm_p.setFixedWidth(90)
        self.sec_normalize.add_row("Mean (x,y,z)", self.norm_mean)
        self.sec_normalize.add_row("Std (x,y,z)", self.norm_std)
        self.sec_normalize.add_row("Max pixel value", self.norm_max)
        self.sec_normalize.add_row("p", self.norm_p)
        body_lay.addWidget(self.sec_normalize)

        # Horizontal flip
        self.sec_hflip = TransformSection("Horizontal Flip")
        self.hflip_p = QDoubleSpinBox(); self.hflip_p.setRange(0.0, 1.0); self.hflip_p.setSingleStep(0.05); self.hflip_p.setValue(1.0); self.hflip_p.setFixedWidth(90)
        self.sec_hflip.add_row("p", self.hflip_p)
        body_lay.addWidget(self.sec_hflip)

        # Vertical flip
        self.sec_vflip = TransformSection("Vertical Flip")
        self.vflip_p = QDoubleSpinBox(); self.vflip_p.setRange(0.0, 1.0); self.vflip_p.setSingleStep(0.05); self.vflip_p.setValue(1.0); self.vflip_p.setFixedWidth(90)
        self.sec_vflip.add_row("p", self.vflip_p)
        body_lay.addWidget(self.sec_vflip)

        # ---------- Bottom area ----------
        # Quick test actions JUST ABOVE the apply/continue row
        test_row = QHBoxLayout()
        test_row.setSpacing(8)

        self.btn_random = QPushButton("Run on one Random Image")
        self.btn_random.setObjectName("GhostBtn")

        self.sample_name = QLineEdit()
        self.sample_name.setPlaceholderText("Enter image name (with or without extension)")
        self.sample_name.setMinimumWidth(160)

        self.btn_sample = QPushButton("Run on a Sample")
        self.btn_sample.setObjectName("PrimaryBtn")

        test_row.addWidget(self.btn_random)
        test_row.addStretch(1)
        test_row.addWidget(QLabel("Sample name"))
        test_row.addWidget(self.sample_name, 1)
        test_row.addWidget(self.btn_sample)
        body_lay.addLayout(test_row)

        # Final action row: Skip / Apply & Continue
        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addStretch(1)
        self.btn_skip = QPushButton("Skip")               # requirement #2
        self.btn_skip.setObjectName("SecondaryBtn")
        self.btn_run = QPushButton("Apply & Continue")
        self.btn_run.setObjectName("CTA")
        actions.addWidget(self.btn_skip)
        actions.addWidget(self.btn_run)
        body_lay.addLayout(actions)

        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # width guard (allow horizontal expansion; no max width cap)
        self.setMinimumWidth(380)

        # Events
        btn_out_img.clicked.connect(lambda: self._choose_dir(self.output_img_dir))
        btn_out_lbl.clicked.connect(lambda: self._choose_dir(self.output_lbl_dir))

        # Stubs for now (requirement #4)
        self.btn_random.clicked.connect(self._on_run_random_one)
        self.btn_sample.clicked.connect(self._on_run_sample)

        # Skip and Apply & Continue (requirement #2 and #5)
        self.btn_skip.clicked.connect(self._on_skip)
        self.btn_run.clicked.connect(self._on_apply_and_continue)

    # ---------------- Task-based UI rules ---------------- #
    def _apply_task_rules(self):
        """
        If task is semantic-3d: Elastic and Erasing are not allowed.
        Clicking their checkbox should politely inform the user and uncheck them.
        """
        if (self._task or "").lower() == "semantic-3d":
            # connect interceptors
            self.sec_elastic.chk.toggled.connect(lambda on: self._block_2d_only(self.sec_elastic, on))
            self.sec_erasing.chk.toggled.connect(lambda on: self._block_2d_only(self.sec_erasing, on))
            # ensure params are off initially
            if self.sec_elastic.is_enabled():
                self.sec_elastic.chk.setChecked(False)
            if self.sec_erasing.is_enabled():
                self.sec_erasing.chk.setChecked(False)

    def _block_2d_only(self, section: TransformSection, on: bool):
        if on:
            QMessageBox.information(
                self,
                "Not available for 3D",
                "Thanks for checking this! Elastic/Erasing transforms are available for 2D tasks only right now. "
                "They’re disabled for Semantic 3D."
            )
            # revert to OFF
            section.chk.blockSignals(True)
            section.chk.setChecked(False)
            section.chk.blockSignals(False)
            section._set_enabled(False)

    # ---------------- Build config ---------------- #
    def _enabled_transforms(self):
        order = [
            ("rotation", self.sec_rotation),
            ("translation", self.sec_translation),
            ("elastic", self.sec_elastic),
            ("erasing", self.sec_erasing),
            ("scaling", self.sec_scaling),
            ("normalize", self.sec_normalize),
            ("horizontal_flip", self.sec_hflip),
            ("vertical_flip", self.sec_vflip),
        ]
        # If 3D, ensure elastic/erasing are not returned even if somehow toggled
        if (self._task or "").lower() == "semantic-3d":
            order = [(k, s) for (k, s) in order if k not in ("elastic", "erasing")]
        return [(k, sec) for (k, sec) in order if sec.is_enabled()]

    def get_config(self) -> dict:
        """
        Build a config dict matching the intended ImageAugmentor signature.
        Input paths come from state; Output paths from UI fields here.
        """
        transform_types = []
        params = {}

        for key, sec in self._enabled_transforms():
            transform_types.append(key)
            if key == "rotation":
                params[key] = {"degrees": self.rotation_degrees.get()}
            elif key == "translation":
                params[key] = {"shift": self.translation_shift.get()}
            elif key == "elastic":
                params[key] = {"alpha": float(self.elastic_alpha.value()), "sigma": float(self.elastic_sigma.value())}
            elif key == "erasing":
                params[key] = {
                    "scale": self.erasing_scale.get(),
                    "ratio": self.erasing_ratio.get(),
                    "p": float(self.erasing_p.value()),
                }
            elif key == "scaling":
                params[key] = {"scale": self.scaling_scale.get()}
            elif key == "normalize":
                params[key] = {
                    "mean": self.norm_mean.get(),
                    "std": self.norm_std.get(),
                    "max_pixel_value": float(self.norm_max.value()),
                    "p": float(self.norm_p.value()),
                }
            elif key == "horizontal_flip":
                params[key] = {"p": float(self.hflip_p.value())}
            elif key == "vertical_flip":
                params[key] = {"p": float(self.vflip_p.value())}

        cfg = {
            # directories (inputs from state)
            "input_img_dir": state.input_img_dir,
            "input_lbl_dir": state.input_lbl_dir,
            "output_img_dir": self.output_img_dir.text().strip(),
            "output_lbl_dir": self.output_lbl_dir.text().strip(),
            # transforms
            "transform_types": transform_types,
            "iterations": int(self.iterations.value()),
            "params": params,
            # meta
            "task": self._task,
        }
        return cfg

    def set_from_config(self, cfg: dict):
        """Optional: prefill UI from a config dict of the same shape."""
        # Inputs are driven by state; only prefill outputs/iters and transforms
        self.output_img_dir.setText(cfg.get("output_img_dir", ""))
        self.output_lbl_dir.setText(cfg.get("output_lbl_dir", ""))
        self.iterations.setValue(int(cfg.get("iterations", 2)))

        ttypes = set(cfg.get("transform_types", []))
        p = cfg.get("params", {})

        def toggle(sec, key):
            sec.chk.setChecked(key in ttypes)

        toggle(self.sec_rotation, "rotation")
        if "rotation" in p: self.rotation_degrees.set(tuple(p["rotation"].get("degrees", (-60, 60))))

        toggle(self.sec_translation, "translation")
        if "translation" in p: self.translation_shift.set(tuple(p["translation"].get("shift", (-0.1, 0.1))))

        toggle(self.sec_elastic, "elastic")
        if "elastic" in p:
            self.elastic_alpha.setValue(float(p["elastic"].get("alpha", 1.0)))
            self.elastic_sigma.setValue(float(p["elastic"].get("sigma", 50.0)))

        toggle(self.sec_erasing, "erasing")
        if "erasing" in p:
            self.erasing_scale.set(tuple(p["erasing"].get("scale", (0.02, 0.33))))
            self.erasing_ratio.set(tuple(p["erasing"].get("ratio", (0.3, 3.3))))
            self.erasing_p.setValue(float(p["erasing"].get("p", 1.0)))

        toggle(self.sec_scaling, "scaling")
        if "scaling" in p: self.scaling_scale.set(tuple(p["scaling"].get("scale", (0.8, 1.2))))

        toggle(self.sec_normalize, "normalize")
        if "normalize" in p:
            self.norm_mean.set(tuple(p["normalize"].get("mean", (0.0, 0.0, 0.0))))
            self.norm_std.set(tuple(p["normalize"].get("std", (1.0, 1.0, 1.0))))
            self.norm_max.setValue(float(p["normalize"].get("max_pixel_value", 255.0)))
            self.norm_p.setValue(float(p["normalize"].get("p", 1.0)))

        toggle(self.sec_hflip, "horizontal_flip")
        if "horizontal_flip" in p: self.hflip_p.setValue(float(p["horizontal_flip"].get("p", 1.0)))

        toggle(self.sec_vflip, "vertical_flip")
        if "vertical_flip" in p: self.vflip_p.setValue(float(p["vertical_flip"].get("p", 1.0)))

    # ---------------- Actions ---------------- #
    def _choose_dir(self, dest: DropLineEdit):
        start = dest.text().strip() or ""
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if dirname:
            dest.setText(dirname)

    # Requirement #4: stubs only
    def _on_run_random_one(self):
        cfg = self.get_config()
        print("[DataProcessingTab] running this button: Run on one Random Image")
        print("[DataProcessingTab] current config:", cfg)

    def _on_run_sample(self):
        cfg = self.get_config()
        print("[DataProcessingTab] running this button: Run on a Sample")
        print("[DataProcessingTab] current config:", cfg)

    # Requirement #2: Skip (go to Train without setting is_aug)
    def _on_skip(self):
        # do not touch state.is_aug
        # do not set aug_output_* here
        self._goto_training_tab()
        self.proceed_to_train.emit()

    # Requirement #5: Apply & Continue (set state + navigate)
    def _on_apply_and_continue(self):
        cfg = self.get_config()
        # Update global state
        state.is_aug = True
        state.aug_output_img_dir = cfg.get("output_img_dir", "")
        state.aug_output_lbl_dir = cfg.get("output_lbl_dir", "")
        # still emit full config for any listeners that want to run augmentation now
        self.requested_run.emit(cfg)
        # navigate to Training
        self._goto_training_tab()
        self.proceed_to_train.emit()

    # ------- Navigation helper: jump to "Training" tab in the parent QTabWidget ------- #
    def _goto_training_tab(self) -> bool:

        """
        Walk up the parent chain to find a QTabWidget (ModelBuilder tabs)
        and switch to the tab named 'Training' (case-insensitive).
        """
        w = self.parent()
        while w is not None:
            if isinstance(w, QtWidgets.QTabWidget):
                # Prefer finding by label
                for i in range(w.count()):
                    if w.tabText(i).strip().lower() == "training":
                        w.setCurrentIndex(i)
                        return True
                # Fallback to index 2 if label not found (Annotation, Data Processing, Training)
                if w.count() >= 3:
                    w.setCurrentIndex(2)
                    return True
                return False
            w = w.parent()
        return False
