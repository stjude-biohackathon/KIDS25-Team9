# FILE: ui/train_tab.py

import json
import os
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QHBoxLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFormLayout, QLineEdit, QProgressBar,
    QFileDialog, QSizePolicy, QMessageBox
)

from ui.common import Card, DropLineEdit, labeled_row
from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ui.state import state
import importlib


class TrainTab(QWidget):
    """
    Dynamic training UI driven by task-specific config.json.

    - Reads task from ui.state
    - Loads config.json for that task and builds parameter editors dynamically
    - Keys containing 'path' or 'folder' -> directory field with Browse…
    - If task == 'fine-tune' -> shows 'Base model path' file picker (added to config before training)
    - Separate card: logs folder picker + progress bar + public receiver `on_progress(int)`
    - Start Training -> writes new JSON from current UI and calls the appropriate TrainerClass.train(config_path)
    """
    start_training = QtCore.Signal(dict)  # emits final config dict (optional for your listeners)

    # ---- Customize these to your project structure ---- #
    CONFIG_LOCATIONS = {
        "semantic-2d": "configs/semantic_2d/config.json",
        "semantic-3d": "configs/semantic_3d/config.json",
        "instance":    "configs/instance/config.json",
        "fine-tune":   "configs/fine_tune/config.json",
    }
    TRAINER_CLASSES = {
        "semantic-2d": "trainers.semantic2d.Semantic2DTrainer",
        "semantic-3d": "trainers.semantic3d.Semantic3DTrainer",
        "instance":    "models.mask_rcnn_model.maskrcnn_final",
        "fine-tune":   "trainers.finetune.FineTuneTrainer",
    }
    # --------------------------------------------------- #

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._task_used = None
        self._config = {}
        self._schema = {}      # dotted_key -> {"type":..., "elem_type":...}
        self._editors = {}     # dotted_key -> QWidget editor
        self._dir_fields = set()   # dotted_keys that are directory fields
        self._base_model_edit = None  # only for fine-tune

        # logs and progress
        self._logs_dir_edit = None
        self._progress = None

        self._build_ui()
        # initial populate
        self._ensure_config_loaded()

    # refresh when shown (handles "Back to Tasks" -> change task -> return)
    def showEvent(self, e: QtGui.QShowEvent) -> None:
        super().showEvent(e)
        self._ensure_config_loaded()

    # ------------------ UI ------------------ #
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        # Outer scroll for the tab
        outer_scroll = QScrollArea()
        outer_scroll.setWidgetResizable(True)
        outer_scroll.setFrameShape(QFrame.NoFrame)
        outer_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        body = QWidget()
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2, 2, 2, 2)
        body_lay.setSpacing(DEFAULT_SPACING)

        # --- (optional) Fine-tune row: Base model path --- #
        self.ft_card = Card()
        self.ft_card.setVisible(False)  # only visible for fine-tune
        ft_lay = self.ft_card.layout()
        ft_title = QLabel("Fine-tune"); ft_title.setObjectName("H2"); ft_lay.addWidget(ft_title)
        self._base_model_edit = QLineEdit(); self._base_model_edit.setReadOnly(True)
        btn_browse_base = QPushButton("Browse"); btn_browse_base.setObjectName("SecondaryBtn")
        row = QHBoxLayout()
        row.addWidget(QLabel("Base model path")); row.addWidget(self._base_model_edit, 1); row.addWidget(btn_browse_base)
        ft_lay.addLayout(row)
        body_lay.addWidget(self.ft_card)
        btn_browse_base.clicked.connect(self._browse_base_model_file)

        # --- Parameters card (scrollable) --- #
        self.param_card = Card()
        pl = self.param_card.layout()
        title = QLabel("Training Parameters"); title.setObjectName("H2")
        pl.addWidget(title)

        # inner scroll inside the card to handle large configs
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setFrameShape(QFrame.NoFrame)
        self.param_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.param_body = QWidget()
        self.param_form = QFormLayout(self.param_body)
        self.param_form.setContentsMargins(2, 2, 2, 2)
        self.param_form.setSpacing(8)

        self.param_scroll.setWidget(self.param_body)
        pl.addWidget(self.param_scroll, 1)
        body_lay.addWidget(self.param_card, 1)

        # --- Progress + Logs card --- #
        self.run_card = Card()
        rl = self.run_card.layout()
        rtitle = QLabel("Run"); rtitle.setObjectName("H2"); rl.addWidget(rtitle)

        # logs folder
        self._logs_dir_edit = DropLineEdit("Drop or browse a folder for training logs…")
        btn_logs = QPushButton("Browse"); btn_logs.setObjectName("SecondaryBtn")
        rl.addLayout(labeled_row("Logs folder", self._logs_dir_edit, btn_logs))
        btn_logs.clicked.connect(lambda: self._browse_dir(self._logs_dir_edit))

        # progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        rl.addWidget(self._progress)

        body_lay.addWidget(self.run_card)

        # --- Actions --- #
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_start = QPushButton("Start Training")
        self.btn_start.setObjectName("CTA")
        actions.addWidget(self.btn_start)
        body_lay.addLayout(actions)

        outer_scroll.setWidget(body)
        root.addWidget(outer_scroll, 1)

        # Events
        self.btn_start.clicked.connect(self._on_start_training)

    # ------------------ Config loading / UI building ------------------ #
    def _ensure_config_loaded(self):
        task_now = (state.task or "").strip().lower()
        if not task_now:
            return
        if task_now == self._task_used and self._config:
            return

        # Show/Hide fine-tune card
        self.ft_card.setVisible(task_now == "fine-tune")

        cfg_path = self._resolve_config_path(task_now)
        if not cfg_path:
            QMessageBox.warning(self, "Config missing",
                                f"Could not locate a config.json for task '{task_now}'.\n"
                                "Please update TrainTab.CONFIG_LOCATIONS or place a config file.")
            return

        try:
            with open(cfg_path, "r") as f:
                self._config = json.load(f)
            self._task_used = task_now
        except Exception as e:
            QMessageBox.critical(self, "Config error", f"Failed to read config:\n{cfg_path}\n\n{type(e).__name__}: {e}")
            return

        # (Re)build parameter UI
        self._schema.clear()
        self._editors.clear()
        self._dir_fields.clear()
        # clear form
        while self.param_form.count():
            item = self.param_form.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        # Build rows from config (flattened dotted keys)
        self._build_param_rows(self._config)

    def _resolve_config_path(self, task: str) -> str:
        # 1) explicit mapping
        if task in self.CONFIG_LOCATIONS:
            path = self.CONFIG_LOCATIONS[task]
            script_dir = os.path.dirname(__file__)
            path = os.path.join(
                script_dir, "..", path
            )
            if os.path.isfile(path):
                return path
        # 2) common fallback
        guess = os.path.join("configs", task.replace("-", "_"), "config.json")
        if os.path.isfile(guess):
            return guess
        # 3) last resort: ask user once
        dlg = QFileDialog(self, "Select config.json for task: " + task)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("JSON files (*.json)")
        if dlg.exec_():
            files = dlg.selectedFiles()
            return files[0] if files else ""
        return ""

    def _build_param_rows(self, cfg: dict, prefix: str = ""):
        """
        Recursively build a dotted-key form from a nested dict config.
        """
        for key, val in cfg.items():
            dkey = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                # recurse into dict
                self._build_param_rows(val, dkey)
            else:
                editor = self._editor_for_value(dkey, val)
                label = QLabel(dkey)
                label.setObjectName("FieldLabel")
                if isinstance(editor, tuple):
                    # (containerWidget, browse_button) for folder/path keys
                    self.param_form.addRow(label, editor[0])
                else:
                    self.param_form.addRow(label, editor)

    def _editor_for_value(self, dotted_key: str, value):
        """
        Create an editor for a leaf value and record its schema.
        Keys containing 'path' or 'folder' -> directory field with Browse.
        """
        lname = dotted_key.lower()
        is_dir = ("path" in lname) or ("folder" in lname)
        self._schema[dotted_key] = self._infer_schema(value)

        if is_dir:
            field = DropLineEdit("Drop or browse a folder…")
            btn = QPushButton("Browse"); btn.setObjectName("SecondaryBtn")
            # container to align field + button
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0); row.setSpacing(6)
            cont = QWidget(); cont.setLayout(row)
            row.addWidget(field, 1); row.addWidget(btn)
            btn.clicked.connect(lambda _, dest=field: self._browse_dir(dest))
            self._editors[dotted_key] = field
            self._dir_fields.add(dotted_key)
            # set initial value if it's a string
            if isinstance(value, str):
                field.setText(value)
            return cont  # container widget for the form row

        # non-directory editors
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(bool(value))
        elif isinstance(value, int):
            w = QSpinBox()
            w.setRange(-1_000_000_000, 1_000_000_000)
            w.setValue(int(value))
        elif isinstance(value, float):
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(-1e12, 1e12)
            w.setSingleStep(0.1)
            w.setValue(float(value))
        elif isinstance(value, (list, tuple)):
            # simple CSV editor; preserve element type from first item
            w = QLineEdit()
            w.setPlaceholderText("comma-separated list")
            try:
                w.setText(",".join(str(x) for x in value))
            except Exception:
                w.setText("")
        else:
            # string or unknown -> plain line edit
            w = QLineEdit()
            if value is not None:
                w.setText(str(value))

        self._editors[dotted_key] = w
        return w

    def _infer_schema(self, value):
        """
        Store minimal type info so we can round-trip from editor -> JSON.
        """
        if isinstance(value, list):
            elem_type = type(value[0]) if value else str
            return {"type": list, "elem_type": elem_type}
        return {"type": type(value)}

    # ------------------ Collect / Write config ------------------ #
    def _collect_values(self) -> dict:
        """
        Rebuild a dict with same structure as self._config from editor values.
        """
        # start from a deep copy of original to preserve structure
        import copy
        out = copy.deepcopy(self._config)

        for dkey, editor in self._editors.items():
            schema = self._schema.get(dkey, {"type": str})
            val = self._value_from_editor(editor, schema)
            self._assign_by_dotted_key(out, dkey, val)

        # Fine-tune: include base model path when present
        if (state.task or "").strip().lower() == "fine-tune":
            if self._base_model_edit and self._base_model_edit.text().strip():
                out["base_model_path"] = self._base_model_edit.text().strip()

        return out

    def _value_from_editor(self, editor, schema: dict):
        # directory fields are DropLineEdit
        if isinstance(editor, DropLineEdit):
            return editor.text().strip()

        # editor might be inside a container when dir field; handle that
        if isinstance(editor, QWidget) and not isinstance(editor, (QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, DropLineEdit)):
            # find first child editor
            line = editor.findChild(QLineEdit)
            if line is not None:
                return line.text().strip()
            return ""

        typ = schema.get("type", str)
        if typ is bool:
            return bool(editor.isChecked())
        if typ is int:
            return int(editor.value())
        if typ is float:
            return float(editor.value())
        if typ is list:
            elem_type = schema.get("elem_type", str)
            raw = editor.text().strip()
            if not raw:
                return []
            parts = [p.strip() for p in raw.split(",")]
            casted = []
            for p in parts:
                try:
                    if elem_type is int:
                        casted.append(int(p))
                    elif elem_type is float:
                        casted.append(float(p))
                    elif elem_type is bool:
                        casted.append(p.lower() in ("1", "true", "yes", "on"))
                    else:
                        casted.append(p)
                except Exception:
                    casted.append(p)
            return casted
        # default: string
        return editor.text().strip()

    def _assign_by_dotted_key(self, d: dict, dotted: str, value):
        parts = dotted.split(".")
        cur = d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def _write_config_json(self, cfg: dict) -> str:
        # Pick logs folder if set, else ask, else fallback to CWD
        dest_dir = self._logs_dir_edit.text().strip()
        if not dest_dir:
            dlg = QFileDialog(self, "Choose folder for training logs")
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly, True)
            if dlg.exec_():
                chosen = dlg.selectedFiles()
                if chosen:
                    dest_dir = chosen[0]
                    self._logs_dir_edit.setText(dest_dir)
        if not dest_dir:
            dest_dir = os.getcwd()
        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, "training_config.json")
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        return out_path

    # ------------------ Browse helpers ------------------ #
    def _browse_dir(self, dest_edit: DropLineEdit):
        start = dest_edit.text().strip() or (state.input_img_dir or os.path.expanduser("~"))
        dirname = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if dirname:
            dest_edit.setText(dirname)

    def _browse_base_model_file(self):
        start = self._base_model_edit.text().strip() or os.path.expanduser("~")
        fname, _ = QFileDialog.getOpenFileName(self, "Select Base Model", start)
        if fname:
            self._base_model_edit.setText(fname)

    # ------------------ Trainer resolve / progress ------------------ #
    def _resolve_trainer_class(self, task: str):
        """
        Import and return the trainer class for the task.
        TRAINER_CLASSES should map task -> "module.ClassName"
        """
        path = self.TRAINER_CLASSES.get(task)
        if not path:
            return None
        try:
            mod_name, cls_name = path.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name, None)
        except Exception:
            return None

    @QtCore.Slot(int)
    def on_progress(self, value: int):
        """Public receiver to update the progress bar from external signals."""
        self._progress.setValue(int(value))

    # ------------------ Start training ------------------ #
    def _on_start_training(self):
        task = (state.task or "").strip().lower()
        if not task:
            QMessageBox.warning(self, "No task", "Task is not set in state.")
            return

        # collect UI -> dict
        cfg = self._collect_values()

        # write to logs folder as training_config.json
        try:
            cfg_path = self._write_config_json(cfg)
        except Exception as e:
            QMessageBox.critical(self, "Write failed", f"Could not write config JSON:\n{type(e).__name__}: {e}")
            return

        # resolve trainer class & run
        TrainerClass = self._resolve_trainer_class(task)
        if TrainerClass is None:
            QMessageBox.critical(self, "Trainer not found",
                                 "Could not resolve trainer class for task '{}'.\n"
                                 "Please set TrainTab.TRAINER_CLASSES.".format(task))
            return

        try:
            trainer = TrainerClass()
            # If trainer exposes a Qt signal for progress (e.g., `progress = Signal(int)`), connect it:
            if hasattr(trainer, "progress"):
                try:
                    trainer.progress.connect(self.on_progress)
                except Exception:
                    pass

            # Call train(config_path) — adjust if your signature differs
            trainer.train(cfg_path)

            # optional: emit final config for listeners
            self.start_training.emit(cfg)
        except Exception as e:
            QMessageBox.critical(self, "Training failed", f"{type(e).__name__}: {e}")
