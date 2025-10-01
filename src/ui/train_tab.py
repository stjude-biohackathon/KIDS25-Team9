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

# == UI imports ==
from ui.common import Card, DropLineEdit, labeled_row
from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ui.state import state

# == Trainer class imports (explicit) ==
# Adjust these imports to your actual project structure.
from models.mask_rcnn_model import maskrcnn_final                  # instance
#from trainers.semantic2d import Semantic2DTrainer                 # semantic-2d
#from trainers.semantic3d import Semantic3DTrainer                 # semantic-3d
#from trainers.finetune import FineTuneTrainer                     # fine-tune


class TrainTab(QWidget):
    """
    Dynamic training UI driven by task-specific config.json.

    - Reads task from ui.state
    - Loads config.json for that task and builds parameter editors dynamically
    - Keys containing 'path' or 'folder' -> directory field with Browse…
    - If task == 'fine-tune' -> shows 'Base model path' file picker (added to config before training)
    - Separate card: logs folder picker + progress bar + public receiver `on_progress(int)`
    - Start Training -> writes new JSON from the UI and calls the appropriate trainer using if/elif
    """
    start_training = QtCore.Signal(dict)  # emits final config dict (optional)

    # ---- Config file locations (adjust paths as needed) ---- #
    CONFIG_LOCATIONS = {
        "semantic-2d": "configs/unet_2d/unet_2d_config.json",
        "semantic-3d": "configs/semantic_3d/config.json",
        "instance":    "configs/instance/maskrcnn_config_train.json",
        "fine-tune":   "configs/fine_tune/config.json",
    }
    # -------------------------------------------------------- #

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._task_used = None
        self._config = {}
        self._schema = {}          # dotted_key -> {"type":..., "elem_type":...}
        self._editors = {}         # dotted_key -> QWidget editor
        self._dir_fields = set()   # dotted_keys that are directory fields
        self._base_model_edit = None  # only for fine-tune

        # logs and progress
        self._logs_dir_edit = None
        self._progress = None

        self._build_ui()
        self._ensure_config_loaded()

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

        self._logs_dir_edit = DropLineEdit("Drop or browse a folder for training logs…")
        btn_logs = QPushButton("Browse"); btn_logs.setObjectName("SecondaryBtn")
        rl.addLayout(labeled_row("Logs folder", self._logs_dir_edit, btn_logs))
        btn_logs.clicked.connect(lambda: self._browse_dir(self._logs_dir_edit))

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

        self.ft_card.setVisible(task_now == "fine-tune")

        cfg_path = self._resolve_config_path(task_now)
        if not cfg_path:
            QMessageBox.warning(
                self, "Config missing",
                f"Could not locate a config.json for task '{task_now}'.\n"
                "Please update TrainTab.CONFIG_LOCATIONS or place a config file."
            )
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
        while self.param_form.count():
            item = self.param_form.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        self._build_param_rows(self._config)

    def _resolve_config_path(self, task: str) -> str:
        # 1) explicit mapping
        if task in self.CONFIG_LOCATIONS:
            path = self.CONFIG_LOCATIONS[task]
            script_dir = os.path.dirname(__file__)
            path = os.path.join(script_dir, "..", path)
            if os.path.isfile(path):
                return path
        # 2) fallback guess
        guess = os.path.join("configs", task.replace("-", "_"), "config.json")
        if os.path.isfile(guess):
            return guess
        # 3) ask user
        dlg = QFileDialog(self, "Select config.json for task: " + task)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("JSON files (*.json)")
        if dlg.exec_():
            files = dlg.selectedFiles()
            return files[0] if files else ""
        return ""

    def _build_param_rows(self, cfg: dict, prefix: str = ""):
        for key, val in cfg.items():
            dkey = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                self._build_param_rows(val, dkey)
            else:
                editor = self._editor_for_value(dkey, val)
                label = QLabel(dkey); label.setObjectName("FieldLabel")
                if isinstance(editor, tuple):
                    self.param_form.addRow(label, editor[0])
                else:
                    self.param_form.addRow(label, editor)

    def _editor_for_value(self, dotted_key: str, value):
        lname = dotted_key.lower()
        is_dir = ("path" in lname) or ("folder" in lname)
        self._schema[dotted_key] = self._infer_schema(value)

        if is_dir:
            field = DropLineEdit("Drop or browse a folder…")
            btn = QPushButton("Browse"); btn.setObjectName("SecondaryBtn")
            row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(6)
            cont = QWidget(); cont.setLayout(row)
            row.addWidget(field, 1); row.addWidget(btn)
            btn.clicked.connect(lambda _, dest=field: self._browse_dir(dest))
            self._editors[dotted_key] = field
            self._dir_fields.add(dotted_key)
            if isinstance(value, str):
                field.setText(value)
            return cont

        if isinstance(value, bool):
            w = QCheckBox(); w.setChecked(bool(value))
        elif isinstance(value, int):
            w = QSpinBox(); w.setRange(-1_000_000_000, 1_000_000_000); w.setValue(int(value))
        elif isinstance(value, float):
            w = QDoubleSpinBox(); w.setDecimals(6); w.setRange(-1e12, 1e12); w.setSingleStep(0.1); w.setValue(float(value))
        elif isinstance(value, (list, tuple)):
            w = QLineEdit(); w.setPlaceholderText("comma-separated list")
            try:
                w.setText(",".join(str(x) for x in value))
            except Exception:
                w.setText("")
        else:
            w = QLineEdit()
            if value is not None:
                w.setText(str(value))

        self._editors[dotted_key] = w
        return w

    def _infer_schema(self, value):
        if isinstance(value, list):
            elem_type = type(value[0]) if value else str
            return {"type": list, "elem_type": elem_type}
        return {"type": type(value)}

    # ------------------ Collect / Write config ------------------ #
    def _collect_values(self) -> dict:
        import copy
        out = copy.deepcopy(self._config)

        for dkey, editor in self._editors.items():
            schema = self._schema.get(dkey, {"type": str})
            val = self._value_from_editor(editor, schema)
            self._assign_by_dotted_key(out, dkey, val)

        if (state.task or "").strip().lower() == "fine-tune":
            if self._base_model_edit and self._base_model_edit.text().strip():
                out["base_model_path"] = self._base_model_edit.text().strip()

        return out

    def _value_from_editor(self, editor, schema: dict):
        if isinstance(editor, DropLineEdit):
            return editor.text().strip()

        if isinstance(editor, QWidget) and not isinstance(
            editor, (QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, DropLineEdit)
        ):
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
        return editor.text().strip()

    def _assign_by_dotted_key(self, d: dict, dotted: str, value):
        parts = dotted.split(".")
        cur = d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def _write_config_json(self, cfg: dict) -> str:
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

    def run_dummy_progress(self, total_ms: int = 8000, steps: int = 100, finish_message: str | None = None):
        """
        Smoothly fills the progress bar to 100% in equal intervals (non-blocking).
        total_ms: total duration in milliseconds (default ~8s)
        steps:    number of ticks (default 100 -> 1% per tick)
        """
        # Lazy-create a timer we reuse across runs
        if not hasattr(self, "_dummy_timer"):
            self._dummy_timer = QtCore.QTimer(self)
            self._dummy_timer.timeout.connect(self._on__dummy_tick)

        # Reset state
        self._dummy_total_steps = max(1, int(steps))
        self._dummy_i = 0
        self._dummy_finish_message = finish_message

        # Configure timer interval and start
        interval = max(10, int(total_ms) // self._dummy_total_steps)
        self._dummy_timer.stop()
        self._dummy_timer.setInterval(interval)
        self._progress.setValue(0)
        self._dummy_timer.start()

    def _on__dummy_tick(self):
        self._dummy_i += 1
        if self._dummy_i >= getattr(self, "_dummy_total_steps", 100):
            self._progress.setValue(100)
            # stop first to avoid re-entry
            self._dummy_timer.stop()
            msg = getattr(self, "_dummy_finish_message", None)
            if msg:
                QtWidgets.QMessageBox.information(self, "Done", msg)
        else:
            pct = int(self._dummy_i * 100 / self._dummy_total_steps)
            self._progress.setValue(pct)

    def cancel_dummy_progress(self):
        """Optional: stop the dummy progress early."""
        t = getattr(self, "_dummy_timer", None)
        if t and t.isActive():
            t.stop()

    # ------------------ Progress receiver ------------------ #
    @QtCore.Slot(int)
    def on_progress(self, value: int):
        self._progress.setValue(int(value))

    # ------------------ Start training ------------------ #
    def _on_start_training(self):
        task = (state.task or "").strip().lower()
        if not task:
            QMessageBox.warning(self, "No task", "Task is not set in state.")
            return

        cfg = self._collect_values()

        try:
            cfg_path = self._write_config_json(cfg)
        except Exception as e:
            QMessageBox.critical(self, "Write failed", f"Could not write config JSON:\n{type(e).__name__}: {e}")
            return

        self.run_dummy_progress()

        """
        # ----- Choose trainer via explicit if/elif imports -----
        trainer = None
        if task == "instance":
            # Mask R-CNN trainer (imported explicitly)
            print("Training preface?")
            trainer = maskrcnn_final(cfg_path)
        elif task == "semantic-2d":
            trainer = Semantic2DTrainer(cfg_path)
        elif task == "semantic-3d":
            trainer = Semantic3DTrainer(cfg_path)
        elif task == "fine-tune":
            trainer = FineTuneTrainer(cfg_path)
        else:
            QMessageBox.critical(self, "Trainer not found", f"No trainer wired for task '{task}'.")
            return

        trainer.progress.connect(self.on_progress)
        print("here")
        trainer.train()
        
        # Connect progress if available
        if hasattr(trainer, "progress"):
            try:
                print("yes or now")
                trainer.progress.connect(self.on_progress)
            except Exception:
                pass

        # Start training (adjust if your signature differs)
        if hasattr(trainer, "train"):
            print("Training started ?")
            trainer.train()
        else:
            QMessageBox.critical(self, "Trainer error", f"Trainer for '{task}' has no .train() method.")
            return


        self.start_training.emit(cfg)
        try:
            if task == "instance":
                # Mask R-CNN trainer (imported explicitly)
                print("Training preface?")
                trainer = maskrcnn_final(cfg_path)
            elif task == "semantic-2d":
                trainer = Semantic2DTrainer(cfg_path)
            elif task == "semantic-3d":
                trainer = Semantic3DTrainer(cfg_path)
            elif task == "fine-tune":
                trainer = FineTuneTrainer(cfg_path)
            else:
                QMessageBox.critical(self, "Trainer not found", f"No trainer wired for task '{task}'.")
                return

            trainer.progress.connect(self.on_progress)
            print("here")
            trainer.train()

            # Connect progress if available
            if hasattr(trainer, "progress"):
                try:
                    print("yes or now")
                    trainer.progress.connect(self.on_progress)
                except Exception:
                    pass

            # Start training (adjust if your signature differs)
            if hasattr(trainer, "train"):
                print("Training started ?")
                trainer.train()
            else:
                QMessageBox.critical(self, "Trainer error", f"Trainer for '{task}' has no .train() method.")
                return


            self.start_training.emit(cfg)
            

        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Training failed", f"{type(e).__name__}: {e}")
        """
