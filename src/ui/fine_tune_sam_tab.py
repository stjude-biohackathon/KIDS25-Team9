import json
import os
import importlib

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QHBoxLayout,
    QFormLayout, QLineEdit, QProgressBar, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox,
    QSizePolicy, QMessageBox
)

from ui.styles import DEFAULT_CONTENT_MARGINS, DEFAULT_SPACING
from ui.common import Card, DropLineEdit, labeled_row


class FineTuneSAMTab(QWidget):
    """
    Dedicated tab for SAM fine-tuning.

    - Base model path (file picker)
    - Dynamic parameter UI (from config.json)
    - Logs folder + progress bar
    - Start Fine-Tuning -> writes a JSON and calls SAMFineTuner.fine_tune(config_path)
    """
    started = QtCore.Signal(dict)  # optional: emits final config dict

    # ---------- Customize to your project ----------
    script_dir = os.path.dirname(__file__)
    path = os.path.join(
        script_dir, "..", "configs/fine_tune_sam/config.json"
    )
    CONFIG_PATH = path
    TRAINER_CLASS = "trainers.sam_finetune.SAMFineTuner"  # module.ClassName
    # ------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._config = {}
        self._schema = {}
        self._editors = {}
        self._dir_fields = set()

        self._build_ui()
        self._load_config()

    # ---------------- UI ---------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        root.setSpacing(DEFAULT_SPACING)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        body = QWidget()
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2, 2, 2, 2)
        body_lay.setSpacing(DEFAULT_SPACING)

        # --- Base model path (file) ---
        base_card = Card(); bl = base_card.layout()
        btitle = QLabel("Base Model"); btitle.setObjectName("H2"); bl.addWidget(btitle)
        self.base_model_edit = QLineEdit(); self.base_model_edit.setReadOnly(True)
        btn_browse_base = QPushButton("Browse"); btn_browse_base.setObjectName("SecondaryBtn")
        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(QLabel("Base model path")); row.addWidget(self.base_model_edit, 1); row.addWidget(btn_browse_base)
        bl.addLayout(row)
        btn_browse_base.clicked.connect(self._browse_base_model_file)
        body_lay.addWidget(base_card)

        # --- Parameters (scrollable card) ---
        self.param_card = Card(); pl = self.param_card.layout()
        ptitle = QLabel("Fine-tuning Parameters"); ptitle.setObjectName("H2")
        pl.addWidget(ptitle)

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

        # --- Run card: logs + progress ---
        run_card = Card(); rl = run_card.layout()
        rtitle = QLabel("Run"); rtitle.setObjectName("H2"); rl.addWidget(rtitle)

        self.logs_dir_edit = DropLineEdit("Drop or browse a folder for training logs…")
        btn_logs = QPushButton("Browse"); btn_logs.setObjectName("SecondaryBtn")
        rl.addLayout(labeled_row("Logs folder", self.logs_dir_edit, btn_logs))
        btn_logs.clicked.connect(lambda: self._browse_dir(self.logs_dir_edit))

        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        rl.addWidget(self.progress)

        body_lay.addWidget(run_card)

        # --- Actions ---
        actions = QHBoxLayout(); actions.setSpacing(8); actions.addStretch(1)
        self.btn_start = QPushButton("Start Fine-Tuning"); self.btn_start.setObjectName("CTA")
        actions.addWidget(self.btn_start)
        body_lay.addLayout(actions)

        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # Events
        self.btn_start.clicked.connect(self._on_start)

    # ---------------- Config load / UI ---------------- #
    def _load_config(self):
        path = self.CONFIG_PATH
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Config missing",
                                f"Could not find config.json for SAM fine-tuning at:\n{path}")
            return
        try:
            import json
            with open(path, "r") as f:
                self._config = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Config error", f"Failed to read config:\n{type(e).__name__}: {e}")
            return

        # build editable rows
        self._schema.clear(); self._editors.clear(); self._dir_fields.clear()
        while self.param_form.count():
            item = self.param_form.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self._build_param_rows(self._config)

    def _build_param_rows(self, cfg: dict, prefix: str = ""):
        for key, val in cfg.items():
            dkey = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                self._build_param_rows(val, dkey)
            else:
                editor = self._editor_for_value(dkey, val)
                label = QLabel(dkey); label.setObjectName("FieldLabel")
                if isinstance(editor, QWidget):
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

        # non-directory: choose widget by type
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
            w = QLineEdit();
            if value is not None:
                w.setText(str(value))
        self._editors[dotted_key] = w
        return w

    def _infer_schema(self, value):
        if isinstance(value, list):
            elem_type = type(value[0]) if value else str
            return {"type": list, "elem_type": elem_type}
        return {"type": type(value)}

    # ---------------- Collect / Write config ---------------- #
    def _collect_values(self) -> dict:
        import copy
        out = copy.deepcopy(self._config)
        for dkey, editor in self._editors.items():
            schema = self._schema.get(dkey, {"type": str})
            val = self._value_from_editor(editor, schema)
            self._assign_by_dotted_key(out, dkey, val)

        # include base_model_path (file) at top level if set
        bmp = self.base_model_edit.text().strip()
        if bmp:
            out["base_model_path"] = bmp
        return out

    def _value_from_editor(self, editor, schema: dict):
        if isinstance(editor, DropLineEdit):
            return editor.text().strip()
        if isinstance(editor, QCheckBox):
            return bool(editor.isChecked())
        if isinstance(editor, QSpinBox):
            return int(editor.value())
        if isinstance(editor, QDoubleSpinBox):
            return float(editor.value())
        if isinstance(editor, QLineEdit):
            typ = schema.get("type", str)
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
                            casted.append(p.lower() in ("1","true","yes","on"))
                        else:
                            casted.append(p)
                    except Exception:
                        casted.append(p)
                return casted
            return editor.text().strip()
        return str(editor.text()).strip() if hasattr(editor, "text") else None

    def _assign_by_dotted_key(self, d: dict, dotted: str, value):
        parts = dotted.split(".")
        cur = d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def _write_config_json(self, cfg: dict) -> str:
        dest_dir = self.logs_dir_edit.text().strip()
        if not dest_dir:
            dlg = QFileDialog(self, "Choose folder for training logs")
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly, True)
            if dlg.exec_():
                chosen = dlg.selectedFiles()
                if chosen:
                    dest_dir = chosen[0]
                    self.logs_dir_edit.setText(dest_dir)
        if not dest_dir:
            dest_dir = os.getcwd()
        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, "sam_finetune_config.json")
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        return out_path

    # ---------------- Browse helpers ---------------- #
    def _browse_dir(self, dest_edit: DropLineEdit):
        start = dest_edit.text().strip() or os.path.expanduser("~")
        dirname = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if dirname:
            dest_edit.setText(dirname)

    def _browse_base_model_file(self):
        start = self.base_model_edit.text().strip() or os.path.expanduser("~")
        fname, _ = QFileDialog.getOpenFileName(self, "Select Base Model", start)
        if fname:
            self.base_model_edit.setText(fname)

    # ---------------- Progress receiver ---------------- #
    @QtCore.Slot(int)
    def on_progress(self, value: int):
        self.progress.setValue(int(value))

    # ---------------- Start fine-tune ---------------- #
    def _on_start(self):
        cfg = self._collect_values()
        try:
            cfg_path = self._write_config_json(cfg)
        except Exception as e:
            QMessageBox.critical(self, "Write failed", f"Could not write config JSON:\n{type(e).__name__}: {e}")
            return

        TrainerClass = self._resolve_trainer_class()
        if TrainerClass is None:
            QMessageBox.critical(self, "Trainer not found",
                                 "Could not resolve SAM fine-tune trainer class.\n"
                                 "Please set FineTuneSAMTab.TRAINER_CLASS.")
            return

        try:
            trainer = TrainerClass()
            if hasattr(trainer, "progress"):
                try:
                    trainer.progress.connect(self.on_progress)
                except Exception:
                    pass

            # Prefer 'fine_tune', fall back to 'train'
            if hasattr(trainer, "fine_tune"):
                trainer.fine_tune(cfg_path)
            else:
                trainer.train(cfg_path)

            self.started.emit(cfg)
        except Exception as e:
            QMessageBox.critical(self, "Fine-tuning failed", f"{type(e).__name__}: {e}")

    def _resolve_trainer_class(self):
        try:
            mod_name, cls_name = self.TRAINER_CLASS.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name, None)
        except Exception:
            return None

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
