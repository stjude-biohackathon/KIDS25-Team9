# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# AI-Image-Lab: Deep Learning Framework for Image Analysis

## Project Overview

AI-Image-Lab is a Napari plugin designed to provide a user-friendly interface for deep learning-based image analysis. It supports training, annotation, inference, and fine-tuning of models for various computer vision tasks.

**Key Purpose**: Enable researchers to train deep learning models for medical/biological image segmentation without requiring deep ML expertise.

**Project Context**: St. Jude Children's Research Hospital BioHackathon 2025 (MIT License)

## Project Structure

```
src/
├── ai_image_lab/          # Napari plugin entry point
│   ├── __init__.py        # Exports Lab widget
│   ├── _widget.py         # Main Lab container (orchestrates UI stages)
│   └── napari.yaml        # Napari plugin manifest
├── models/                # Model implementations (PyTorch)
│   ├── base_model.py      # ABC for all model architectures
│   ├── unet_2d.py         # 2D U-Net semantic segmentation
│   ├── unet_2d_run_script.py  # Standalone runner
│   └── mask_rcnn_model.py # Mask R-CNN for instance segmentation
├── ui/                    # Qt-based UI (qtpy abstraction layer)
│   ├── main_tab.py        # ProjectSetupTab: task & dataset selection
│   ├── annotation_tab.py  # AnnotationTab: model-assisted labeling
│   ├── data_processing_tab.py  # DataProcessingTab: augmentation
│   ├── train_tab.py       # TrainTab: dynamic training UI
│   ├── inference_tab.py   # InferenceTab: model inference runner
│   ├── fine_tune_sam_tab.py    # FineTuneSAMTab: SAM fine-tuning
│   ├── common.py          # Reusable components (DropLineEdit, Card, SelectableCard)
│   ├── styles.py          # QSS stylesheet + constants
│   ├── state.py           # AppState dataclass (shared mutable state)
│   └── all.py             # Utility functions
├── data_annotation/       # Annotation utilities
│   ├── data_annotator.py  # Checks for missing labels
│   └── mask_interpolator.py   # Interpolates masks across frames
├── data_augmentation/     # Data augmentation
│   └── augment_images.py  # ImageAugmentor (uses albumentations)
├── utilities/             # Dataset builders
│   └── maskrcnn_dataset_builder.py  # CellDataset, conversion utils
└── configs/               # Task-specific configurations
    ├── instance/config.json           # Mask R-CNN config template
    └── fine_tune_sam/config.json      # SAM fine-tune config template
```

## Technical Stack

### Core Dependencies
- **UI Framework**: Qt (via qtpy - abstracts PyQt5/PySide2)
- **Deep Learning**: PyTorch + TorchVision
- **Image Processing**: scikit-image, OpenCV, tifffile, PIL
- **Data Augmentation**: albumentations
- **Image Viewer Integration**: napari

### Build System
- **Package Manager**: setuptools + wheel
- **Configuration**: pyproject.toml (PEP 517/518 compliant)
- **Entry Point**: napari.manifest plugin system

### Python Version
- Python 3.8+

## Architecture & Key Patterns

### 1. UI Architecture: Multi-Stage Stacked Layout

The main Lab widget uses a QStackedLayout with 4 stages:

```
Stage 0: ProjectSetupTab     - Task selection + dataset paths
Stage 1: ModelBuilderWidget  - Tabbed UI (Annotation, Processing, Training)
Stage 2: InferenceTab        - Standalone inference
Stage 3: FineTuneSAMTab      - SAM fine-tuning
```

**Navigation Flow**:
1. User selects task + dataset in ProjectSetupTab, state.task is updated
2. Clicking "Continue" switches to Stage 1 (ModelBuilderWidget)
3. ModelBuilderWidget contains a TabWidget with 3 tabs (Annotation, Data Processing, Training)
4. Clicking "Back to Tasks" returns to Stage 0

**Key Signal Flow**:
- ProjectSetupTab.continued -> Lab switches stage
- ModelBuilderWidget.back_requested -> Lab switches stage
- InferenceTab.back_requested -> Lab switches stage

### 2. State Management

Uses a global mutable AppState dataclass in ui/state.py (singleton pattern):

```python
@dataclass
class AppState:
    task: str              # "semantic-2d" | "semantic-3d" | "instance" | "fine-tune"
    input_img_dir: str
    input_lbl_dir: str
    aug_output_img_dir: str
    aug_output_lbl_dir: str
    is_aug: bool
```

All tabs import and share the same state instance. This eliminates prop drilling and makes state accessible everywhere.

### 3. Model Architecture Pattern

All models inherit from BaseModel ABC in src/models/base_model.py:

```python
class BaseModel(ABC):
    @abstractmethod
    def architecture(self):
        pass
    
    @abstractmethod
    def train(self, data):
        pass
    
    @abstractmethod
    def infer(self, input_data):
        pass
```

**Current Implementations**:
- UNet: 2D semantic segmentation (encoder-decoder with skip connections)
- MaskRCNN: Instance segmentation (ResNet50-FPN backbone)

### 4. Dynamic Configuration System

The TrainTab builds UI dynamically from JSON configs:

```python
CONFIG_LOCATIONS = {
    "semantic-2d": "configs/semantic_2d/config.json",
    "semantic-3d": "configs/semantic_3d/config.json",
    "instance":    "configs/instance/config.json",
    "fine-tune":   "configs/fine_tune/config.json",
}
```

**How It Works**:
1. Load task-specific config.json
2. For each key-value pair:
   - If key contains "path"/"folder" -> create directory picker UI
   - If value is int -> create spinbox
   - If value is float -> create double spinbox
3. User edits values in the dynamic UI
4. On "Start Training" -> write modified JSON + call trainer class via importlib

**Trainer Resolution**:
```python
TRAINER_CLASSES = {
    "semantic-2d": "trainers.semantic2d.Semantic2DTrainer",
    "semantic-3d": "trainers.semantic3d.Semantic3DTrainer",
    "instance":    "models.mask_rcnn_model.maskrcnn_final",
    "fine-tune":   "trainers.finetune.FineTuneTrainer",
}
```

### 5. Dataset Building Pattern

MaskRCNNDataset builder converts semantic labels to instance masks:

```python
def semantic_to_instance_masks(mask):
    # Convert HxW semantic mask -> list of instance binary masks + class labels
    # - Find unique class IDs (skip 0 background)
    # - For each class: use connected components (ndimage.label)
    # - Each connected component = separate instance
    # - Return: instance_masks[], labels[]
```

This enables Mask R-CNN to learn instance-level segmentation from semantic labels.

### 6. UI Component Library

Reusable components in ui/common.py:

- **Card**: Styled QFrame wrapper for sections
- **SelectableCard**: Clickable card with hidden radio button for selection
- **DropLineEdit**: QLineEdit that accepts drag-drop folders
- **labeled_row()**: Helper function for label + widget + button rows

## How to Extend

### Adding a New Model

1. Create src/models/my_model.py:

```python
from models.base_model import BaseModel

class MyModel(BaseModel):
    def architecture(self):
        # Return PyTorch nn.Module
        return nn.Sequential(...)
    
    def train(self, train_loader, val_loader, **kwargs):
        # Training loop
        pass
    
    def infer(self, input_data):
        # Inference logic
        pass
```

2. Register in TrainTab TRAINER_CLASSES dict (if adding new task type)
3. Add config JSON in src/configs/{task}/config.json

### Adding a New UI Tab

1. Create src/ui/new_feature_tab.py:

```python
from qtpy.QtWidgets import QWidget, QVBoxLayout
from ui.styles import DEFAULT_CONTENT_MARGINS

class NewFeatureTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(*DEFAULT_CONTENT_MARGINS)
        # Add widgets
```

2. Import in ai_image_lab/_widget.py
3. Add to ModelBuilderWidget's tab widget
4. Use state if needed: `from ui.state import state`

### Adding Data Augmentation Transforms

The ImageAugmentor class in data_augmentation/augment_images.py uses albumentations. To add transforms, add to the transform_types list and implement a corresponding _get_<transform>_transform() method.

## Development Workflow

### Running the Application

```bash
# Install in development mode
pip install -e .

# Launch napari (auto-discovers the plugin)
napari

# Go to Plugins -> AI Image Lab to open the widget
```

### Development Commands

```bash
# Build package
python -m build

# Install from source
pip install -e .

# Check plugin registration
napari --info  # Should list ai-image-lab plugin

# Run standalone model training
python src/models/unet_2d_run_script.py
```

### Testing a Model Standalone

Each model has a runner script:

```bash
python src/models/unet_2d_run_script.py --config src/configs/semantic_2d/config.json
```

### Modifying Styles

All Qt stylesheets live in src/ui/styles.py as a STYLE_SHEET string. Constants for margins/spacing are also defined there:

```python
DEFAULT_CONTENT_MARGINS = (16, 16, 16, 16)  # left, top, right, bottom
DEFAULT_SPACING = 12
```

### Common Patterns

**Opening file/folder dialogs**:
```python
from qtpy.QtWidgets import QFileDialog
path = QFileDialog.getExistingDirectory(self, "Select folder...")
```

**Creating labeled rows**:
```python
from ui.common import labeled_row
lay.addLayout(labeled_row("Label text", widget, button))
```

**Accessing shared state**:
```python
from ui.state import state
state.task = "semantic-2d"
state.input_img_dir = "/path/to/images"
```

**Signal/slot connections**:
```python
self.some_button.clicked.connect(self.on_button_clicked)

def on_button_clicked(self):
    # handle event
    pass
```

## Configuration Files

### pyproject.toml
Defines package metadata, dependencies, and napari plugin entry point.

### napari.yaml
Plugin manifest. Registers the Lab widget as a napari command:
- **Command ID**: `ai-image-lab.enter_lab`
- **Widget Entry**: `ai_image_lab:Lab` (points to Lab class in `_widget.py`)

### Task Config JSONs
Templates in src/configs/{task}/config.json. Keys become UI parameter names:
- learning_rate, batch_size, num_epochs
- input_data_path, saved_model_path (directory fields get Browse buttons)

## Current Limitations

1. **Incomplete Trainers**: semantic2d, semantic3d, finetune trainers not fully implemented
2. **No Test Suite**: No tests exist (would need pytest + fixtures)
3. **Hardcoded Paths**: Config templates have absolute paths requiring updates
4. **SAM Integration**: Fine-tune SAM tab exists but incomplete
5. **Model Zoo**: Inference references Hugging Face/Bioimage IO but no integration
6. **Error Handling**: Limited user-facing error messages

## Key Files to Know

Core entry points and important modules:

- src/ai_image_lab/_widget.py: Main orchestration + stage switching
- src/ui/state.py: Global state management (singleton AppState)
- src/ui/styles.py: All Qt styling constants and stylesheet
- src/ui/main_tab.py: Task selection UI
- src/ui/train_tab.py: Dynamic training config UI
- src/ui/common.py: Reusable UI components (Card, DropLineEdit, etc)
- src/models/base_model.py: Model ABC base class
- src/models/mask_rcnn_model.py: Instance segmentation model
- src/models/unet_2d.py: 2D semantic segmentation model
- src/data_augmentation/augment_images.py: Data augmentation engine

## Code Style & Conventions

### Naming
- Classes: PascalCase (e.g., UNet, MaskRCNNModel, AnnotationTab)
- Functions/methods: snake_case (e.g., train_model(), _crop_and_concat())
- Private methods: prefix with _ (e.g., _build_ui())
- Constants: UPPER_SNAKE_CASE (e.g., DEFAULT_SPACING)

### UI Methods
- _build_ui(): Constructs layout hierarchy
- showEvent(): Refreshes UI when displayed (used for state sync)
- Signal naming: <action>_<object> or <past_tense> (e.g., continued, back_requested)

### Imports
- Qt imports via qtpy (allows PyQt5/PySide2 switching)
- Avoid circular imports between ui/ and models/

## Quick Debug Tips

1. **Check state**: Print state.__dict__ to see current app state
2. **UI not showing**: Verify _build_ui() is called from __init__
3. **Config not loading**: Check file path in CONFIG_LOCATIONS dict
4. **Import errors**: Ensure trainer classes exist in actual project structure
5. **Qt styling**: Examine STYLE_SHEET constant in styles.py

## Git Workflow

Main branch: main

Recent contributors (from commit history):
- Catherine (2D U-Net semantic segmentation)
- Krishnan (refinements)
- Chen Li (Mask R-CNN instance segmentation)
- Nishant (initial structure)

Branch pattern: Feature branches named after contributors/tasks

## Next Steps for New Contributors

1. **Understand the layout**: Open _widget.py and trace stage switching
2. **Study state management**: See how state is used in TrainTab
3. **Try modifying styles**: Edit styles.py to change colors/spacing
4. **Implement missing trainer**: Pick semantic2d or semantic3d trainer
5. **Add a test**: Create tests for data augmentation or model utilities
