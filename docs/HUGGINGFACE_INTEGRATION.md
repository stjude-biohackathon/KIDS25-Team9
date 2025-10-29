# Hugging Face Model Integration

This document describes the Hugging Face API integration for listing and selecting models in AI-Image-Lab.

## Overview

The Hugging Face integration allows users to:
- Browse models from Hugging Face Hub
- Search for specific models by name or description
- Filter models by task type (e.g., image-segmentation, object-detection)
- View model details including downloads, likes, and tags
- Select a model for inference

## Components

### 1. HuggingFaceClient (`src/utilities/huggingface_client.py`)

A client class for interacting with the Hugging Face API.

**Key Methods:**
- `list_models(search, task, limit, sort)` - List models with optional filters
- `search_models(query, limit)` - Search for models by query
- `get_model_info(model_id)` - Get detailed information about a specific model

**Example Usage:**
```python
from utilities.huggingface_client import HuggingFaceClient

client = HuggingFaceClient()

# List popular models
models = client.list_models(limit=20)

# Search for segmentation models
models = client.search_models("segmentation", limit=10)

# Filter by task
models = client.list_models(task="image-segmentation", limit=10)
```

### 2. HuggingFaceModelDialog (`src/ui/huggingface_model_dialog.py`)

A Qt dialog for browsing and selecting Hugging Face models.

**Features:**
- Search bar for filtering models
- Task filter dropdown (image segmentation, object detection, etc.)
- Model list with download and like counts
- Model details pane showing additional information
- Async loading of models in background thread

**Usage:**
```python
from ui.huggingface_model_dialog import HuggingFaceModelDialog

dialog = HuggingFaceModelDialog(parent_widget)
if dialog.exec_():
    model_id = dialog.get_selected_model_id()
    print(f"Selected model: {model_id}")
```

### 3. Inference Tab Integration (`src/ui/inference_tab.py`)

The inference tab has been updated to include a "Browse" button for Hugging Face models.

**Changes:**
- Added Browse button next to the model URL field
- Button is only visible when "Hugging Face" is selected as the model source
- Clicking Browse opens the HuggingFaceModelDialog
- Selected model ID is automatically populated in the URL field

## User Workflow

1. Open the Inference tab in AI-Image-Lab
2. Select "Hugging Face" from the "Model source" dropdown
3. Click the "Browse" button next to the model field
4. In the dialog:
   - Search for models using the search bar
   - Filter by task type using the dropdown
   - Select a model from the list
   - View model details in the right pane
   - Click "Select" to choose the model
5. The model ID is now populated in the inference tab
6. Continue with the inference workflow

## API Reference

The integration uses the Hugging Face Hub API:
- Base URL: `https://huggingface.co/api/models`
- Documentation: https://huggingface.co/docs/hub/api

## Network Requirements

This feature requires internet access to communicate with the Hugging Face Hub API.

## Future Enhancements

Potential improvements for future versions:
- Cache popular models locally
- Show model preview images
- Display model architecture information
- Allow filtering by license type
- Support for private models (with authentication)
- Direct model download functionality
