from dataclasses import dataclass

@dataclass
class AppState:
    task: str = ""                 # "semantic-2d" | "semantic-3d" | "instance" | "fine-tune" | "inference"
    input_img_dir: str = ""
    input_lbl_dir: str = ""
    aug_output_img_dir: str = ""
    aug_output_lbl_dir: str = ""

# A single shared instance you can import anywhere
state = AppState()