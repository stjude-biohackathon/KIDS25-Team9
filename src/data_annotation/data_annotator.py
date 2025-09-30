import os

class DataAnnotator:
    def __init__(self, input_folder: str, file_ext: str = ".tif"):
        self.images_folder = os.path.join(input_folder, "Images")
        self.labels_folder = os.path.join(input_folder, "Labels")
        self.file_ext = file_ext

    def _get_files_without_ext(self, folder: str) -> set:
        """Return a set of filenames without extension in a folder."""
        return {os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(self.file_ext)}

    def find_images_without_labels(self) -> list:
        """Return a sorted list of image filenames without corresponding labels."""
        image_files = self._get_files_without_ext(self.images_folder)
        label_files = self._get_files_without_ext(self.labels_folder)
        missing_labels = image_files - label_files
        return sorted(f"{img}{self.file_ext}" for img in missing_labels)

    def print_missing_labels(self):
        """Print the images that don't have labels."""
        missing = self.find_images_without_labels()
        if missing:
            return missing
        else:
            return "All images have corresponding labels."


# Example usage
if __name__ == "__main__":
    checker = DataAnnotator("../data_augmentation/augmented/2d")
    checker.print_missing_labels()