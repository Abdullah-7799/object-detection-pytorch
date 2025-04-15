import os
import cv2
import yaml
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import albumentations as A

class CustomDataset:
    """YOLO-formatted dataset loader with augmentations.
    Args:
        img_dir (str): Path to images folder.
        label_dir (str): Path to labels folder.
        transforms (albumentations.Compose): Augmentations pipeline.
        img_size (int): Target image size (square).
    """
    def __init__(self, img_dir, label_dir, transforms=None, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.img_size = img_size
        self.image_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
        self.class_names = self._load_class_names()

    def _load_class_names(self):
        """Load class names from labels.yaml."""
        with open(os.path.join(self.label_dir, "labels.yaml"), 'r') as f:
            return yaml.safe_load(f)['names']

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label (YOLO format: [class, x_center, y_center, w, h] normalized)
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            bboxes = np.array([list(map(float, line.strip().split())) for line in f.readlines()])

        # Apply augmentations
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=bboxes[:, 1:], class_labels=bboxes[:, 0])
            image, bboxes = transformed['image'], transformed['bboxes']
            bboxes = np.column_stack((transformed['class_labels'], bboxes))

        # Convert to tensor and resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        return image, bboxes

def create_train_val_split(data_root, val_ratio=0.2):
    """Split dataset into train/val and generate YAML config.
    Args:
        data_root (str): Root folder containing 'images' and 'labels' subfolders.
        val_ratio (float): Ratio for validation split (0.0 to 1.0).
    """
    # Get all images and labels
    images = sorted(glob(os.path.join(data_root, "images/*.jpg")))
    labels = [img.replace('images', 'labels').replace('.jpg', '.txt') for img in images]

    # Split dataset
    train_img, val_img, train_lbl, val_lbl = train_test_split(
        images, labels, test_size=val_ratio, random_state=42
    )

    # Generate YAML config
    config = {
        'train': {'images': train_img, 'labels': train_lbl},
        'val': {'images': val_img, 'labels': val_lbl},
        'names': ['class1', 'class2']  # Update with your classes
    }

    with open(os.path.join(data_root, "dataset.yaml"), 'w') as f:
        yaml.dump(config, f)

# Example usage
if __name__ == "__main__":
    # Define augmentations
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Initialize dataset
    dataset = CustomDataset(
        img_dir="data/images",
        label_dir="data/labels",
        transforms=transforms
    )