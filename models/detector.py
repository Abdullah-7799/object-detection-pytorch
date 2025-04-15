import torch
import torch.nn as nn

class ObjectDetector(nn.Module):
    """Object Detection Model (Backbone + Head)."""
    def __init__(self, backbone, num_classes, anchors):
        super().__init__()
        self.backbone = backbone  # e.g., ResNet
        self.head = DetectionHead(backbone.out_channels, num_classes, anchors)
    
    def forward(self, x):
        features = self.backbone(x)
        preds = self.head(features)
        return preds  # Shape: (batch, num_anchors, 5 + num_classes)

class DetectionHead(nn.Module):
    """Prediction head for bounding boxes and classes."""
    def __init__(self, in_channels, num_classes, anchors):
        super().__init__()
        self.anchors = anchors
        self.conv = nn.Conv2d(in_channels, len(anchors)*(5 + num_classes), kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)