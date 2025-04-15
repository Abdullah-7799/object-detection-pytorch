import cv2
import torch
from utils.visualize import draw_boxes

def detect(image_path, model, confidence_thresh=0.5):
    # Load image
    image = cv2.imread(image_path)
    img_tensor = preprocess(image)  # Resize, normalize
        
    # Inference
    with torch.no_grad():
        preds = model(img_tensor)
    
    # Post-processing (NMS)
    boxes = non_max_suppression(preds, confidence_thresh)
    
    # Draw boxes
    output_image = draw_boxes(image, boxes)
    cv2.imwrite("output.jpg", output_image)