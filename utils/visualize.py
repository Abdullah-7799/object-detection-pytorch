import cv2
import numpy as np

def draw_boxes(image, boxes, class_names, colors=None):
    """Draw bounding boxes and labels on image.
    Args:
        image: Input image (numpy array).
        boxes: List of boxes in format [x1, y1, x2, y2, class_id, confidence].
        class_names: List of class names (e.g., ['cat', 'dog']).
        colors: Optional list of colors for each class.
    Returns:
        Image with drawn boxes.
    """
    if colors is None:
        colors = np.random.randint(0, 255, size=(len(class_names), 3))
    
    for box in boxes:
        x1, y1, x2, y2, class_id, conf = box
        color = colors[class_id]
        label = f"{class_names[class_id]}: {conf:.2f}"
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        
        # Put text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    return image