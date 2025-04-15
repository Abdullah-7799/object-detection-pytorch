def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes.
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        IoU score (0 to 1).
    """
    # Calculate intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union

def calculate_map(pred_boxes, true_boxes, iou_thresh=0.5):
    """Calculate mean Average Precision (mAP).
    Args:
        pred_boxes: List of predicted boxes with confidence scores.
        true_boxes: List of ground truth boxes.
    Returns:
        mAP score.
    """
    # Sort predictions by confidence (descending)
    pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)
    
    TP = np.zeros(len(pred_boxes))
    FP = np.zeros(len(pred_boxes))
    
    for i, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for j, true in enumerate(true_boxes):
            iou = calculate_iou(pred[:4], true[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_thresh:
            TP[i] = 1
            true_boxes.pop(best_gt_idx)  # Remove matched GT
        else:
            FP[i] = 1
    
    # Calculate precision-recall curve
    TP_cumsum = np.cumsum(TP)
    FP_cumsum = np.cumsum(FP)
    recalls = TP_cumsum / len(true_boxes)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
    
    # Calculate AP (Area Under PR Curve)
    ap = np.trapz(precisions, recalls)
    return ap