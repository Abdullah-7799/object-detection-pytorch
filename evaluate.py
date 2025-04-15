import torch
from utils.metrics import calculate_map

def evaluate(model, dataloader, device="cuda"):
    """Evaluate model on validation dataset.
    Args:
        model: Trained model.
        dataloader: Validation dataloader.
    Returns:
        mAP score.
    """
    model.eval()
    pred_boxes_all = []
    true_boxes_all = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            preds = model(images)
            
            # Convert predictions to boxes format
            pred_boxes = postprocess(preds)  # [x1, y1, x2, y2, class, conf]
            pred_boxes_all.extend(pred_boxes)
            
            # Convert ground truth to boxes format
            true_boxes = convert_targets(targets)
            true_boxes_all.extend(true_boxes)
    
    # Calculate mAP
    map_score = calculate_map(pred_boxes_all, true_boxes_all)
    return map_score