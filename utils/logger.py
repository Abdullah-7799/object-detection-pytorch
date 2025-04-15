from torch.utils.tensorboard import SummaryWriter
import datetime

class Logger:
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = f"logs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value (e.g., loss, accuracy)."""
        self.writer.add_scalar(tag, value, step)
    
    def log_images(self, tag, images, step):
        """Log a list of images with bounding boxes."""
        self.writer.add_images(tag, images, step)
    
    def close(self):
        self.writer.close()