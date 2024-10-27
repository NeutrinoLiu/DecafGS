import torch
import math
from omegaconf import ListConfig

import os
import shutil
from PIL import Image
import torchvision.transforms as T

def cur_time():
    import datetime
    return datetime.datetime.now().strftime("%H-%M-%S")

class GlobalWriter:
    _writer = None
    _step = 0
    def __init__(self, writer):
        if GlobalWriter._writer is None:
            print("GlobalWriter initialized")
        else:
            print("GlobalWriter re-initialized")
        GlobalWriter._writer = writer
        GlobalWriter._step = 0

    @classmethod
    def add_scalar(cls, label, data, step=None, **kwargs):
        if cls._writer is None:
            print("GlobalWriter not initialized")
            return
        if step is None:
            step = cls._step
        else:
            cls._step = step
        cls._writer.add_scalar(label, data, step, **kwargs)
    @classmethod
    def add_histogram(cls, label, data, step=None, **kwargs):
        if cls._writer is None:
            print("GlobalWriter not initialized")
            return
        if step is None:
            step = cls._step
            if step % 100 != 0:
                return
        else:
            cls._step = step
        cls._writer.add_histogram(label, data, step, **kwargs)


class LogDirMgr:
    def __init__(self, root):
        if os.path.exists(root):
            # overwrite = input(f"Log directory {root} already exists. Press <y> to overwrite: ")
            # workaround = overwrite.lower() != "y"
            if True:
                root = f"{root}_{cur_time()}"
            else:
                shutil.rmtree(root)
        os.makedirs(root)
        print(f"Log directory: {root}")

        self.root = root
        self.tb = root
        self.config = os.path.join(root, "config.json")
        self.log = os.path.join(root, "log.json")
        self.stat = os.path.join(root, "stat.json")
        self.ckpt = self._dir_builder("ckpt", lambda x: f"ckpt_{x}.pth")
        self.render = self._dir_builder("render", lambda x: f"render_{x}.png")
    def _dir_builder(self, dir, fmt):
        return lambda x: os.path.join(self.root, dir, fmt(x))

def save_tensor_images(img_tensor, gt_tensor, save_path):
    """
    Save two [3,H,W] tensors side by side as a single image.
    
    Args:
        img_tensor (torch.Tensor): First image tensor [3,H,W]
        gt_tensor (torch.Tensor): Second image tensor [3,H,W]
        save_path (str): Path to save the combined image
    """
    dir = os.path.dirname(save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Ensure the tensors are in the correct format
    assert img_tensor.shape == gt_tensor.shape, "Tensors must have the same shape"
    assert len(img_tensor.shape) == 3 and img_tensor.shape[0] == 3, "Tensors must be [3,H,W]"
    
    # Convert tensors to PIL images
    to_pil = T.ToPILImage()
    
    # If tensors are not in range [0,1], normalize them
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0
    if gt_tensor.max() > 1.0:
        gt_tensor = gt_tensor / 255.0
    
    img_pil = to_pil(img_tensor)
    gt_pil = to_pil(gt_tensor)
    
    # Create a new image with twice the width
    w, h = img_pil.size
    combined = Image.new('RGB', (w * 2, h))
    
    # Paste the images side by side
    combined.paste(img_pil, (0, 0))
    combined.paste(gt_pil, (w, 0))
    
    # Save the combined image
    combined.save(save_path)
    
    return combined

def get_adam_and_lr_sched(to_be_optimized, opt_cali, max_step):
    ret_opts = {}
    ret_lr_sched = {}
    for attr_name, attr, attr_lr in to_be_optimized:
        if isinstance(attr_lr, ListConfig):
            assert len(attr_lr) >= 2, "lr list should have at least 2 elements"
            lr_init = attr_lr[0]
            lr_end = attr_lr[1]
            gamma = (lr_end / lr_init) ** (1.0 / max_step)
            ret_opts[attr_name] = torch.optim.Adam(
                [{
                    'name': attr_name,
                    'params': attr,
                    'lr': lr_init * math.sqrt(opt_cali)
                }],
                eps=1e-15 / math.sqrt(opt_cali),
                betas=(1 - opt_cali * (1 - 0.9), 1 - opt_cali * (1 - 0.999))
            )
            ret_lr_sched[attr_name] = torch.optim.lr_scheduler.ExponentialLR(
                ret_opts[attr_name],
                gamma=gamma,
            )
            print(f"lr for {attr_name} initialized with exp decay: ({lr_init}->{lr_end})")
        else:
            ret_opts[attr_name] = torch.optim.Adam(
                [{
                    'name': attr_name,
                    'params': attr,
                    'lr': attr_lr * math.sqrt(opt_cali)
                }],
                eps=1e-15 / math.sqrt(opt_cali),
                betas=(1 - opt_cali * (1 - 0.9), 1 - opt_cali * (1 - 0.999))
            )
            print(f"lr for {attr_name} initialized with constant: {attr_lr}")
    print()
    return ret_opts, ret_lr_sched


def mem_profile_start():
    torch.cuda.memory._record_memory_history(
        max_entries=100_000
    )

def mem_profile_end():
    try:
        torch.cuda.memory._dump_snapshot(f"mem.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")

    # Stop recording memory snapshot history.
    torch.cuda.memory._record_memory_history(enabled=None)