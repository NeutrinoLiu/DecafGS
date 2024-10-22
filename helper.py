import torch
import math
from omegaconf import ListConfig

import os
from PIL import Image
import torchvision.transforms as T

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

def get_adam_and_lr_sched(to_be_optimized, opt_cali):
    ret_opts = {}
    ret_lr_sched = {}
    for attr_name, attr, attr_lr in to_be_optimized:
        if isinstance(attr_lr, ListConfig):
            assert len(attr_lr) == 4, "lr list should have 4 elements"
            lr_init = attr_lr[0]
            lr_end = attr_lr[1]
            lr_decay = attr_lr[2]
            max_iter = attr_lr[3]
            gamma = (lr_end / lr_init) ** (1.0 / max_iter)
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