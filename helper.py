import torch
import math
from omegaconf import ListConfig

import threading
import os
import shutil
from PIL import Image
import time
import torchvision.transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import List, Dict, Any

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
        self.onnx = os.path.join(root, "model.onnx")
        self.config = os.path.join(root, "config.json")
        self.log = os.path.join(root, "log.json")
        self.stat = os.path.join(root, "stat.json")
        self.summary = os.path.join(root, "summary.json")
        self.ckpt = self._dir_builder("ckpt", lambda x: f"ckpt_{x}.pth")
        self.render = self._dir_builder("render", lambda x: f"render_{x}.png")
    def _dir_builder(self, dir, fmt):
        return lambda x: os.path.join(self.root, dir, fmt(x))

def save_tensor_images_threaded(img_tensor, gt_tensor, save_path):
    thread = threading.Thread(target=save_tensor_images, args=(img_tensor, gt_tensor, save_path))
    thread.start()
    return 

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
        gt_tensor = gt_tensor / 255.0

    l1diff = torch.abs(img_tensor - gt_tensor)

    # Compute SSIM map using torchmetrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True).to(img_tensor.device)
    # Add batch dimension as required by torchmetrics
    img_batch = img_tensor.unsqueeze(0)
    gt_batch = gt_tensor.unsqueeze(0)
    _, ssim_map = ssim(img_batch, gt_batch)
    
    # Convert SSIM map to RGB for visualization (ssim_map is already in range [0,1])
    ssim_map = ssim_map.squeeze(0)  # Remove batch dimension
    # ssim_map_rgb = ssim_map.repeat(3, 1, 1)  # Convert to RGB

    
    img_pil = to_pil(img_tensor)
    gt_pil = to_pil(gt_tensor)
    l1diff_pil = to_pil(l1diff)
    ssim_map_pil = to_pil(ssim_map)
    
    # Create a new image with twice the width
    w, h = img_pil.size
    combined = Image.new('RGB', (w * 4, h))
    
    # Paste the images side by side
    combined.paste(gt_pil, (0, 0))
    combined.paste(img_pil, (w, 0))
    combined.paste(l1diff_pil, (w*2, 0))
    combined.paste(ssim_map_pil, (w*3, 0))
    
    # Save the combined image
    combined.save(save_path)
    return 

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

@torch.no_grad()
def count_opt_params(optimizer):
    """
    Calculate the total number of parameters being optimized and their memory usage.
    """
    total_params = 0    
    for i, param_group in enumerate(optimizer.param_groups):
        group_params = 0
        for param in param_group['params']:
            group_params += param.numel()
        total_params += group_params
    
    return total_params

# function executiong time profiling decorator
def timeit(func):
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        print(f"{func.__name__} executed in {te-ts:.2f} s")
        return result
    return timed

# -------------------- per batch analysis helper functions ------------------- #

def opacity_analysis(batch_state, K, dead_thres):
    # pc_batched is a list of pc, each pc is a Gaussians instance
    # we need to analyse the batched pc to get the average spawn and opacity
    anchor_avg_spawn = []
    anchor_opacity = []
    ops = batch_state["ops"]
    for op in ops:
        ops_per_anchor = op.detach().reshape(-1, K)
        spawn_per_anchor = torch.sum(ops_per_anchor > dead_thres, dim=-1)
        opacity_per_anchor = torch.mean(ops_per_anchor, dim=-1)
        anchor_avg_spawn.append(spawn_per_anchor)
        anchor_opacity.append(opacity_per_anchor)
    anchor_avg_spawn = sum(anchor_avg_spawn) / len(ops)
    anchor_opacity = sum(anchor_opacity) / len(ops)

    return anchor_avg_spawn, anchor_opacity

# borrowed from gsplat.strategy.default, used to calculate grad2d for gs
def calculate_grads(
        batch_state: Dict[str, Any],
        N: int,
        K: int,
        device,
        packed: bool = True,
        absgrad: bool = True,
    ):
        batched_info = batch_state["info"]
        key_for_gradient = "means2d"
        n_gaussian = K * N
        # initialize state on the first run
        grad2d = torch.zeros(n_gaussian, device=device)
        count = torch.zeros(n_gaussian, device=device)

        for info in batched_info:
            filter_idx = info["filter_idx"]

            # normalize grads to [-1, 1] screen space
            if absgrad:
                grads = info[key_for_gradient].absgrad.clone()
            else:
                grads = info[key_for_gradient].grad.clone()

            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

            # update the running state
            if packed:
                # grads is [nnz, 2]
                gs_ids = info["gaussian_ids"]  # [nnz]
            else:
                # grads is [C, N, 2]
                sel = info["radii"] > 0.0  # [C, N]
                gs_ids = torch.where(sel)[1]  # [nnz]
                grads = grads[sel]  # [nnz, 2]

            gs_ids = filter_idx[gs_ids]     # map from filtered gs to full NxK gs
            grad2d.index_add_(0, gs_ids, grads.norm(dim=-1))
            count.index_add_(
                0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
            )

        # mapper from child gs to anchors
        grad2d_aks = grad2d.reshape(-1, K).sum(dim=-1)
        count_aks = count.reshape(-1, K).sum(dim=-1)

        return grad2d_aks, count_aks

def normalize(v, eps=1e-6):
    return (v - v.min()) / (v.max() - v.min() + eps)

def ewma_update(d, new_kv, alpha=0.9):
    for k, v in new_kv.items():
        if d.get(k) is not None:
            d[k] = d[k] * alpha + v * (1 - alpha)
        else:
            d[k] = v