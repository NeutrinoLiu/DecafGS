import torch
import math
from omegaconf import ListConfig

import threading
import os
import shutil
from PIL import Image
import time
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import LambdaLR
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
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True).to(img_tensor.device)
    # Add batch dimension as required by torchmetrics
    img_batch = img_tensor.unsqueeze(0)
    gt_batch = gt_tensor.unsqueeze(0)
    _, ssim_map = ssim_fn(img_batch, gt_batch)
    
    # Convert SSIM map to RGB for visualization (ssim_map is already in range [0,1])
    ssim_map = ((1-ssim_map)/2).squeeze(0)  # Remove batch dimension
    ssim_map = ssim_map.mean(dim=0).expand(3, -1, -1)  # Average over channels
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

def lr_lambda_builder(ratio, min_step, max_step):
    # build exponential decay lr scheduler
    def lr_lambda(step):
        if step < min_step:
            return 1.
        if step > max_step:
            return ratio
        return ratio ** ((step-min_step) / (max_step-min_step))
    return lr_lambda

def get_adam_and_lr_sched(to_be_optimized, opt_cali, max_step):
    ret_opts = {}
    ret_lr_sched = {}
    for attr_name, attr, attr_lr in to_be_optimized:
        if isinstance(attr_lr, ListConfig):
            assert len(attr_lr) >= 2, "lr list should have at least 2 elements"
            lr_init = attr_lr[0]
            lr_end = attr_lr[1]
            lr_delay = attr_lr[2] if len(attr_lr) == 3 else 0
            ratio = lr_end / lr_init
            ret_opts[attr_name] = torch.optim.Adam(
                [{
                    'name': attr_name,
                    'params': attr,
                    'lr': lr_init * math.sqrt(opt_cali)
                }],
                eps=1e-15 / math.sqrt(opt_cali),
                betas=(1 - opt_cali * (1 - 0.9), 1 - opt_cali * (1 - 0.999))
            )
            ret_lr_sched[attr_name] = LambdaLR(
                ret_opts[attr_name], 
                lr_lambda_builder(ratio, lr_delay, max_step)
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

def tile_average(image, c):
    H, W = image.shape
    pad_h = (c - H % c) % c  # Calculate how much to pad in height
    pad_w = (c - W % c) % c  # Calculate how much to pad in width
    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    tiles = padded_image.unfold(0, c, c).unfold(1, c, c)  # shape: (new_H//c, new_W//c, c, c)
    tile_means = tiles.mean(dim=(-1, -2))  # shape: (new_H//c, new_W//c)
    return tile_means

def mask_image_by_tile(image, mask, c): # mask in [h//c, w//c]
    # Step 1: Pad the image so H and W are multiples of c
    H, W = image.shape
    pad_h = (c - H % c) % c  # Calculate how much to pad in height
    pad_w = (c - W % c) % c  # Calculate how much to pad in width
    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    H, W = padded_image.shape
    h, w = H // c, W // c
    if len(mask.shape) == 1:
        assert mask.shape[0] == h * w, "mask shape mismatch"
        mask = mask.reshape(h, w)
    # Step 2: Reshape into tiles
    tiles = padded_image.unfold(0, c, c).unfold(1, c, c)  # shape: (new_H//c, new_W//c, c, c)
    mask_tiles = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, c, c)
    # Step 3: Calculate the mean in each tile
    masked_tiles = tiles * mask_tiles
    return masked_tiles.permute(0, 2, 1, 3).contiguous().view(h * c, w * c)

def save_grey_image(img_tensor, save_path):
    # input tensor should be [H, W], in range [0, 1]
    img_pil = T.ToPILImage()(img_tensor)
    img_pil.save(save_path)

def get_gs_idx_from_tile(offsets, records, query_tile):
    tile_num = len(offsets)
    start = offsets[query_tile]
    end = offsets[query_tile + 1] if query_tile < tile_num - 1 else len(records)
    return records[start:end]

# not necessary for each batch, but needed if error-oriented densification is expected
def calculate_blames(
        batch_state: Dict[str, Any],
        N: int,
        K: int,
        device):
    
    topk = 200
    dssim_min = 0.1
    n_gaussian = K * N
    batched_info = batch_state["info"]
    batched_img =  batch_state["img"]
    batched_gt = batch_state["gt"]
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True).to(device)
    blame = torch.zeros(n_gaussian, device=device)

    for (info, img, gt) in zip(batched_info, batched_img, batched_gt):

        tile_size = info["tile_size"]

        # --------------------------- get a list of tile id -------------------------- #
        ssim_error = (1 - ssim_fn(img[None], gt[None])[-1].squeeze().mean(dim=0)) / 2
        # ssim_error = torch.abs(img-gt).mean(dim=0)

        ssim_error_tiled = tile_average(ssim_error, tile_size)
        ssim_error_tiled = ssim_error_tiled.flatten()
        topk_ssim, topk_idx = torch.topk(ssim_error_tiled, topk)
        # remove idx by ssim larger than dssim_min
        topk_idx = topk_idx[topk_ssim > dssim_min]
        topk_ssim = topk_ssim[topk_ssim > dssim_min]
        
        # with open("topk.txt", "a") as f:
        #     sorted_tensor = torch.sort(topk_ssim, descending=True).values
        #     # Print elements with spacing
        #     f.write(' '.join(map(str, sorted_tensor.tolist())) + '\n')

        # ----------------------- save masked ssim map for ref ----------------------- #
        # topk_mask = torch.ones_like(ssim_error_tiled) * 0.3
        # topk_mask[topk_idx] = 1
        # masked_ssim = mask_image_by_tile(ssim_error, topk_mask, tile_size)
        # name = torch.randint(0, 10, (1,)).item()
        # thread = threading.Thread(target=save_grey_image, args=(masked_ssim, f"masked_ssim_{name}.png"))
        # thread.start()
        
        isect_offsets = info["isect_offsets"].flatten()
        filter_idx = info["filter_idx"]
        isect_ids = info["flatten_ids"]
        # add bad tiles's ssim to the gs blame
        for idx, score in zip(topk_idx, topk_ssim):
            gs_idx = get_gs_idx_from_tile(isect_offsets, isect_ids, idx)
            gs_idx = filter_idx[gs_idx]
            blame[gs_idx] += score
    
    blame_aks = blame.reshape(-1, K).sum(dim=-1)
    return blame_aks

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