import torch
import math
from omegaconf import ListConfig

import threading
import os
import shutil
from PIL import Image, ImageFilter
import cv2
import imageio
import time
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Dict, Any
from interface import Gaussians
import numpy as np


def cached_func(func, *args, **kwargs):
    ret = func(*args, **kwargs)
    return lambda x: ret

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
        soft_link = f"{root}_latest"
        if os.path.exists(soft_link):
            os.remove(soft_link)
        if os.path.exists(root):
            root = f"{root}_{cur_time()}"
        os.makedirs(root, exist_ok=True)
        abs_root = os.path.abspath(root)
        os.system(f"ln -s {abs_root} {soft_link}")
        print(f"Log directory: {root}")

        self.root = root
        self.tb = root
        self.onnx = os.path.join(root, "model.onnx")
        self.config = os.path.join(root, "config.json")
        self.log = os.path.join(root, "log.json")
        self.stat = os.path.join(root, "stat.json")
        self.summary = os.path.join(root, "summary.json")
        self.chkpt = self._dir_builder("chkpt", lambda x: f"chkpt_{x}.pth")
        self.render = self._dir_builder("render", lambda x: f"render_{x}.png")
    def _dir_builder(self, dir, fmt):
        return lambda x: os.path.join(self.root, dir, fmt(x))

def threaded(fn, *args, **kwargs):
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
    thread.start()
    return thread

def save_tensor_images_threaded(img_tensor, gt_tensor, save_path):
    thread = threading.Thread(target=save_tensor_images, args=(img_tensor, gt_tensor, save_path))
    thread.start()
    return 

def save_video(img_list, save_path, fps=30):
    """
    Save a list of [3,H,W] tensors as a mp4 video.
    
    Args:
        img_list (List[torch.Tensor]): List of image tensors [3,H,W]
        save_path (str): Path to save the video
        fps (int): Frames per second
    """

    dir = os.path.dirname(save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    height, width = img_list[0].shape[1:]
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for img in img_list:
        img_np = img.permute(1, 2, 0).cpu().numpy() * 255
        img_np = img_np.astype(np.uint8)
        video.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    video.release()
    print(f"Video saved to {save_path}")


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

    # Convert tensors to PIL images
    to_pil = T.ToPILImage()

    if gt_tensor==None:
        # save the image tensor directly
        img_pil = to_pil(img_tensor)
        img_pil.save(save_path)
        return

    # Ensure the tensors are in the correct format
    assert img_tensor.shape == gt_tensor.shape, "Tensors must have the same shape"
    assert len(img_tensor.shape) == 3 and img_tensor.shape[0] == 3, "Tensors must be [3,H,W]"

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
            lr_start = attr_lr[2] if len(attr_lr) > 2 else 0
            lr_stop = attr_lr[3] if len(attr_lr) > 3 else max_step
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
                lr_lambda_builder(ratio, lr_start, lr_stop)
            )
            # print(f"lr for {attr_name} initialized with exp decay: ({lr_init}->{lr_end})")
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
            # print(f"lr for {attr_name} initialized with constant: {attr_lr}")
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

def opacity_analysis(batch_state, K, dead_thres, per_frame=False):
    # retain frame info
    # opacity is camera invariant
    anchor_spawn = {}
    anchor_opacity = {}
    ops = batch_state["ops"]
    for (op, f) in zip(ops, batch_state["frames"]):
        ops_per_anchor = op.detach().reshape(-1, K)
        spawn_per_anchor = torch.sum(ops_per_anchor > dead_thres, dim=-1)
        opacity_per_anchor = torch.mean(ops_per_anchor, dim=-1)
        anchor_spawn[f] = spawn_per_anchor
        anchor_opacity[f] = opacity_per_anchor
    if not per_frame:
        anchor_spawn = sum(anchor_spawn.values()) / len(anchor_spawn)
        anchor_opacity = sum(anchor_opacity.values()) / len(anchor_opacity)

    return anchor_spawn, anchor_opacity

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

def get_gs_idx_from_tile(offsets, records, query_tile, max_num):
    tile_num = len(offsets)
    start = offsets[query_tile]
    end = offsets[query_tile + 1] if query_tile < tile_num - 1 else len(records)
    end = min(start + max_num, end)
    return records[start:end]

# not necessary for each batch, but needed if error-oriented densification is expected
def calculate_blames(
        batch_state: Dict[str, Any],
        N: int,
        K: int,
        device,
        max_gs_per_tile: int = 1000,
        per_frame: bool = False,
        ):
    
    topk = 200
    dssim_min = 0.1
    n_gaussian = K * N
    batched_info = batch_state["info"]
    batched_img =  batch_state["img"]
    batched_gt = batch_state["gt"]
    batched_frames = batch_state["frames"]

    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True).to(device)
    if per_frame:
        blame_per_frame = {f: torch.zeros(n_gaussian, device=device) for f in batched_frames}
    else:
        blame = torch.zeros(n_gaussian, device=device)

    for (info, img, gt, frame) in zip(batched_info, batched_img, batched_gt, batched_frames):

        if per_frame:
            blame = blame_per_frame[frame]

        tile_size = info["tile_size"]

        # --------------------------- get a list of tile id -------------------------- #
        ssim_error = (1 - ssim_fn(img[None], gt[None])[-1].squeeze().mean(dim=0)) / 2
        # ssim_error = torch.abs(img-gt).mean(dim=0)

        ssim_error_tiled = tile_average(ssim_error, tile_size)
        ssim_error_tiled = ssim_error_tiled.flatten()
        topk_ssim, topk_idx = torch.topk(ssim_error_tiled, topk)
        # remove those tiles with low dssim
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
        culling_idx = info["gaussian_ids"]

        # add bad tiles's ssim to the gs blame
        for idx, score in zip(topk_idx, topk_ssim):
            gs_idx = get_gs_idx_from_tile(isect_offsets, isect_ids, idx, max_gs_per_tile)
            gs_idx_flatten = filter_idx[culling_idx[gs_idx]]
            blame[gs_idx_flatten] += score
    
    if per_frame:
        blame_aks = {f: blame_per_frame[f].reshape(-1, K).sum(dim=-1) for f in batched_frames}
    else:
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
        per_frame: bool = False,
    ):
        batched_info = batch_state["info"]
        batched_frames = batch_state["frames"]
        key_for_gradient = "means2d"
        n_gaussian = K * N
        # initialize state on the first run
        if per_frame:
            grad2d_per_frame = {
                f: torch.zeros(n_gaussian, device=device) for f in batched_frames
            }
            count_per_frame = {
                f: torch.zeros(n_gaussian, device=device) for f in batched_frames
            }
        else:
            grad2d = torch.zeros(n_gaussian, device=device)
            count = torch.zeros(n_gaussian, device=device)

        for info, frame in zip(batched_info, batched_frames):
            filter_idx = info["filter_idx"]
            if per_frame:
                grad2d = grad2d_per_frame[frame]
                count = count_per_frame[frame]

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
        if per_frame:
            grad2d_aks = {
                f: grad2d_per_frame[f].reshape(-1, K).sum(dim=-1) for f in batched_frames}
            count_aks = {
                f: count_per_frame[f].reshape(-1, K).sum(dim=-1) for f in batched_frames}
        else:
            grad2d_aks = grad2d.reshape(-1, K).sum(dim=-1)
            count_aks = count.reshape(-1, K).sum(dim=-1)

        return grad2d_aks, count_aks

def normalize(v, eps=1e-6, simple=False):
    if simple:
        return v / (v.max()+eps)
    return (v - v.min()) / (v.max() - v.min() + eps)
def standardize(v, eps=1e-6):
    ret = ((v - v.mean()) / (v.std() + eps) + 1) /2
    return torch.clamp(ret, 0, 1)

def ewma_update(d, new_kv, alpha=0.9):
    for k, v in new_kv.items():
        if d.get(k) is not None and v is not None:
            d[k] = d[k] * alpha + v * (1 - alpha)
        else:
            d[k] = v

def update_state(d, new_history, special_aggr={}):
    ret = {}
    if d.get("history", None) is None:
        d["history"] = {}
    for k, hist in new_history.items():
        if hist is None:
            continue
        d["history"][k] = d["history"].get(k, {})
        d["history"][k].update(hist)
        full_hist = d["history"][k]
        d["history_length"] = len(full_hist)
        if k not in special_aggr:
            d[k] = sum(full_hist.values()) / len(full_hist)
        else:
            d[k] = special_aggr[k](full_hist)
        ret[k] = d[k].clone()
    return ret
    

def gaussian_blur(img, radius):
    img = img.clamp(0, 1)  # Ensure tensor is within range
    img_pil = T.ToPILImage()(img)
    blurred_pil = img_pil.filter(ImageFilter.GaussianBlur(radius))
    blurred_tensor = T.ToTensor()(blurred_pil)
    return blurred_tensor.to(img.device)

def gaussian_blur_diff(img: torch.Tensor, r: float) -> torch.Tensor:
    """
    Apply a differentiable Gaussian blur to an image.
    
    Parameters:
        img (torch.Tensor): Input image tensor of shape (3, H, W).
        r (float): Blur radius (sigma) for the Gaussian.
        
    Returns:
        torch.Tensor: Blurred image tensor of shape (3, H, W).
    """
    # If r is zero or negative, return the image unchanged.
    if r <= 0:
        return img

    # Determine kernel size: cover roughly ±3σ.
    radius = math.ceil(3 * r)
    kernel_size = 2 * radius + 1

    # Create a 1D coordinate tensor centered at zero.
    x = torch.arange(kernel_size, dtype=img.dtype, device=img.device) - radius

    # Compute the Gaussian kernel (without the 1/(sqrt(2pi)*sigma) factor, as we'll normalize later).
    gauss_kernel = torch.exp(-(x ** 2) / (2 * r ** 2))
    gauss_kernel = gauss_kernel / gauss_kernel.sum()  # Normalize the kernel

    # Reshape for separable convolution
    gauss_kernel_h = gauss_kernel.view(1, 1, 1, kernel_size)  # horizontal kernel
    gauss_kernel_v = gauss_kernel.view(1, 1, kernel_size, 1)  # vertical kernel

    # Add a batch dimension (now shape is [1, 3, H, W])
    img = img.unsqueeze(0)

    # Apply horizontal convolution
    img = F.pad(img, pad=(radius, radius, 0, 0), mode='reflect')
    kernel_h = gauss_kernel_h.repeat(img.shape[1], 1, 1, 1)  # one kernel per channel
    img = F.conv2d(img, weight=kernel_h, groups=img.shape[1])

    # Apply vertical convolution
    img = F.pad(img, pad=(0, 0, radius, radius), mode='reflect')
    kernel_v = gauss_kernel_v.repeat(img.shape[1], 1, 1, 1)
    img = F.conv2d(img, weight=kernel_v, groups=img.shape[1])

    # Remove the batch dimension
    return img.squeeze(0)
    
def calc_tempo_decay(x, mean, std, steepness, wider=1.):
    # print(f"""devices:
    #       x: {x.device}
    #       mean: {mean.device}
    #       std: {std.device}
    #       steepness: {steepness.device}""")
    # no activation for mean
    std_wider = torch.sigmoid(std) * wider  # 0-1 to slightly wider range
    beta = torch.sigmoid(steepness) * 10 + 1 # 1-infinity
    exponent = - (
        ( (x - mean) / (std_wider + 1e-6)
        ) ** 2
        ) ** beta
    # print(f"at {x}: {mean.data}, {std_wider.data}")
    decay  = torch.exp(exponent).clamp(0.01, 1)
    # print(f"at {x}: {mean.data}, {std_wider.data}")
    # print(f"decay is {decay.data}")
    return decay

def calculate_ratio(left, right, valid):
    """
    Computes the length of overlap for each data point in [left, right] with the `valid` binary tensor.

    Args:
        left (torch.Tensor): Tensor of shape (N,) representing the left bounds of the intervals.
        right (torch.Tensor): Tensor of shape (N,) representing the right bounds of the intervals.
        valid (torch.Tensor): Binary tensor of shape (T,) indicating valid positions (1 for valid, 0 otherwise).

    Returns:
        torch.Tensor: Tensor of shape (N,) representing the length of overlap for each interval in [left, right] with valid.
    """
    # Ensure left and right bounds are within the range of T
    T = valid.size(0)
    V = valid.sum()
    left_clamped = torch.clamp(left, 0, T - 1)
    right_clamped = torch.clamp(right, 0, T - 1)

    # Create a mask for all intervals
    N = left.size(0)
    range_tensor = torch.arange(T, device=valid.device).unsqueeze(0).expand(N, -1)

    # Generate masks for each interval
    interval_masks = (range_tensor >= left_clamped.unsqueeze(1)) & (range_tensor <= right_clamped.unsqueeze(1))

    # Compute overlaps using batch matrix multiplication
    overlaps = torch.matmul(interval_masks.float(), valid.float().unsqueeze(1)).squeeze(1).long()
    overlaps = torch.clamp(overlaps, 1, V)

    return overlaps / V

@torch.no_grad()
def calculate_perturb(
    pc,
    intensity: float = 0.0,
):
    # add noise to indexed anchors
    if intensity == 0: return torch.zeros_like(pc.means)

    def op_sigmoid(x, k=100, x0=0.995):
        """higher opacity, less noise"""
        return 1 / (1 + torch.exp(-k * (x - x0)))
    
    ops = pc.opacities
    means = pc.means
    scales = pc.scales
    rotations = pc.quats

    # learn from mcmc
    L = build_cov(scales, rotations)
    cov = L @ L.transpose(1, 2)

    noise_resistance = op_sigmoid(1 - ops).unsqueeze(1)
    noise = torch.randn_like(means) * noise_resistance * intensity
    noise = torch.bmm(cov, noise.unsqueeze(-1)).squeeze(-1)
    # print(f"shape of noise: {noise.shape}")
    # print(f"norm of noise: {noise.norm(dim=-1).mean()}")
    # print(f"std of noise: {noise.norm(dim=-1).std()}")

    return noise

def build_rotation(r):
    # r is [N, 4]
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.shape[0], 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_cov(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def mul_quat(q1, q2):
    """
    q1: [N, 4]
    q2: [N, 4]
    """
    a1, b1, c1, d1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    a2, b2, c2, d2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
    return torch.stack([a, b, c, d], dim=-1)

def inverse_sigmoid(x, eps=1e-6):
    """Safe inverse sigmoid with numerical stability."""
    x = x.clamp(eps, 1-eps)
    return torch.log(x / (1 - x))


def decay(precursor, decay, naive=False):
    """
    decay opacities
    to facilitize opacity split during densification
    check: 
    https://arxiv.org/abs/2404.06109
    https://arxiv.org/abs/2404.09591
    """
    if naive:
        return torch.sigmoid(decay) * precursor
    exp_decay = torch.exp(decay)
    eps = 1e-6
    header = precursor / (exp_decay + eps)
    return 1 - torch.exp(-header)

def update_decay_after_split(
        decays: torch.Tensor,
        counts: torch.Tensor,
        naive=False,
        reloc_ops_decay=0.8) -> torch.Tensor:
    """Compute the new decay after relocating the anchor.

    # ----------------------------- none naive setup ----------------------------- #
    decay_new = decay_old + log(C)
    so that after exp:
    decay_new = decay_old * C
    
    since opacity = 1 - exp(-precursor/decay)
    
    so that:
    1 - opacity_new = exp(-precursor/decay_new)
                    = exp(-precursor/decay_old / C)
                    = exp(-precursor/decay_old) ^ (1/C)
                    = (1 - opacity_old) ^ (1/C)
    (1 - opacity_new) ^ C = 1 - opacity_old

    # ----------------------------- naive setup ----------------------------- #
    decay_new = decay_old - log(0.8), simply enlarge the decay
    """
    if naive:
        before = torch.sigmoid(decays)
        after = before * reloc_ops_decay
        return inverse_sigmoid(after)
    return decays + torch.log(counts.float())