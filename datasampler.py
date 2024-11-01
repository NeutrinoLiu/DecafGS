import random
import numpy as np

# ------------------- C&F sampler, generate triads directly ------------------ #

def sequential_sampler(cams, frames, downscale, info=None, batch_size=None):
    """
    sequential sampling policy
    """
    return [(c, f, downscale) for c in cams for f in frames]

# ------------------------------- frame sampler ------------------------------ #
def uniform_frame_sampler(frames, K):
    """
    uniform frame sampling policy
    """
    return np.random.choice(frames, K, replace=True)
def worst_frame_sampler(frames, K, psnr):
    """
    worst frame sampling policy
    """
    assert len(frames) == len(psnr), "frame and psnr mismatch"
    prob = (psnr.max() - psnr) / (psnr.max() - psnr.min() + 1e-6)
    ret = np.random.choice(frames, K, replace=True, p=prob)
    return ret

# ------------------------------ camera sampler ------------------------------ #
def uniform_camera_sampler(cams, K):
    """
    uniform camera sampling policy
    """
    return np.random.choice(cams, K, replace=True)
def max_parallex_sampler(cams, K, distance, batch_size):
    """
    max parallex sampling policy
    in the actual paper, lets find a fancy name for it
    """
    assert distance.shape == (len(cams), len(cams)), "affinity shape mismatch"
    def remove_col(mat, idx):
        return np.delete(mat, idx, axis=1)
    def remove_ele(mat, idx):
        return np.delete(mat, idx, axis=0)

    prev_idx = []
    ret_idx = []

    aff = distance.copy()
    cam_idx = np.arange(len(cams))
    while len(ret_idx) < K:
        if len(ret_idx) == 0:
            idx = random.choice(cam_idx)
            ret_idx.append(idx)
            prev_idx.append(idx)
            cam_idx = remove_ele(cam_idx, idx)
            aff = remove_col(aff, idx)

        else:
            scoring = aff[prev_idx, :].sum(axis=0)
            p = scoring / scoring.sum()
            assert scoring.shape == cam_idx.shape, "scoring shape mismatch"
            idx = np.random.choice(len(cam_idx), p=p)

            ret_idx.append(cam_idx[idx])
            prev_idx.append(cam_idx[idx])
            cam_idx = remove_ele(cam_idx, idx)
            aff = remove_col(aff, idx)
            if prev_idx == 2 * batch_size:
                prev_idx = prev_idx[batch_size:]

        if len(cam_idx) == 0:
            aff = distance.copy()
            cam_idx = np.arange(len(cams))
    
    return [cams[i] for i in ret_idx]