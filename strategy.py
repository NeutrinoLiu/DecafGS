import math
from dataclasses import dataclass
from typing import Any, Dict, Union, List
import torch

from gsplat.strategy.ops import _update_param_with_optimizer


def op_sigmoid(x, k=100, x0=0.995):
    return 1 / (1 + torch.exp(-k * (x - x0)))

def compute_impact_after_split(
        impacts: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return 1 - (1 - impacts) ** (1. / counts)

def compute_decay_after_split(
        decays: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Compute the new decay after relocating the anchor.

    decay_new = decay_old + log(C)
    so that after exp:
    decay_new = decay_old * C
    
    since opacity = 1 - exp(-precursor/decay)
    
    so that:
    1 - opacity_new = exp(-precursor/decay_new)
                    = exp(-precursor/decay_old * C)
                    = exp(-precursor/decay_old) ^ C
                    = (1 - opacity_old) ^ C
    """
    # return decays
    return decays + torch.log(counts.float())

class DecafMCMCStrategy:
    """Strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`_

    This strategy will:
    - Periodically teleport GSs with low opacity to a place that has high opacity.
    - Periodically introduce new GSs sampled based on the opacity distribution.
    - Periodically perturb the GSs locations.

    Args:
        cap_max (int): Maximum number of GSs. Default to 1_000_000.
        noise_lr (float): MCMC samping noise learning rate. Default to 5e5.
        refine_start_iter (int): Start refining GSs after this iteration. Default to 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default to 25_000.
        refine_every (int): Refine GSs every this steps. Default to 100.
        min_opacity (float): GSs with opacity below this value will be pruned. Default to 0.005.
        verbose (bool): Whether to print verbose information. Default to False.
    """

    def __init__(self, train_cfg, max_cap, verbose = True):
        self.cap_max: int = max_cap
        self.growing_rate: float = train_cfg.growing_rate
        self.noise_lr: float = train_cfg.perturb_intensity
        self.scale_decay: float = train_cfg.scale_decay
        self.refine_start_iter: int = train_cfg.reloc_start_iter
        self.refine_stop_iter: int = train_cfg.reloc_stop_iter
        self.refine_every: int = train_cfg.reloc_every // train_cfg.batch_size
        self.min_opacity: float = train_cfg.reloc_dead_thres
        self.impact_momentum: float = train_cfg.impact_momentum
        self.verbose: bool = verbose

    def initialize_state(self, N, device) -> Dict[str, Any]:
        return {
            "anchor_impacts": torch.ones(N, device=device, dtype=torch.float32),
        }

    def step_pre_backward(self, state, update):
        assert "anchor_impacts" in state, "state should have <anchor_impacts>"
        # nothing need to be done for MCMC strategy pre backward
        # just follow the convention of gsplat strategy
        assert len(update) == state["anchor_impacts"].shape[0], f"shape mismatch: {len(update)} vs {state['anchor_impacts'].shape[0]}"
        state["anchor_impacts"] = self.impact_momentum * state["anchor_impacts"] + (1 - self.impact_momentum) * update


    def step_post_backward(
        self,
        state: Dict[str, Any],
        aks_params: Dict[str, torch.nn.Parameter],
        aks_opts: Dict[str, torch.optim.Optimizer],
        step: int,
        anchor_xyz_lr: float,
    ):
        assert "anchor_impacts" in state, "state should have <anchor_impacts>"
        report = {}
        if (step < self.refine_stop_iter
            and step >= self.refine_start_iter
            and step % self.refine_every == 0):
            # --------------------------------- relocate --------------------------------- #
            n_reloacted_aks, dead_idx, target_idx = self._reloate_anchor(
                state=state,
                aks_params=aks_params,
                opts=aks_opts
            )
            if n_reloacted_aks > 0:
                # update anchor_impacts
                state["anchor_impacts"][target_idx] = compute_impact_after_split(
                    impacts=state["anchor_impacts"][target_idx],
                    counts=torch.bincount(target_idx)[target_idx] + 1,
                )
                state["anchor_impacts"][dead_idx] = state["anchor_impacts"][target_idx]
                if self.verbose:
                    print(f"Step {step}: Relocated {n_reloacted_aks} Anchors.")

            # ---------------------------------- add aks --------------------------------- #
            n_new_aks, growed_idx = self._grow_anchor(
                state=state,
                aks_params=aks_params,
                opts=aks_opts,
            )
            if n_new_aks > 0:
                # add new anchor impacts
                state["anchor_impacts"][growed_idx] = compute_impact_after_split(
                    impacts=state["anchor_impacts"][growed_idx],
                    counts=torch.bincount(growed_idx)[growed_idx] + 1,
                )
                state["anchor_impacts"] = torch.cat(
                    [state["anchor_impacts"],
                     state["anchor_impacts"][growed_idx]])
                if self.verbose:
                    print(
                        f"Step {step}: Added {n_new_aks} Anchors. "
                        f"Now having {aks_params['anchor_offsets'].shape[0]} Anchors."
                    )

            torch.cuda.empty_cache()

            report.update({
                "relocated": n_reloacted_aks if n_reloacted_aks > 0 else None,
                "grew": n_new_aks if n_new_aks > 0 else None
            })

        # --------------------------------- add noise -------------------------------- #
        self._inject_noise_to_position(
            state=state,
            aks_params=aks_params,
            intensity=self.noise_lr * anchor_xyz_lr
        )

        return report

    @torch.no_grad()
    def _reloate_anchor(
        self,
        state: Dict[str, Any],
        aks_params: Dict[str, torch.nn.Parameter],
        opts: Dict[str, torch.optim.Optimizer],
    ):
        anchor_impacts = state["anchor_impacts"]
        N = aks_params["anchor_offsets"].shape[0]
        assert anchor_impacts.shape[0] == N, "shape mismatch"
        dead_mask = anchor_impacts <= self.min_opacity
        n = dead_mask.sum().item()
        if n == 0: return 0, None

        # use neural gs's opacity to decide dead/alive anchor
        dead_idx = dead_mask.nonzero(as_tuple=True)[0]
        alive_idx = (~dead_mask).nonzero(as_tuple=True)[0]
        probs = anchor_impacts[alive_idx]

        sampled_idx = torch.multinomial(probs, n, replacement=True)
        sampled_idx = alive_idx[sampled_idx]

        new_opacity_decay = compute_decay_after_split(
            decays = aks_params["anchor_opacity_decay"][sampled_idx],
            counts = torch.bincount(sampled_idx)[sampled_idx] + 1,
        )

        new_anchor_scale_extend = aks_params["anchor_scale_extend"][sampled_idx] * self.scale_decay
        def param_update(name: str, p: torch.Tensor) -> torch.Tensor:
            if name == "anchor_opacity_decay":
                p[sampled_idx] = new_opacity_decay
            if name == "anchor_scale_extend":
                p[sampled_idx] = new_anchor_scale_extend
            p[dead_idx] = p[sampled_idx]
            return torch.nn.Parameter(p)
        def opt_update(key: str, v: torch.Tensor) -> torch.Tensor:
            v[sampled_idx] = 0
            return v
        
        _update_param_with_optimizer(param_update, opt_update, aks_params, opts)

        return n, dead_idx, sampled_idx

    @torch.no_grad()
    def _grow_anchor(
        self,
        state: Dict[str, Any],
        aks_params: Dict[str, torch.nn.Parameter],
        opts: Dict[str, torch.optim.Optimizer],
    ):
        N = aks_params["anchor_offsets"].shape[0]
        ratio = 1 + self.growing_rate
        N_target = min(self.cap_max, int(ratio * N))
        n = max(0, N_target - N)
        if n == 0: return 0, None

        probs = state["anchor_impacts"]
        sampled_idx = torch.multinomial(probs, n, replacement=True)
        new_opacity_decay = compute_decay_after_split(
            decays = aks_params["anchor_opacity_decay"][sampled_idx],
            counts = torch.bincount(sampled_idx)[sampled_idx] + 1,
        )

        new_anchor_scale_extend = aks_params["anchor_scale_extend"][sampled_idx] * self.scale_decay
        def param_update(name: str, p: torch.Tensor) -> torch.Tensor:
            if name == "anchor_opacity_decay":
                p[sampled_idx] = new_opacity_decay
            if name == "anchor_scale_extend":
                p[sampled_idx] = new_anchor_scale_extend
            p = torch.cat([p, p[sampled_idx]])
            return torch.nn.Parameter(p)
        def opt_update(key: str, v: torch.Tensor) -> torch.Tensor:
            v_new = torch.zeros((len(sampled_idx), *v.shape[1:]), device=v.device)
            return torch.cat([v, v_new])
        
        _update_param_with_optimizer(param_update, opt_update, aks_params, opts)

        return n, sampled_idx

    @torch.no_grad()
    def _inject_noise_to_position(
        self,
        state: Dict[str, Any],
        aks_params: Dict[str, torch.nn.Parameter],
        intensity,
    ):
        noise_resistance = op_sigmoid(1 - state["anchor_impacts"]).unsqueeze(1)
        noise = torch.randn_like(aks_params["anchor_xyz"]) \
                 * noise_resistance * intensity * aks_params["anchor_scale_extend"]
        
        aks_params["anchor_xyz"] += noise