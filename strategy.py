import math
from dataclasses import dataclass
from typing import Any, Dict, Union, List
import torch

from gsplat.strategy.ops import _update_param_with_optimizer
from helper import update_decay_after_split

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

    def __init__(self, train_cfg, max_cap, verbose = True, naive_decay=False):
        self.scale_decay: float = train_cfg.reloc_scale_decay
        self.ops_decay: float = train_cfg.reloc_ops_decay
        self.naive_decay: bool = naive_decay
        
        self.reloc_start_iter: int = train_cfg.reloc_start_iter
        self.reloc_stop_iter: int = train_cfg.reloc_stop_iter
        self.reloc_every: int = train_cfg.reloc_every

        self.densify_start_iter: int = train_cfg.grow_start_iter
        self.densify_every: int = train_cfg.grow_every
        self.growing_rate: float = train_cfg.grow_rate
        self.cap_max: int = max_cap

        # self.noise_all: bool = train_cfg.perturb_all
        # self.noise_intensity: float = train_cfg.perturb_intensity
        # self.noise_start_iter: int = train_cfg.perturb_start_iter

        self.min_opacity: float = train_cfg.reloc_dead_thres
        self.verbose: bool = verbose

    def initialize_state(self) -> Dict[str, Any]:
        return {}

    def step_pre_backward(self, state, info):
        pass

    def step_post_backward(
        self,
        state: Dict[str, Any],
        aks_params: Dict[str, torch.nn.Parameter],
        aks_opts: Dict[str, torch.optim.Optimizer],
        step: int,
        dead_func,
        impact_func,
    ):
        report = {}

        # index tracking
        dead_idx = None
        appended_idx = None
        idx_mapping = torch.arange(aks_params["anchor_offsets"].shape[0], device=aks_params["anchor_offsets"].device)

        # ---------------------------------- relocate gs -------------------------------- #
        if (step < self.reloc_stop_iter
            and step > self.reloc_start_iter
            and (step - self.reloc_start_iter) % self.reloc_every == 0):
            n_reloacted_aks, dead_idx, target_idx = self._reloate_anchor(
                state=state,
                aks_params=aks_params,
                opts=aks_opts,
                dead_func=dead_func,
                impact_func=impact_func
            )
            if n_reloacted_aks > 0:
                idx_mapping[dead_idx] = target_idx
                if self.verbose:
                    print(f"Step {step}: Relocated {n_reloacted_aks} Anchors.")

            report["relocated"] = n_reloacted_aks if n_reloacted_aks > 0 else None
            report["relocated_idx"] = dead_idx if n_reloacted_aks > 0 else None

        # ---------------------------------- grow gs --------------------------------- #
        if (step > self.densify_start_iter
            and (step - self.densify_start_iter) % self.densify_every == 0):
            n_new_aks, appended_idx, growed_idx = self._grow_anchor(
                state=state,
                aks_params=aks_params,
                opts=aks_opts,
                impact_func=impact_func
            )
            if n_new_aks > 0:
                idx_mapping = torch.cat([idx_mapping, growed_idx])
                if self.verbose:
                    print(
                        f"Step {step}: Added {n_new_aks} Anchors. "
                        f"Now having {aks_params['anchor_offsets'].shape[0]} Anchors."
                    )

            report["grew"] = n_new_aks if n_new_aks > 0 else None
            report["grew_idx"] = appended_idx if n_new_aks > 0 else None

        # reset state after any relocate or grow
        if len(report) > 0:
            # torch.cuda.empty_cache()
            for k in state:
                state[k] = None

        return report

    @torch.no_grad()
    def _reloate_anchor(
        self,
        state: Dict[str, Any],
        aks_params: Dict[str, torch.nn.Parameter],
        opts: Dict[str, torch.optim.Optimizer],
        dead_func,
        impact_func,
    ):
        aks_impacts = impact_func(state)
        dead_mask = dead_func(state)

        N = aks_params["anchor_offsets"].shape[0]
        assert aks_impacts.shape[0] == N, "shape mismatch"
        n = dead_mask.sum().item()
        if n == 0: return 0, None, None

        # use neural gs's opacity to decide dead/alive anchor
        dead_idx = dead_mask.nonzero(as_tuple=True)[0]
        alive_idx = (~dead_mask).nonzero(as_tuple=True)[0]
        probs = aks_impacts[alive_idx]
        if probs.sum() == 0 or probs.isnan().any():
            print(state["anchor_opacity"])
            print(f"number of nan in opacity: {state['anchor_opacity'].isnan().sum().item()}")
            print(state["anchor_blame"])
            print(f"number of nan in blame: {state['anchor_blame'].isnan().sum().item()}")
            assert False, "NaN in probs"
        else:
            print(f"""stats of probs:
    max: {probs.max().item()}
    min: {probs.min().item()}
    mean: {probs.mean().item()}
    std: {probs.std().item()}
    median: {probs.median().item()}""")

        sampled_idx = torch.multinomial(probs, n, replacement=True)
        sampled_idx = alive_idx[sampled_idx]
        new_opacity_decay = update_decay_after_split(
            decays = aks_params["anchor_opacity_decay"][sampled_idx],
            counts = torch.bincount(sampled_idx)[sampled_idx] + 1,
            naive=self.naive_decay,
            reloc_ops_decay=self.ops_decay
        )
        new_anchor_scale_extend = aks_params["anchor_scale_extend"][sampled_idx] + math.log(self.scale_decay)
        
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
        impact_func,
    ):
        N = aks_params["anchor_offsets"].shape[0]
        ratio = 1 + self.growing_rate
        N_target = min(self.cap_max, int(ratio * N))
        n = max(0, N_target - N)
        if n == 0: return 0, None, None

        probs = impact_func(state)
        if probs.sum() == 0 or probs.isnan().any():
            assert False, "NaN in probs"

        sampled_idx = torch.multinomial(probs, n, replacement=True)
        new_opacity_decay = update_decay_after_split(
            decays = aks_params["anchor_opacity_decay"][sampled_idx],
            counts = torch.bincount(sampled_idx)[sampled_idx] + 1,
            naive=self.naive_decay,
            reloc_ops_decay=self.ops_decay
        )
        # new_anchor_scale_extend = aks_params["anchor_scale_extend"][sampled_idx] * self.scale_decay
        
        def param_update(name: str, p: torch.Tensor) -> torch.Tensor:
            if name == "anchor_opacity_decay":
                p[sampled_idx] = new_opacity_decay
            # if name == "anchor_scale_extend":
            #     p[sampled_idx] = new_anchor_scale_extend
            p = torch.cat([p, p[sampled_idx]])
            return torch.nn.Parameter(p)
        def opt_update(key: str, v: torch.Tensor) -> torch.Tensor:
            v_new = torch.zeros((len(sampled_idx), *v.shape[1:]), device=v.device)
            return torch.cat([v, v_new])
        
        _update_param_with_optimizer(param_update, opt_update, aks_params, opts)
        appended_idx = torch.arange(N, N_target, device=sampled_idx.device)

        return n, appended_idx, sampled_idx