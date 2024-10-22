import torch
import math
from omegaconf import ListConfig


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