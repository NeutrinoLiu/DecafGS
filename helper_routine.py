import torch

class RoutineMgr:
    def __init__(self, model, raw_frames):
        self.model = model
        self.raw_frames = raw_frames
        self.n_frames = len(raw_frames)
        self.states ={}

    @property
    def freezed(self):
        return self.states.get("freeze", False)
    
    def frames(self, idx):
        if isinstance(idx, int):
            return self.raw_frames[idx]
        elif isinstance(idx, list):
            return [self.raw_frames[i] for i in idx]
        else:
            raise ValueError(f"idx should be int or list of int, got {type(idx)}")

    def frame_by_frame_routine(
        self,
        init_phase,
        freeze_phase,
        mixing_phase
    ):
        self.states["unlocked"] = 0
        self.states["freeze"] = False

        def unlock_one_frame():
            self.states["unlocked"] += 1
            if self.states["unlocked"] > 0 and self.states["unlocked"] < self.n_frames:
                last_frame = self.frames(self.states["unlocked"] - 1)
                new_frame = self.frames(self.states["unlocked"])
                print(f">>>>> unlock training frame {new_frame}")
                with torch.no_grad():
                    self.model.deform.deform_params["frame_embed"][new_frame].data += \
                        self.model.deform.deform_params["frame_embed"][last_frame].data
                        
        def freeze_scaffold_and_anchor_deform():
            self.states["freeze"] = True
            unlock_one_frame()
            for p in self.model.scaffold.parameters():
                p.requires_grad = False
            for p in self.model.deform.anchor_params.values():
                p.requires_grad = False
            for p in self.model.deform.deform_params["mlp_deform"]:
                p.requires_grad = False

        def unfreeze_scaffold_and_anchor_deform():
            self.states["freeze"] = False
            for p in self.model.scaffold.parameters():
                p.requires_grad = True
            for p in self.model.deform.anchor_params.values():
                p.requires_grad = True
            for p in self.model.deform.deform_params["mlp_deform"]:
                p.requires_grad = True
        
        routine = {}
        total_interval = freeze_phase + mixing_phase

        for i in range(self.n_frames - 1):
            if freeze_phase > 0:        # iters reserved for train a single frame
                f_range = list(range(i + 1))
                routine[init_phase + i * total_interval] = (f_range, unfreeze_scaffold_and_anchor_deform)
                routine[init_phase + i * total_interval + mixing_phase] = ([i+1], freeze_scaffold_and_anchor_deform)
            else:                           # no such reserved iters
                f_range = list(range(i + 1))
                routine[init_phase + i * total_interval] = (f_range, unlock_one_frame)

        # add first iter 
        routine[0] = ([0], unfreeze_scaffold_and_anchor_deform)
        # drop the last iter
        routine.pop(max(routine.keys()))

        return routine

    def fence_by_fence_routine(
        self,
        fence_interval,
        iters_shift,
        iters_per_fence
    ):
        init_fences = list(range(0, self.n_frames, fence_interval))
        if self.n_frames - 1 not in init_fences:
            init_fences.append(self.n_frames - 1)
        routine = {
            0: (self.frames(init_fences), 
                lambda: print(f">>>>> unlock {len(init_fences)} training frame"))
            }
        unlocked = init_fences.copy()
        next_iter = iters_shift

        while len(unlocked) < self.n_frames:
            new_unlocked = set()
            for fence in unlocked:
                lf = fence - 1
                rf = fence + 1
                if lf >= 0 and lf not in unlocked:
                    new_unlocked.add(lf)
                if rf < self.n_frames and rf not in unlocked:
                    new_unlocked.add(rf)

            new_unlocked = list(new_unlocked)
            next_iter += iters_per_fence
            routine[next_iter] = (
                self.frames(unlocked + new_unlocked),
                self.gen_frame_init_fn(new_unlocked, unlocked)
            )
            unlocked += new_unlocked
        return routine

    def gen_frame_init_fn(self, new_unlocked, unlocked):
        new_frames = new_unlocked.copy()
        old_frames = unlocked.copy()
        def ret_fn():
            for f in new_frames:
                if f-1 in old_frames:
                    last_frame = self.frames(f-1)
                    new_frame = self.frames(f)
                    with torch.no_grad():
                        self.model.deform.deform_params["frame_embed"][last_frame].data += \
                            self.model.deform.deform_params["frame_embed"][new_frame].data
                elif f+1 in old_frames:
                    last_frame = self.frames(f+1)
                    new_frame = self.frames(f)
                    with torch.no_grad():
                        self.model.deform.deform_params["frame_embed"][last_frame].data += \
                            self.model.deform.deform_params["frame_embed"][new_frame].data
                else:
                    raise ValueError(f"frame {f} is has no neighbor as unlocked frames")
            print(f">>>>> {len(old_frames)} + newly unlocked: {new_frames}")
        return ret_fn
            
        