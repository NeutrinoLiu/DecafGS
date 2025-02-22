import math
class RoutineConsts:
    WARMUP_FREEZE = {
        "deform": [],
        "scaffold": []
    }
    APPEARANCE_FREEZE = {
        "deform": [
            "anchor_embed",
            "frame_embed",
            "mlp_deform_feature"
            "anchor_opacity_decay",
            "anchor_opacity_mean",
            "anchor_opacity_std",
            "anchor_opacity_steepness",
            ],
        "scaffold": [
            "mlp_scales",
            "mlp_quats",
            "mlp_opacities",
            "mlp_colors"
            ]
    }
    DEFORM_FREEZE = {
        "deform": [
            "anchor_xyz",
            "anchor_offsets",
            "anchor_offset_extend",
            "anchor_scale_extend",
            "anchor_quat",
            "anchor_delta_embed",
            "frame_delta_embed",
            "mlp_deform_delta"
            ],
        "scaffold": []
    }


class RoutineMgrNull:
    def __init__(self, chkpt=None):
        # subclass no need to super__init__ since it's a null class
        pass
    def checkin(self, step):
        pass
    def is_warming_up(self):
        return False
    def in_first_stage(self):
        return False
    def first_stage_lasts(self):
        return None
    def dump_chkpt(self):
        return {}
    def model_reconfig(self):
        pass

class RoutineMgrFavorNewFrame(RoutineMgrNull):
    """
    first stage train new frame only
    second stage train all frames
    both are no freezing
    """
    def __init__(self,
                 first_frame_iters,
                 stage_1_iters,
                 stage_2_iters,
                 runner,
                 chkpt=None):
        self.first_frame_iters = first_frame_iters
        self.stage_total_iters = stage_1_iters + stage_2_iters
        self.stage_1_iters = stage_1_iters
        self.stage_2_iters = stage_2_iters
        self.runner = runner
        # runtime
        self.step = None
        self.unlocked = None
        self.full_frames = False
        self.parse_chkpt(chkpt)

    def parse_chkpt(self, chkpt):
        if chkpt is not None:
            self.step = chkpt["step"]
            self.unlocked = chkpt["unlocked"]
            self.full_frames = chkpt["full_frames"]
            print(f"RoutineMgr: loaded from checkpoint. step: {self.step}, unlocked: {self.unlocked}, full_frames: {self.full_frames}")

    def dump_chkpt(self):
        return {
            "step": self.step,
            "unlocked": self.unlocked,
            "full_frames": self.full_frames
        }

    def checkin(self, step):
        if self.full_frames:
            return
        self.step = step
        if step <= self.first_frame_iters:
            unlocked = 0
        else:
            unlocked = math.ceil((step - self.first_frame_iters) / self.stage_total_iters)
        if unlocked != self.unlocked:
            if unlocked >= len(self.runner.all_frames):
                self.full_frames = True
                print(f"RoutineMgr: all frames unlocked. current frames: {self.runner.all_frames}")
                return
            self.unlocked = unlocked
            new_frame = self.runner.all_frames[unlocked]
            # unlocked_frames = self.runner.all_frames[:unlocked+1]
            unlocked_frames = [new_frame]
            print(f"RoutineMgr: unlocked #{new_frame}")
            print(f"RoutineMgr: flush data loaders. current frames: {unlocked_frames}")
            self.runner.train_loader_gen.frames = unlocked_frames
            self.runner.test_loader_gen.frames = unlocked_frames
            self.runner.train_loader = iter([])
            self.runner.test_loader = iter([])
            if unlocked > 0:
                last_frame = self.runner.all_frames[unlocked-1]
                print(f"RoutineMgr: copy frame embed from {last_frame} to {new_frame}")
                self.runner.model.deform.copy_frame_embed(last_frame, new_frame)
        else:
            if unlocked > 0 and len(self.runner.train_loader_gen.frames) == 1 and \
                (step - self.first_frame_iters) % self.stage_total_iters > self.stage_1_iters:
                unlocked_frames = self.runner.all_frames[:unlocked+1]
                print(f"RoutineMgr: flush data loaders. current frames: {unlocked_frames}")
                self.runner.train_loader_gen.frames = unlocked_frames
                self.runner.test_loader_gen.frames = unlocked_frames
                self.runner.train_loader = iter([])
                self.runner.test_loader = iter([])
    
    def is_warming_up(self):
        return self.step < self.first_frame_iters

class RoutineMgrIncremental(RoutineMgrNull):
    """
    first stage train position only
    second stage train all attributes
    both trains on current full unlocked frames
    """
    def __init__(self,
                 first_frame_iters,
                 stage_1_iters,
                 stage_2_iters,
                 runner,
                 chkpt=None):
        self.first_frame_iters = first_frame_iters
        self.stage_total_iters = stage_1_iters + stage_2_iters
        self.stage_1_iters = stage_1_iters
        self.stage_2_iters = stage_2_iters
        self.runner = runner
        # runtime
        self.step = None
        self.unlocked = None
        self.full_frames = False
        self.parse_chkpt(chkpt)
    
    def parse_chkpt(self, chkpt):
        if chkpt is not None:
            self.step = chkpt["step"]
            self.unlocked = chkpt["unlocked"]
            self.full_frames = chkpt["full_frames"]
            print(f"RoutineMgr: loaded from checkpoint. step: {self.step}, unlocked: {self.unlocked}, full_frames: {self.full_frames}")
    
    def dump_chkpt(self):
        return {
            "step": self.step,
            "unlocked": self.unlocked,
            "full_frames": self.full_frames
        }

    def checkin(self, step):
        if self.full_frames:
            return
        self.step = step
        if step <= self.first_frame_iters:
            unlocked = 0
        else:
            unlocked = math.ceil((step - self.first_frame_iters) / self.stage_total_iters)
        if unlocked != self.unlocked:
            if unlocked >= len(self.runner.all_frames):
                self.full_frames = True
                print(f"RoutineMgr: all frames unlocked. current frames: {self.runner.all_frames}")
                return
            self.unlocked = unlocked
            new_frame = self.runner.all_frames[unlocked]
            unlocked_frames = self.runner.all_frames[:unlocked+1]
            print(f"RoutineMgr: unlocked #{new_frame}")
            print(f"RoutineMgr: flush data loaders. current frames: {unlocked_frames}")
            self.runner.train_loader_gen.frames = unlocked_frames
            self.runner.test_loader_gen.frames = unlocked_frames
            self.runner.train_loader = iter([])
            self.runner.test_loader = iter([])
            if unlocked > 0:
                last_frame = self.runner.all_frames[unlocked-1]
                print(f"RoutineMgr: copy frame embed from {last_frame} to {new_frame}")
                self.runner.model.deform.copy_frame_embed(last_frame, new_frame)
    
    def in_first_stage(self):
        if self.full_frames:
            return False
        if self.unlocked == 0:
            return False
        elif (self.step - self.first_frame_iters) % self.stage_total_iters <= self.stage_1_iters:
            return True
        return False
    def first_stage_lasts(self):
        if self.step < self.first_frame_iters:
            return None
        if (self.step - self.first_frame_iters) % self.stage_total_iters > self.stage_1_iters:
            return None
        return (self.step - self.first_frame_iters) % self.stage_total_iters
    def is_warming_up(self):
        return self.step < self.first_frame_iters

    def model_reconfig(self):
        model = self.runner.model
        if self.is_warming_up():
            model.freeze_only(
                deformable_paras = RoutineConsts.WARMUP_FREEZE["deform"],
                scaffold_paras = RoutineConsts.WARMUP_FREEZE["scaffold"]
            )
        elif self.in_first_stage():
            # optimize deform in the first stage
            model.freeze_only(
                deformable_paras = RoutineConsts.APPEARANCE_FREEZE["deform"],
                scaffold_paras = RoutineConsts.APPEARANCE_FREEZE["scaffold"]
            )
            # optimize all in the second stage
        else:
            model.unfreeze()

class RoutineMgrFenceSimple(RoutineMgrNull):
    """
    train the init frames first the ndirectly unlock all frames
    """
    def __init__(self,
                 first_frame_iters,
                 runner,
                 init_frames,
                 chkpt=None):
        self.first_frame_iters = first_frame_iters
        self.init_frames = init_frames 
        self.runner = runner
        # runtime
        self.initiated = False
        self.full_frames = False
        self.step = None

    def checkin(self, step):
        self.step = step
        if not self.initiated:
            self.initiated = True
            self.runner.setup_loader(self.init_frames.copy())
            print(f"RoutineMgr: initiated. current frames: {sorted(self.init_frames)}")
            return
        if self.step > self.first_frame_iters and not self.full_frames:
            self.full_frames = True
            self.runner.setup_loader(self.runner.all_frames)
            print(f"RoutineMgr: all frames unlocked. current frames: {sorted(self.runner.all_frames)}")
            return
        return
        
    def dump_chkpt(self):
        return {}
    
    def is_warming_up(self):
        return self.step < self.first_frame_iters
    
    def model_reconfig(self):
        model = self.runner.model
        if self.is_warming_up():
            model.freeze_only(
                deformable_paras = RoutineConsts.WARMUP_FREEZE["deform"],
                scaffold_paras = RoutineConsts.WARMUP_FREEZE["scaffold"]
            )
        else:
            model.unfreeze()

class RoutineMgrFenceSimpleDecoupled(RoutineMgrFenceSimple):
    """
    dataset train as fence,
    yet feature and geometry are decoupled during different iterations
    """
    def __init__(self,
                 first_frame_iters,
                 stage_1_iters,
                 stage_2_iters,
                 runner,
                 init_frames,
                 chkpt=None):
        super().__init__(first_frame_iters, runner, init_frames, chkpt)
        self.stage_total_iters = stage_1_iters + stage_2_iters
        self.stage_1_iters = stage_1_iters
        self.stage_2_iters = stage_2_iters

    def is_first_stage(self):
        if self.step < self.first_frame_iters:
            return False
        if (self.step - self.first_frame_iters) % self.stage_total_iters <= self.stage_1_iters:
            return True
        return False
    
    def first_stage_lasts(self):
        if self.step < self.first_frame_iters:
            return None
        if (self.step - self.first_frame_iters) % self.stage_total_iters > self.stage_1_iters:
            return None
        return (self.step - self.first_frame_iters) % self.stage_total_iters

    def checkin(self, step):
        super().checkin(step)

    def model_reconfig(self):
        model = self.runner.model
        if self.is_warming_up():
            model.freeze_only(
                deformable_paras = RoutineConsts.WARMUP_FREEZE["deform"],
                scaffold_paras = RoutineConsts.WARMUP_FREEZE["scaffold"]
            )
        elif self.in_first_stage():
            # optimize deform in the first tage
            model.freeze_only(
                deformable_paras = RoutineConsts.APPEARANCE_FREEZE["deform"],
                scaffold_paras = RoutineConsts.APPEARANCE_FREEZE["scaffold"]
            )
        else:
            model.unfreeze()

class RoutineMgrFence(RoutineMgrNull):
    """
    first stage train position(i.e means) only
    second stage train all attributes
    both trains on current full unlocked frames
    yet expand frames as a fence
    """
    def __init__(self,
                 first_frame_iters,
                 stage_1_iters,
                 stage_2_iters,
                 runner,
                 init_frames,
                 std_grow=False,
                 chkpt=None):
        self.first_frame_iters = first_frame_iters
        self.stage_total_iters = stage_1_iters + stage_2_iters
        self.stage_1_iters = stage_1_iters
        self.stage_2_iters = stage_2_iters
        self.std_grow = std_grow
        self.runner = runner
        # runtime
        self.step = None
        self.full_frames = False
        self.initiated = False
        self.init_frames = init_frames
        self.parse_chkpt(chkpt)
    
    def parse_chkpt(self, chkpt):
        if chkpt is not None:
            self.step = chkpt["step"]
            self.full_frames = chkpt["full_frames"]
            self.initiated = chkpt["initiated"]
            self.init_frames = chkpt["init_frames"]
            # recover the unlocked frames
            self.runner.setup_loader(chkpt["unlocked_frames"])
    
    def dump_chkpt(self):
        return {
            "step": self.step,
            "full_frames": self.full_frames,
            "initiated": self.initiated,
            "init_frames": self.init_frames,
            # save the unlocked frames
            "unlocked_frames": self.current_frames()
        }
    
    def current_frames(self):
        return self.runner.train_loader_gen.frames
    def total_frames(self):
        return self.runner.all_frames
    def try_to_extend(self):
        after_extended = []
        for unlocked in self.current_frames():
            after_extended.append(unlocked-1)
            after_extended.append(unlocked)
            after_extended.append(unlocked+1)
        after_extended = set(after_extended) & set(self.total_frames())
        new_frames = after_extended - set(self.current_frames())
        # print(f"current frames: {sorted(self.current_frames())}")
        # print(f"after extended: {sorted(list(after_extended))}")
        # print(f"new frames: {sorted(list(new_frames))}")
        # print(f"total frames: {sorted(self.total_frames())}")
        return after_extended, new_frames
    def copy_embeds(self, befores, news):
        for f in news:
            template = f - 1
            if template not in befores:
                template = f + 1
            assert template in befores, f"RoutineMgr: template frame {template} for new frame {f} not inited"
            self.runner.model.deform.copy_frame_embed(template, f, offseted=False)

    def checkin(self, step):
        self.step = step
        if not self.initiated:
            self.initiated = True
            self.runner.setup_loader(self.init_frames.copy())
            print(f"RoutineMgr: initiated. current frames: {sorted(self.init_frames)}")
            return
        if self.full_frames or self.step < self.first_frame_iters:
            return
        elif (step - self.first_frame_iters) % self.stage_total_iters == 0:
            if len(self.current_frames()) >= len(self.runner.all_frames):
                # when frame has been fully unlocked
                self.full_frames = True
                print(f"RoutineMgr: all frames unlocked. current frames: {sorted(self.runner.all_frames)}")
                return
            after_extended, new_frames = self.try_to_extend()
            print(f"RoutineMgr: unlocked #{sorted(list(new_frames))}, init frame embeds")
            # TODO screen the copy_embeds
            self.copy_embeds(
                after_extended - new_frames, new_frames)
            if self.std_grow:
                total_lengths = len(self.runner.all_frames)
                self.runner.model.deform.increase_all_std(1/total_lengths)
            print(f"RoutineMgr: flush data loaders. current training frames: {sorted(list(after_extended))}")
            self.runner.setup_loader(list(after_extended))
            
    def in_first_stage(self):
        if self.full_frames or self.step < self.first_frame_iters:
            return False
        elif (self.step - self.first_frame_iters) % self.stage_total_iters <= self.stage_1_iters:
            return True
        return False
    
    def first_stage_lasts(self):
        if self.step < self.first_frame_iters:
            return None
        if (self.step - self.first_frame_iters) % self.stage_total_iters > self.stage_1_iters:
            return None
        return (self.step - self.first_frame_iters) % self.stage_total_iters

    def is_warming_up(self):
        return self.step < self.first_frame_iters

    def model_reconfig(self):
        model = self.runner.model
        if self.is_warming_up():
            model.freeze_only(
                deformable_paras = RoutineConsts.WARMUP_FREEZE["deform"],
                scaffold_paras = RoutineConsts.WARMUP_FREEZE["scaffold"]
            )
        elif self.in_first_stage():
            # optimize deform in the first stage
            model.freeze_only(
                deformable_paras = RoutineConsts.APPEARANCE_FREEZE["deform"],
                scaffold_paras = RoutineConsts.APPEARANCE_FREEZE["scaffold"]
            )
            # optimize all in the second stage
        else:
            model.unfreeze()