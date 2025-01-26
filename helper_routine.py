import math

class RoutineMgrNull:
    def __init__(self):
        pass
    def checkin(self, step):
        pass
    def means_opt_only(self):
        return False

class RoutineMgrDensify:
    """
    first stage train new frame only
    second stage train all frames
    both are no freezing
    """
    def __init__(self,
                 first_frame_iters,
                 stage_1_iters,
                 stage_2_iters,
                 runner):
        self.first_frame_iters = first_frame_iters
        self.stage_total_iters = stage_1_iters + stage_2_iters
        self.stage_1_iters = stage_1_iters
        self.stage_2_iters = stage_2_iters
        self.runner = runner
        # runtime
        self.step = None
        self.unlocked = None
        self.full_frames = False

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

    def means_opt_only(self):
        return False

class RoutineMgrIncremental:
    """
    first stage train position only
    second stage train all attributes
    both trains on current full unlocked frames
    """
    def __init__(self,
                 first_frame_iters,
                 stage_1_iters,
                 stage_2_iters,
                 runner):
        self.first_frame_iters = first_frame_iters
        self.stage_total_iters = stage_1_iters + stage_2_iters
        self.stage_1_iters = stage_1_iters
        self.stage_2_iters = stage_2_iters
        self.runner = runner
        # runtime
        self.step = None
        self.unlocked = None
        self.full_frames = False

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
    
    def means_opt_only(self):
        if self.full_frames:
            return False
        if self.unlocked == 0:
            return False
        elif (self.step - self.first_frame_iters) % self.stage_total_iters <= self.stage_1_iters:
            return True
        return False

class RoutineMgrFence:
    """
    first stage train position only
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
                 std_grow=False):
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
        return after_extended, new_frames
    def copy_embeds(self, befores, news):
        for f in news:
            template = f - 1
            if template not in befores:
                template = f + 1
            assert template in befores, f"RoutineMgr: template frame {template} for new frame {f} not inited"
            self.runner.model.deform.copy_frame_embed(template, f)

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
            self.copy_embeds(
                after_extended - new_frames, new_frames)
            if self.std_grow:
                total_lengths = len(self.runner.all_frames)
                self.runner.model.deform.increase_all_std(1/total_lengths)
            print(f"RoutineMgr: flush data loaders. current training frames: {sorted(list(after_extended))}")
            self.runner.setup_loader(list(after_extended))
            
    def means_opt_only(self):
        if self.full_frames or self.step < self.first_frame_iters:
            return False
        elif (self.step - self.first_frame_iters) % self.stage_total_iters <= self.stage_1_iters:
            return True
        return False