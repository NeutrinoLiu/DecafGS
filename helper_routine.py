import math

class RoutineMgr:
    def __init__(self,
                 first_frame_iters,
                 per_frame_means_iters,
                 per_frame_iters,
                 runner):
        self.first_frame_iters = first_frame_iters
        self.per_frame_means_iters = per_frame_means_iters
        self.per_frame_iters = per_frame_iters
        self.runner = runner
        # runtime
        self.step = None
        self.unlocked = None
        self.full_frames = False

    def checkin(self, step):
        self.step = step
        if step <= self.first_frame_iters:
            unlocked = 0
        else:
            unlocked = math.ceil((step - self.first_frame_iters) / self.per_frame_iters)
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
        elif (self.step - self.first_frame_iters) % self.per_frame_iters <= self.per_frame_means_iters:
            return True
        return False
