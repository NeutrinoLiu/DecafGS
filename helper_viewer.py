import torch
import nerfview
import viser
import time
from interface import Camera, Gaussians
from examples.utils import set_random_seed, rgb_to_sh
from helper import normalize

class ViewerMgr:
    def __init__(self, runner, min_frame=0, max_frame=1):
        self.runner = runner
        self.vis_modes = ["RGB", "density", "childs", "relocate", "ops", "avg-spawn", "avg-ops", "avg-grad", "avg-blame"]

        self.server = viser.ViserServer(port=8080, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self.callback,
            vis_options=self.vis_modes,
            mode="training",
            min_frame=min_frame,
            max_frame=max_frame
        )

        # runtime
        self.last_time_img = None
        self.tic = None

    def checkin(self):
        while self.viewer.state.status == "paused":
            time.sleep(0.01)
        self.viewer.lock.acquire()
        self.tic = time.time()
        return self.tic

    def checkout(self, num_rays, step):
        self.viewer.lock.release()
        num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
        num_train_rays_per_sec = num_rays * num_train_steps_per_sec
        # static update
        self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
        self.viewer.update(step, num_rays)

    @torch.no_grad()
    def callback(
        self, 
        camera_state: nerfview.CameraState,
        img_wh: "tuple[int, int]",
        frame=0,
        mode="RGB",
        init_scale=0.02
    ):
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float()
        K = torch.from_numpy(K).float()
        c2w_R = c2w[:3, :3]
        c2w_t = c2w[:3, 3]

        cam = Camera(
            path=None,
            intri=K,
            c2w_R=c2w_R,
            c2w_t=c2w_t,
            width=W,
            height=H,
            frame=frame
        )

        # --------------------------- pure image rendering --------------------------- #
        if mode == "RGB":
            pc, _ = self.runner.model.produce(cam)
        elif mode == "density":
            pc, _ = self.runner.model.produce(cam)
            white_color = torch.ones(pc.means.shape[0], 3, device=self.runner.device)
            pc._params["sh0"] = rgb_to_sh(white_color).unsqueeze(1)
        # ---------------------------- childs gs rendering --------------------------- #
        elif mode == "childs":
            gs, aks = self.runner.model.produce(cam)
            aks_means = aks.anchor_xyz
            aks_quats = torch.zeros(aks_means.shape[0], 4, device=self.runner.device)
            aks_quats[:, 0] = 1.0
            aks_scales = torch.ones(aks_means.shape[0], 3, device=self.runner.device) * init_scale*0.5
            aks_opacities = torch.ones(aks_means.shape[0], device=self.runner.device)
            aks_colors = torch.ones(aks_means.shape[0], 3, device=self.runner.device)

            child_means = aks.childs_xyz.reshape(-1, 3)
            child_quats = torch.zeros(child_means.shape[0], 4, device=self.runner.device)
            child_quats[:, 0] = 1.0
            child_scales = torch.ones(child_means.shape[0], 3, device=self.runner.device) * init_scale * 0.2
            child_opacities = torch.ones(child_means.shape[0], device=self.runner.device) * 0.3
            child_colors = torch.ones(child_means.shape[0], 3, device=self.runner.device)
            child_colors[:, 0] = 0.0
            child_colors[:, 2] = 0.0

            pc_dict = {
                "means" : torch.cat([aks_means, child_means], dim=0),
                "quats" : torch.cat([aks_quats, child_quats], dim=0),
                "scales" : torch.cat([aks_scales, child_scales], dim=0),
                "opacities" : torch.cat([aks_opacities, child_opacities], dim=0),
                "sh0": torch.cat([rgb_to_sh(aks_colors), rgb_to_sh(child_colors)], dim=0).unsqueeze(1),
                "shN": torch.zeros(aks_means.shape[0] + child_means.shape[0], 0, 3, device=self.runner.device)
            }
            pc = Gaussians(pc_dict)
        # ------------------- anchor location and opacity rendering ------------------ #
        else:
            gs, aks = self.runner.model.produce(cam)
            num_childs = aks.childs_xyz.shape[1]
            mask = None
            if mode == "ops":
                vis_ops = gs.opacities.reshape(-1, num_childs).mean(dim=1)
            elif mode == "avg-spawn":
                vis_ops = self.runner.state.get("anchor_childs", None)
                if vis_ops is None:
                    return self.last_time_img.cpu().numpy()
                N = vis_ops.shape[0]
                hc = N // 20
                mask = torch.zeros(N, dtype=torch.bool).to(self.runner.device).scatter(
                                        0, torch.topk(vis_ops, hc, largest=False).indices, True)
            elif mode == "avg-ops":
                vis_ops = self.runner.state.get("anchor_opacity", None)
                if vis_ops is None:
                    return self.last_time_img.cpu().numpy()
                N = vis_ops.shape[0]
                hc = N // 20
                mask = torch.zeros(N, dtype=torch.bool).to(self.runner.device).scatter(
                                        0, torch.topk(vis_ops, hc, largest=False).indices, True)
            elif mode == "avg-grad":
                vis_ops = self.runner.state.get("anchor_grad2d", None)
            elif mode == "avg-blame":
                vis_ops = self.runner.state.get("anchor_blame", None)
            elif mode == "relocate":
                try:
                    vis_ops = self.runner.densify_prefer_func(self.runner.state)
                    mask = self.runner.densify_dead_func(self.runner.state)
                except:
                    vis_ops = None
            if vis_ops is None:
                return self.last_time_img.cpu().numpy()
            vis_ops = normalize(vis_ops)
            quats = torch.zeros(aks.anchor_xyz.shape[0], 4, device=self.runner.device)
            quats[:, 0] = 1.0
            scales = torch.ones(aks.anchor_xyz.shape[0], 3, device=self.runner.device) * init_scale
            colors = torch.ones(aks.anchor_xyz.shape[0], 3, device=self.runner.device) 
            if mask is not None:
                colors[mask, 0] = 0.0
                colors[mask, 2] = 0.0
                vis_ops[mask] = 0.5

            center = {
                "means" : torch.tensor([[0,0,0]]).to(self.runner.device),
                "quats" : torch.tensor([[1,0,0,0]]).to(self.runner.device),
                "scales" : torch.tensor([[init_scale * 3] * 3]).to(self.runner.device),
                "opacities" : torch.tensor([1.0]).to(self.runner.device),
                "sh0": rgb_to_sh(torch.tensor([[1,0,0]])).to(self.runner.device).unsqueeze(1),
                "shN": torch.zeros(1, 0, 3).to(self.runner.device),
            }
            box = {
                "means" : torch.tensor([[-1,-1,-1], [-1, 1,-1], [ 1, 1,-1], [ 1,-1,-1],
                                        [-1,-1, 1], [-1, 1, 1], [ 1, 1, 1], [ 1,-1, 1]]).to(self.runner.device),
                "quats" : torch.tensor([[1,0,0,0]]).repeat(8, 1).to(self.runner.device),
                "scales" : torch.tensor([[init_scale * 2] * 3]).repeat(8, 1).to(self.runner.device),
                "opacities" : torch.tensor([1.0]).repeat(8).to(self.runner.device),
                "sh0": rgb_to_sh(torch.tensor([[0,0,1]])).repeat(8, 1).to(self.runner.device).unsqueeze(1),
                "shN": torch.zeros(8, 0, 3).to(self.runner.device),
            }
            pc_dict = {
                "means" : aks.anchor_xyz,
                "quats" : quats,
                "scales" : scales,
                "opacities" : vis_ops,
                "sh0": rgb_to_sh(colors).unsqueeze(1),
                "shN": torch.zeros(aks.anchor_xyz.shape[0], 0, 3, device=self.runner.device)
            }

            pc_dict = {
                "means" : torch.cat([center["means"], box["means"], pc_dict["means"]], dim=0),
                "quats" : torch.cat([center["quats"], box["quats"], pc_dict["quats"]], dim=0),
                "scales" : torch.cat([center["scales"], box["scales"], pc_dict["scales"]], dim=0),
                "opacities" : torch.cat([center["opacities"], box["opacities"], pc_dict["opacities"]], dim=0),
                "sh0": torch.cat([center["sh0"], box["sh0"], pc_dict["sh0"]], dim=0),
                "shN": torch.cat([center["shN"], box["shN"], pc_dict["shN"]], dim=0)
            }
            pc = Gaussians(pc_dict)


        img, _, _ = self.runner.single_render(
            pc=pc,
            cam=cam,
            permute=False,      # no need to permute, viewer need (H, W, 3)
            )

        self.last_time_img = img
        return img.cpu().numpy()