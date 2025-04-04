import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device
import matplotlib.pyplot as plt


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        action_noise,
        sample_type,
        lr,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

    def init_actions(self, obs_0, actions=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
        """
        n_evals = obs_0["visual"].shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                new_actions = torch.randn(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return actions

    def get_action_optimizer(self, actions):
        return torch.optim.SGD([actions], lr=self.lr)

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """

        obs_0["visual"] = rearrange(obs_0["visual"], "b t h w c -> b t c h w")
        obs_g["visual"] = rearrange(obs_g["visual"], "b t h w c -> b t c h w")

        trans_obs_0 = move_to_device(
           { key: torch.tensor(value) for key, value in obs_0.items() }, self.device
        )
        trans_obs_g = move_to_device(
            { key: torch.tensor(value) for key, value in obs_g.items() }, self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        z_obs_g_detached = {key: value.detach() for key, value in z_obs_g.items()}

        actions = self.init_actions(obs_0, actions).to(self.device)
        actions.requires_grad = True
        optimizer = self.get_action_optimizer(actions)
        n_evals = actions.shape[0]

        for i in range(self.opt_steps):
            optimizer.zero_grad()
            i_z_obses, i_zs = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
            loss = self.objective_fn(i_z_obses, z_obs_g_detached)  # (n_evals, )
            total_loss = loss.mean() * n_evals  # loss for each eval is independent
            total_loss.backward()
            with torch.no_grad():
                actions_new = actions - optimizer.param_groups[0]["lr"] * actions.grad
                actions_new += (
                    torch.randn_like(actions_new) * self.action_noise
                )  # Add Gaussian noise
                actions.copy_(actions_new)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": total_loss.item(), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        with torch.no_grad():

            z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )

            obs_rollout_final, _ = self.wm.decode_obs(z_obses)
            print(obs_rollout_final["visual"].shape)
            print(obs_rollout_final["visual"].min(), obs_rollout_final["visual"].max())
            obs_rollout_final = obs_rollout_final["visual"] * 0.5 + 0.5
            print(obs_rollout_final.min(), obs_rollout_final.max())
            obs_rollout_final = rearrange(obs_rollout_final, "b t c h w -> b t h w c")
            print(obs_rollout_final.shape)
            obs_rollout_final = obs_rollout_final[:,-1]

            print(trans_obs_g["visual"].shape)
            print(trans_obs_g["visual"].min(), trans_obs_g["visual"].max())
            obs_g_raw = trans_obs_g["visual"] * 0.5 + 0.5
            print(obs_g_raw.min(), obs_g_raw.max())
            obs_g_raw = rearrange(obs_g_raw, "b t c h w -> b t h w c")
            print(obs_g_raw.shape)

            obs_g_enc_dec, _ = self.wm.decode_obs(z_obs_g)
            print(obs_g_enc_dec["visual"].shape)
            print(obs_g_enc_dec["visual"].min(), obs_g_enc_dec["visual"].max())
            obs_g_enc_dec = obs_g_enc_dec["visual"] * 0.5 + 0.5
            print(obs_g_enc_dec.min(), obs_g_enc_dec.max())
            obs_g_enc_dec = rearrange(obs_g_enc_dec, "b t c h w -> b t h w c")
            print(obs_g_enc_dec.shape)

            obs_0_raw = trans_obs_0
            print(obs_0_raw["visual"].shape)
            print(obs_0_raw["visual"].min(), obs_0_raw["visual"].max())
            obs_0_raw = obs_0_raw["visual"] * 0.5 + 0.5
            print(obs_0_raw.min(), obs_0_raw.max())
            obs_0_raw = rearrange(obs_0_raw, "b t c h w -> b t h w c")
            print(obs_0_raw.shape)

            for i in range(n_evals):
                plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(obs_0_raw[i].squeeze().cpu())
                plt.xlabel("Initial")
                plt.subplot(2,2,2)
                plt.imshow(obs_g_enc_dec[i].squeeze().cpu())
                plt.xlabel("Goal")
                plt.subplot(2,2,3)
                plt.imshow(obs_g_enc_dec[i].squeeze().cpu())
                plt.xlabel("Goal encoded and decoded")
                plt.subplot(2,2,4)
                plt.imshow(obs_rollout_final[i].squeeze().cpu())
                plt.xlabel("End of WM rollout with predicted actions")
                plt.show()
        
        return actions, np.full(n_evals, np.inf)  # all actions are valid
