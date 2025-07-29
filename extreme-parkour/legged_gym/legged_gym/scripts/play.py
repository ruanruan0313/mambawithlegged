from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from legged_gym.envs.wrappers import HistoryWrapper
from rsl_rl.modules.configuration_mamba import MyMambaConfig
from rsl_rl.modules.modeling_mamba import MyMambaModel
import torch
import torch.nn as nn
def play(args):
    faulthandler.enable()

    log_pth = "/media/gh/1CEB5CCE6D1163181/rcy/extreme-parkour/legged_gym/logs/isaacgym/2/model_5000.pt"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 修改环境参数
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.,
                                    "parkour_hurdle": 0.,
                                    "parkour_flat": 0.1,
                                    "parkour_step": 0.,
                                    "parkour_gap": 0.,
                                    "demo": 0.}
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    env_cfg.domain_rand.randomize_friction = True

    # 创建环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = HistoryWrapper(env)
    obs_dict = env.reset()
    print("obs_dict : ", obs_dict)

    # --------- 定义 Actor ---------
    actor = nn.Sequential(
        nn.Linear(71, 512),  # 增加 Mamba 输出维度
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 12)
    )

    # --------- 定义 Mamba 模型 ---------
    config_mamba = MyMambaConfig(
        d_model=139,  
        n_layer=4,
        ssm_cfg={"d_state": 16},
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=True,
        norm_epsilon=1e-5
    )
    mamba_gru = MyMambaModel(config_mamba)

    # --------- 加载 Actor 权重 ---------
    checkpoint_path = os.path.join(log_pth)  # 修改为真实路径
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # actor.load_state_dict(checkpoint["model_state_dict"])
    full_state_dict = checkpoint["model_state_dict"]
    actor_state_dict = {k.replace("actor.", ""): v for k, v in full_state_dict.items() if k.startswith("actor.")}
    mamba_gru_state_dict = {k.replace("mamba_gru.", ""): v for k, v in full_state_dict.items() if k.startswith("mamba_gru.")}
    print("mamba_gru_state_dict : ", mamba_gru_state_dict.keys())
	# # 加载到你的 actor 模型
	# actor = YourActorClass(
    actor.load_state_dict(actor_state_dict)

    actor = actor.to(env.device)
    mamba_gru = mamba_gru.to(env.device)

    # --------- 初始化 ---------
    hidden_state = torch.zeros(env.num_envs, config_mamba.d_model, device=env.device)
    actions = torch.zeros(env.num_envs, 12, device=env.device)

    # --------- 仿真循环 ---------
    for step in range(10 * int(env.max_episode_length)):
        with torch.no_grad():
            obs = obs_dict["obs"]
            # print("obs : ", obs)
            obs_history = obs_dict["obs_history"]
            scan_history = obs_dict["scandots_history"]
            pose_token = obs_history[:, :, :7]
            latent_input = torch.cat((scan_history, pose_token),dim=-1)

            # 用 Mamba 更新历史 latent
            latent = mamba_gru(latent_input)
            # print("mamba_gru output : ", latent)

            # 拼接 obs + latent
            policy_input = torch.cat([obs, latent], dim=-1)

            # 输出动作
            actions = actor(policy_input)

        # 执行动作
        obs_dict, _,  _, infos = env.step(actions)

        if args.web:
            webviewer.WebViewer().render(
                fetch_results=True,
                step_graphics=True,
                render_all_camera_sensors=True,
                wait_for_page_load=True
            )

        # Debug 输出
        look_id = env.lookat_id
        print(f"[{step}] time={env.episode_length_buf[look_id].item()/50:.2f}, "
              f"cmd_vx={env.commands[look_id,0].item():.2f}, "
              f"actual_vx={env.base_lin_vel[look_id,0].item():.2f}")

if __name__ == "__main__":
    args = get_args()
    play(args)

