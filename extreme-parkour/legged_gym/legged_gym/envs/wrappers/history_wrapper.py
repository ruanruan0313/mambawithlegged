import isaacgym
assert isaacgym
import torch
import gym

#
# 获取历史scandots
class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.obs_history_length = self.env.cfg.env.num_history_len
        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.obs_history_length, self.num_obs, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        # 4维的历史扫描点
        self.scandots_history = torch.zeros(self.env.num_envs, self.obs_history_length, self.num_height_points, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs

    def step(self, action):
        # privileged information and observation history are stored in info
        obs, scan, privileged_obs, rew, done, info = self.env.step(action)
        # 为history_buffer添加动作延迟，减小sim2real的差距
        self.obs_history = torch.cat((self.obs_history[:, 1:], obs.unsqueeze(1)), dim=1)
        # print("obs history shape ",self.obs_history.shape)
        self.scandots_history = torch.cat((self.scandots_history[:, 1:], scan.unsqueeze(1)), dim=1)
        # print("scandots history shape ", self.scandots_history.shape)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history,
                'scandots_history': self.scandots_history}, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        scandots = self.env.get_scandots()
        self.obs_history = torch.cat((self.obs_history[:, 1:], obs.unsqueeze(1)), dim=1)
        self.scandots_history = torch.cat((self.scandots_history[:, 1:], scandots.unsqueeze(1)), dim=1)
        return {'obs': obs, 'privileged_obs': privileged_obs,
                'obs_history': self.obs_history,'scandots_history': self.scandots_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids, :, :] = 0
        self.scandots_history[env_ids, :, :] = 0
        return ret

    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :, :] = 0
        self.scandots_history[:, :, :] = 0
        return {"obs": ret[0], "privileged_obs": privileged_obs,
                "obs_history": self.obs_history, 'scandots_history': self.scandots_history}



