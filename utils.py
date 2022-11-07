import numpy as np
import pickle
import torch
from dm_control import suite
import wrappers

class DataLoader:
    def __init__(self, data_pth):
        with open(data_pth, 'rb') as f:
            self.data = pickle.load(f)
        self.n_data = len(self.data['obs'])
        self.act_dim = self.data['act'].shape[-1]

    def sample(self, size, length):
        batch = {
            'obs': np.zeros([length, size, 64, 64, 3], dtype=np.float32),
            'act': np.zeros([length, size, self.act_dim], dtype=np.float32)
        }
        n_sampled = 0
        while n_sampled < size:
            idx = np.random.randint(self.n_data - length)
            if (self.data['done'][idx:idx + length - 1] == 0).all():
                batch['obs'][:, n_sampled, :, :, :] = self.data['obs'][idx:idx + length]
                batch['act'][:, n_sampled, :] = self.data['act'][idx:idx + length]
                n_sampled += 1
        return batch

def preprocess_obs(obs, bit_depth=5):
    obs = torch.floor(obs / 2 ** (8 - bit_depth))
    obs = obs / 2 ** bit_depth - 0.5
    obs = obs.transpose(3, 4).transpose(2, 3)
    return obs

def EnvWrapper(name, dtype='uint'):
    if name == 'cartpole_swingup':
        action_repeat = 8
        max_length = 1000 // action_repeat
        state_components = ['position', 'velocity']
        domain = 'cartpole'
        task = 'swingup'
        camera_id = 0
    elif name == 'finger_spin':
        action_repeat = 2
        max_length = 1000 // action_repeat
        state_components = ['position', 'velocity', 'touch']
        domain = 'finger'
        task = 'spin'
        camera_id = 0
    elif name == 'cup_catch':
        action_repeat = 4
        max_length = 1000 // action_repeat
        state_components =['position', 'velocity']
        domain = 'ball_in_cup'
        task = 'catch'
        camera_id = 0
    elif name == 'reacher_easy':
        action_repeat = 4
        max_length = 1000 // action_repeat
        state_components = ['position', 'velocity', 'to_target']
        domain = 'reacher'
        task = 'easy'
        camera_id = 0
    elif name == 'cheetah_run':
        action_repeat = 4
        max_length = 1000 // action_repeat
        state_components = ['position', 'velocity']
        domain = 'cheetah'
        task = 'run'
        camera_id = 0
    elif name == 'walker_walk':
        action_repeat = 2
        max_length = 1000 // action_repeat
        state_components = ['height', 'orientations', 'velocity']
        domain = 'walker'
        task = 'walk'
        camera_id = 0

    env = suite.load(domain, task)
    env = wrappers.DeepMindWrapper(env, (64, 64), camera_id=0)
    env = wrappers.ActionRepeat(env, action_repeat)
    env = wrappers.NormalizeActions(env)
    env = wrappers.MaximumDuration(env, max_length)
    env = wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
    # env = wrappers.ConvertTo32Bit(env)
    env = wrappers.MiniWrapper(env, state_components, _dtype=dtype)
    return env, state_components

def state_dict_to_vec(state, state_components):
    state_vec = []
    for k in state_components:
        state_vec.append(state[k])
    return np.concatenate(state_vec)
