from networks import *
from utils import *

import torch
from torch.optim import Adam
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import os
import progressbar

class DAME:
    def __init__(self, env_name, state_dim, metric_dim, beta, beta_global, gamma, free_nat, device):
        self.env_name = env_name
        self.env, _ = EnvWrapper(env_name)
        act_dim = self.env.action_space.shape[0]

        self.data_loader = DataLoader(os.path.join('data', env_name + '.pkl'))

        self.encoder = Encoder().to(device)
        self.decoder = Decoder(state_dim).to(device)
        self.transition = Transition(state_dim, act_dim).to(device)
        self.posterior = Posterior(state_dim).to(device)
        self.discriminator = Discriminator(state_dim).to(device)
        self.metric = Metric(state_dim, metric_dim).to(device)

        self.main_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) \
                           + list(self.transition.parameters()) + list(self.posterior.parameters()) \
                           + list(self.metric.parameters())
        self.disc_params = list(self.discriminator.parameters())
        self.main_opt = Adam(self.main_params, lr=1e-3, eps=1e-4)
        self.disc_opt = Adam(self.disc_params, lr=1e-3, eps=1e-4)
        self.state_dim, self.act_dim = state_dim, act_dim
        self.beta, self.beta_global = beta, beta_global
        self.gamma = gamma
        self.free_nat = free_nat

        if not os.path.exists('logs/'):
            os.mkdir('logs/')
        self.writer = SummaryWriter(os.path.join('logs', env_name))
        if not os.path.exists('models/'):
            os.mkdir('models/')
        self.model_dir = os.path.join('models', env_name)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.device = device

    def update_disc(self, obs, act):
        # Train discriminator to estimate gamma-discounted state occupancy distribution, instead
        # of K-step reachability. See N. Eysenbach et.al., 'Replacing Rewards with Examples:
        # Example-Based Policy Search via Recursive Classification', NeurIPS 2021 for more details.
        L, B = obs.size(0), obs.size(1)
        state = torch.zeros([B, self.state_dim], device=self.device)
        with torch.no_grad():
            embedded_obs = self.encoder(obs.reshape(-1, 3, 64, 64)).reshape(L, B, 1024)
            for l in range(L - 1):
                prior = self.transition(state, act[l])
                posterior = self.posterior(prior.mean, prior.stddev, embedded_obs[l])
                state = posterior.sample()
            next_prior = self.transition(state, act[l])
            next_posterior = self.posterior(next_prior.mean, next_prior.stddev, embedded_obs[-1])
            next_state = next_posterior.sample()
            goal_state = next_posterior.sample()
            goal_state = torch.cat([goal_state[1:], goal_state[:1]], dim=0)

            y_next = self.discriminator(next_state, goal_state)
            weight = y_next / (1. - y_next).clamp(min=1 - self.gamma)
        y = self.discriminator(state, goal_state)
        y_goal = self.discriminator(torch.cat([next_state[1:], next_state[:1]], dim=0), goal_state)
        disc_loss_1 = -(1. - self.gamma) * torch.log(y_goal)
        disc_loss_2 = -self.gamma * weight * torch.log(y)
        disc_loss_3 = -torch.log(1. - y)
        disc_loss = (disc_loss_1 + disc_loss_2 + disc_loss_3).mean()

        self.disc_opt.zero_grad()
        disc_loss.backward()
        self.disc_opt.step()

        return disc_loss

    def update_main(self, obs, act):
        L, B = obs.size(0), obs.size(1)
        states = torch.zeros([L, B, self.state_dim], device=self.device)
        state = torch.zeros([B, self.state_dim], device=self.device)
        global_prior = Normal(torch.zeros_like(state), torch.ones_like(state))
        embedded_obs = self.encoder(obs.reshape(-1, 3, 64, 64)).reshape(L, B, 1024)
        kl_loss = 0
        for l in range(L):
            prior = self.transition(state, act[l])
            posterior = self.posterior(prior.mean, prior.stddev, embedded_obs[l])
            state = posterior.rsample()
            states[l] = state
            kl = kl_divergence(prior, posterior).sum(-1)
            kl_global = kl_divergence(posterior, global_prior).sum(-1)
            kl_loss += self.beta * kl.clamp(min=self.free_nat).mean() \
                       + self.beta_global * kl_global.clamp(min=self.free_nat).mean()
        kl_loss = kl_loss / L
        obs_hat = self.decoder(states.reshape(-1, self.state_dim)).reshape(L * B, 64 * 64 * 3)
        obs_target = obs.reshape(L * B, 64 * 64 * 3)
        recons_loss = (obs_hat - obs_target).pow(2).sum(-1).mean()

        goal_state = states[-1, :1].repeat(L * B, 1)
        w = self.metric(states.reshape(-1, self.state_dim))
        w_goal = self.metric(goal_state)
        with torch.no_grad():
            y = self.discriminator(states.reshape(-1, self.state_dim), goal_state)
            rho = y / (1. - y).clamp(min=1 - self.gamma)
            rho = rho / rho.sum()
        logits = torch.norm(w - w_goal, dim=-1, keepdim=True).pow(2)
        Q = 1. / (1. + logits)
        Q = Q / Q.sum()
        metric_loss = (-rho * torch.log(Q.clamp(min=1e-5))).sum()

        loss = recons_loss + kl_loss + metric_loss

        self.main_opt.zero_grad()
        loss.backward()
        clip_grad_norm_(self.main_params, 10.)
        self.main_opt.step()

        return loss, recons_loss, kl_loss, metric_loss

    def train(self, batch_size=20, traj_length=20, epochs=500, update_per_epoch=10000):
        n_update = 0
        for epoch in range(epochs):
            bar = progressbar.ProgressBar()
            for _ in bar(range(update_per_epoch)):
                batch = self.data_loader.sample(batch_size, traj_length)
                obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=self.device)
                obs = preprocess_obs(obs)
                act = torch.as_tensor(batch['act'], dtype=torch.float32, device=self.device)
                disc_loss = self.update_disc(obs, act)
                loss, recons_loss, kl_loss, metric_loss = self.update_main(obs, act)

                self.writer.add_scalar('loss', loss, n_update)
                self.writer.add_scalar('recons_loss', recons_loss, n_update)
                self.writer.add_scalar('kl_loss', kl_loss, n_update)
                self.writer.add_scalar('metric_loss', metric_loss, n_update)
                self.writer.add_scalar('disc_loss', disc_loss, n_update)
                n_update += 1
            predicted_obs = self.predict(batch['obs'][:4, :1], batch['act'][:4, :1], batch['act'][4:, :1])
            result_img = make_grid(torch.cat([obs[:, 0] + 0.5, predicted_obs], dim=0), nrow=traj_length)
            self.writer.add_image('prediction', result_img, n_update)
            torch.save(self.encoder.state_dict, os.path.join(self.model_dir, 'encoder.pt'))
            torch.save(self.decoder.state_dict, os.path.join(self.model_dir, 'decoder.pt'))
            torch.save(self.transition.state_dict, os.path.join(self.model_dir, 'transition.pt'))
            torch.save(self.discriminator.state_dict, os.path.join(self.model_dir, 'discriminator.pt'))
            torch.save(self.posterior.state_dict, os.path.join(self.model_dir, 'posterior.pt'))
            torch.save(self.metric.state_dict, os.path.join(self.model_dir, 'metric.pt'))

    def get_action(self, prev_obs, prev_act, goal_obs, goal_act, n_iter, n_sample, horizon, n_elite):
        prev_obs = torch.as_tensor(prev_obs, dtype=torch.float32, device=self.device).unsqueeze(1)
        prev_obs = preprocess_obs(prev_obs)
        goal_obs = torch.as_tensor(goal_obs, dtype=torch.float32, device=self.device).unsqueeze(1)
        goal_obs = preprocess_obs(goal_obs)
        prev_act = torch.as_tensor(prev_act, dtype=torch.float32, device=self.device).unsqueeze(1)
        goal_act = torch.as_tensor(goal_act, dtype=torch.float32, device=self.device).unsqueeze(1)
        state = torch.zeros([1, self.state_dim], device=self.device)
        with torch.no_grad():
            embedded_goal_obs = self.encoder(goal_obs.reshape(-1, 3, 64, 64)).reshape(-1, 1, 1024)
            for l in range(len(goal_obs)):
                prior = self.transition(state, goal_act[l])
                posterior = self.posterior(prior.mean, prior.stddev, embedded_goal_obs[l])
                goal_state = posterior.mean
            embedded_obs = self.encoder(prev_obs.reshape(-1, 3, 64, 64)).reshape(-1, 1, 1024)
            for l in range(len(prev_obs)):
                prior = self.transition(state, prev_act[l])
                posterior = self.posterior(prior.mean, prior.stddev, embedded_obs[l])
                state = posterior.mean
            act = torch.randn([horizon, n_sample, self.act_dim], device=self.device)
            init_state = state.repeat(n_sample, 1)
            w_goal = self.metric(goal_state)
            for _ in range(n_iter):
                state = init_state
                J = torch.zeros([n_sample], device=self.device)
                for t in range(horizon):
                    state = self.transition(state, torch.tanh(act[t])).mean
                    w = self.metric(state)
                    J += torch.norm(w - w_goal, dim=-1)
                elite_act = act[:, torch.argsort(J)[:n_elite]]
                mean_act = torch.mean(elite_act, dim=1, keepdim=True).repeat(1, n_sample, 1)
                std_act = torch.std(elite_act, dim=1, keepdim=True).repeat(1, n_sample, 1)
                act = Normal(mean_act, std_act).sample()
        return torch.tanh(mean_act[0].squeeze()).cpu().detach().numpy()

    def predict(self, prev_obs, prev_act, future_act):
        prev_obs = torch.as_tensor(prev_obs, dtype=torch.float32, device=self.device)
        prev_obs = preprocess_obs(prev_obs)
        prev_act = torch.as_tensor(prev_act, dtype=torch.float32, device=self.device)
        future_act = torch.as_tensor(future_act, dtype=torch.float32, device=self.device)
        state = torch.zeros([1, self.state_dim], device=self.device)
        obs_hats = torch.zeros([len(prev_obs) + len(future_act), 3, 64, 64], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            embedded_obs = self.encoder(prev_obs.reshape(-1, 3, 64, 64)).reshape(-1, 1, 1024)
            for l in range(len(prev_obs)):
                prior = self.transition(state, prev_act[l])
                state = self.posterior(prior.mean, prior.stddev, embedded_obs[l]).mean
                obs_hat = self.decoder(state)
                obs_hats[l] = obs_hat.squeeze() + 0.5
            for t in range(len(future_act)):
                state = self.transition(state, future_act[l]).mean
                obs_hat = self.decoder(state)
                obs_hats[len(prev_obs) + t] = obs_hat.squeeze() + 0.5
        return obs_hats
