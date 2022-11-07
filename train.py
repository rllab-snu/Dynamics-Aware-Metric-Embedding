from agent import DAME
import torch

def main():
    import argparse

    # Set beta=1, beta_global=0.1 for Cartpole and Reacher
    # Set beta=3e-3, beta_global=3e-4 for Cup
    # Try latent overshooting presented in 'Learning Latent Dynamics for Planning from Pixels'

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole_swingup')
    parser.add_argument('--state_dim', type=int, default=30)
    parser.add_argument('--metric_dim', type=int, default=8)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--beta_global', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--free_nat', type=float, default=3)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--traj_length', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--update_per_epoch', type=int, default=10000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = DAME(args.env, args.state_dim, args.metric_dim, args.beta, args.beta_global, args.gamma, args.free_nat, device)

    agent.train(args.batch_size, args.traj_length, args.epochs, args.update_per_epoch)

if __name__ == '__main__':
    main()