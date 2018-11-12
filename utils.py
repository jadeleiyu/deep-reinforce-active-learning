import argparse
from collections import namedtuple

Config = namedtuple('parameters',
                    ['state_dim', 'input_dim', 'hidden', 'output_dim', 'epsilon'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--homepath', type=str, default='/home/ml/lyu40/')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_set_size', type=int, default=10000)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--episode_length', type=int, default=50)
    parser.add_argument('--episode_number', type=int, default=300)
    parser.add_argument('--capacity', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--reward_amplify', type=float, default=1)
    parser.add_argument('--passive_drive', action='store_true', help='enable a passive learner as reward baseline')

    return parser.parse_args()
