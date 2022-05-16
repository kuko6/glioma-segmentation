import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--wandb', type=str, help='wandb id')
    parser.add_argument('-a', type=str, help='a')
    args = parser.parse_args()

    if args.data_path:
        data = args.data_path
    else:
        data = os.path.join(os.getcwd(), 'BraTS2021')
    print(os.listdir(os.getcwd()))
    # parser.add_argument('--data_path', type=str, help='path to data')
    # parser.add_argument('--wandb', type=str, help='wandb id')
    # args = parser.parse_args()