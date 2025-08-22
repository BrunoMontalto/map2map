from .args import get_args
from . import train
from . import test
import os


def main():

    args = get_args()

    if not os.path.exists(args.states_folder):
        print(f"states folder '{args.states_folder}' does not exist.")
        os.makedirs(args.states_folder)
        print(f"states folder '{args.states_folder}' created.")
    else:
        #log that the folder already exists and the states filenames it contains
        print(f"NOTE: states folder '{args.states_folder}' already exists. "
                f"\t-contains files: {os.listdir(args.states_folder)}")

    if args.mode == 'train':
        train.node_worker(args)
    elif args.mode == 'test':
        test.test(args)


if __name__ == '__main__':
    main()
