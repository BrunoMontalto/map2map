from .args import get_args
from . import train
from . import train_no_dist
from . import test
from . import estimate_gpu_mem




def main():

    args = get_args()


    if args.mode == 'train':
        if args.not_distributed:
            train_no_dist.node_worker(args)
        else:
            train.node_worker(args)

    elif args.mode == 'test':
        test.test(args)
    elif args.mode == 'estimate_gpu_mem':
        estimate_gpu_mem.estimate_gpu_memory_MB(args)


if __name__ == '__main__':
    main()
