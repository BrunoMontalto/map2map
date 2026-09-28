from .args import get_args
from . import train
from . import train_no_dist
from . import test
from . import estimate_gpu_mem

import os, glob


def main():

    args = get_args()


    if args.mode == 'train':
        if args.load_state=="!last":
            files = glob.glob(args.states_folder + "/state_*.pt")
            if not files:
                args.load_state = None
            else:
                args.load_state = max(files, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    
        node = os.environ.get("SLURMD_NODENAME", "unknown_node")
        rank = os.environ.get("SLURM_PROCID", "unknown_rank")
        print(f"[Node: {node} | Rank: {rank}] load_state = {args.load_state}", flush=True)
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
