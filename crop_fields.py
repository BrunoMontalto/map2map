import argparse
import os
import numpy as np
from glob import glob
from multiprocessing import Pool, cpu_count
import gc
#import torch

def str_list(s):  # from args.py
    return s.split(',')


# Each worker has to know its starting 3d index and the size of its subcube
def subcube_worker(args):
    arr, start_idx, subcube_side, crop_size, output_path, base_filename, global_offset, filename_digits_per_coordinate, use_pt = args
    x_start, y_start, z_start = start_idx
    ox, oy, oz = global_offset

    # Extract the subcube
    subcube = arr[
        :,
        x_start : x_start + subcube_side,
        y_start : y_start + subcube_side,
        z_start : z_start + subcube_side,
    ]

    # Crop subcube into smaller cubes
    for i in range(0, subcube_side, crop_size):
        for j in range(0, subcube_side, crop_size):
            for k in range(0, subcube_side, crop_size):
                crop = subcube[
                    :,
                    i : i + crop_size,
                    j : j + crop_size,
                    k : k + crop_size,
                ]
                global_x = ox + x_start + i
                global_y = oy + y_start + j
                global_z = oz + z_start + k

                #out_name = (
                #    f'{base_filename}_crop_{global_x}_{global_y}_{global_z}.npy'
                #)
                
                #format the filename so that each global coordinate has fixed number of digits
                out_name = (
                    f'{base_filename}_crop_'
                    f'{global_x:0{filename_digits_per_coordinate}d}_'
                    f'{global_y:0{filename_digits_per_coordinate}d}_'
                    f'{global_z:0{filename_digits_per_coordinate}d}.npy'
                )

                if use_pt:
                    out_name = out_name.replace('.npy', '.pt')
                    crop = torch.from_numpy(crop.astype(np.float32))


                out_file = os.path.join(output_path, out_name)
                if use_pt:
                    torch.save(crop, out_file)
                else:
                    np.save(out_file, crop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-patterns', type=str_list, required=True)
    parser.add_argument('--out-folder', type=str, required=True)
    parser.add_argument('--subcube-side', type=int, required=True)
    parser.add_argument('--crop-size', type=int, required=True)
    parser.add_argument('--workers', type=int, default=cpu_count())
    parser.add_argument('--crop-before-processing', type=int, default=None)
    parser.add_argument('--crop-before-processing-offset', type=int, nargs='*', default=(0,0,0)) 

    parser.add_argument('--filename-digits-per-coordinate', type=int, default=3)

    # Add option to use .pt format (torch tensor) instead of .npy
    parser.add_argument('--use-pt', action='store_true')

    args = parser.parse_args()

    print('in-patterns:', args.in_patterns)
    print('out-folder:', args.out_folder)
    print('subcube-side:', args.subcube_side)
    print('crop-size:', args.crop_size)
    print('workers:', args.workers)
    print('crop-before-processing:', args.crop_before_processing)
    if args.crop_before_processing is not None:
        print('crop-before-processing-offset:', args.crop_before_processing_offset)
    print('')

    node_name = os.environ.get("SLURMD_NODENAME", "unknown")
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    print(f"Running on node: {node_name}, task ID: {task_id}")
    
    if task_id == 0: #first job in the array
        print(f'[TASK_ID = 0] creating {args.out_folder} folder.')
        

        if os.path.exists(args.out_folder): # NOTE: this check is not necessary
            print("[TASK_ID = 0] WARNING: Output folder already exists.")

        os.makedirs(args.out_folder, exist_ok=False)
        print(f'[TASK_ID = 0] folder {args.out_folder} created.\n')
    
    elif task_id == -1: # this happens when not running as a SLURM job array
        print(f'Creating {args.out_folder} folder.')
        os.makedirs(args.out_folder, exist_ok=False)
        print(f'folder {args.out_folder} created.\n')

    while not os.path.exists(args.out_folder):
        print(f'Waiting for {args.out_folder} to be created...')
        import time

        time.sleep(1)

    input_files = []
    for pattern in args.in_patterns:
        input_files.extend(glob(pattern))

    print(f'Found {len(input_files)} input files.\n')

    for input_file in input_files:
        gc.collect()
        print(f'Processing {input_file}...\n')

        if args.crop_before_processing is not None:
            arr_mmap = np.load(input_file, mmap_mode='r')
            ox, oy, oz = args.crop_before_processing_offset
            side = args.crop_before_processing
            arr = arr_mmap[
                :,
                ox : ox + side,
                oy : oy + side,
                oz : oz + side,
            ]
            Ng = side
        else:
            arr = np.load(input_file)
            ox, oy, oz = (0,0,0)
            _, Ng, Ng2, Ng3 = arr.shape
            assert Ng == Ng2 == Ng3, f'Input array must be cubic, but got shape {arr.shape}.'

        assert len(arr.shape) == 4, f'Input array must be 4D, but got shape {arr.shape}.'
        assert Ng % args.subcube_side == 0, f'Ng must be a multiple of subcube_side.'
        assert args.subcube_side % args.crop_size == 0, f'subcube_side must be a multiple of crop_size.'

        base_filename = os.path.splitext(os.path.basename(input_file))[0]

        # Prepare worker tasks
        tasks = []
        for x in range(0, Ng, args.subcube_side):
            for y in range(0, Ng, args.subcube_side):
                for z in range(0, Ng, args.subcube_side):
                    tasks.append(
                        (arr, (x, y, z), args.subcube_side, args.crop_size, args.out_folder, base_filename, (ox, oy, oz), args.filename_digits_per_coordinate, args.use_pt)
                    )

        # Run in parallel
        with Pool(processes=args.workers) as pool:
            pool.map(subcube_worker, tasks)

        print(f'Finished {input_file}.\n')

        del arr