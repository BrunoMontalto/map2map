import os
import sys
import warnings
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.fields_test import FieldDataset
from .data import norms
from . import models
from .models import narrow_cast
from .utils import import_attr, load_model_state_dict

from glob import glob
from .models.power_loss import LogSpectralDistance, PowerL2E_non_diff


def test(args):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            warnings.warn('Not parallelized but given more than 1 GPUs')

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda', 0)

        torch.backends.cudnn.benchmark = True
    else:  # CPU multithreading
        device = torch.device('cpu')

        if args.num_threads is None:
            args.num_threads = int(os.environ['SLURM_CPUS_ON_NODE'])

        torch.set_num_threads(args.num_threads)

    print('pytorch {}'.format(torch.__version__))
    pprint(vars(args))
    sys.stdout.flush()


    #assuming only one test pair of fields, batch size 1

    #check if assumptions for testing are met
    if args.batch_size != 1:
        raise ValueError('for testing, batch size must be 1')
    if len(glob(args.test_in_patterns[0])) > 1:
        raise NotImplementedError('for testing, only one input pattern is supported')
    
    if len(args.test_in_patterns) > 2:
        raise NotImplementedError('for testing, only two input fields are supported (dis and vel)')

    if args.crop is None or args.crop_start is None or args.crop_stop is None:
        raise ValueError('for testing, crop, crop_start and crop_stop must be specified')

    test_dataset = FieldDataset(
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=False,
        aug_shift=None,
        aug_add=None,
        aug_mul=None,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        mmap_only=True, #NOTE: added this line
        ignore_target=not args.test_tgt_patterns[0],
        **args.misc_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    crop_start, crop_stop, crop_step = args.crop_start, args.crop_stop, test_dataset.crop_step

    in_chan, out_chan = test_dataset.in_chan, test_dataset.tgt_chan

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(sum(in_chan), sum(out_chan),
                  scale_factor=args.scale_factor, **args.misc_kwargs)
    model.to(device)

    criterion = import_attr(args.criterion, torch.nn, models,
                            callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    logspectraldist = LogSpectralDistance()
    logspectraldist.to(device)

    powerL2E_non_diff = PowerL2E_non_diff()
    powerL2E_non_diff.to(device)

    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    print('model state at epoch {} loaded from {}'.format(
        state['epoch'], args.load_state))
    del state

    model.eval()

    print('length of test dataset: {}'.format(len(test_dataset)))

    #prepare two (3, N, N, N) np arrays (one for dis, one for vel), with N =((crop-stop - crop-start)*2)**3
    N = ((crop_stop - crop_start) * args.scale_factor)
    print(f'SR Ng: {N}')
    dis_array = np.empty((3, N, N, N), dtype=np.float32)
    if len(args.test_in_patterns) == 2:
        vel_array = np.empty((3, N, N, N), dtype=np.float32)




    crop_start = np.broadcast_to(crop_start, (3,))
    crop_stop = np.broadcast_to(crop_stop, (3,))
    crop_step = np.broadcast_to(crop_step, (3,))

    anchors = np.stack(np.mgrid[tuple(
        slice(crop_start[d], crop_stop[d], crop_step[d])
        for d in range(3)
    )], axis=-1).reshape(-1, 3)
    ncrop = len(anchors)

    

    with torch.no_grad():
        average_loss = 0
        average_logspectraldist = 0
        average_powerL2E_non_diff = 0
        for i, data in enumerate(test_loader):
            input, target = data['input'], data['target']

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            if i < 5:
                print('##### sample :', i)
                print('input shape :', input.shape)
                print('output shape :', output.shape)
                print('target shape :', target.shape)

            output, target = narrow_cast(output, target) #NOTE: removed input from narrow_cast
            if i < 5:
                print('narrowed shape :', output.shape, flush=True)

            

            #if args.in_norms is not None:
            #    start = 0
            #    for norm, stop in zip(test_dataset.in_norms, np.cumsum(in_chan)):
            #        norm = import_attr(norm, norms, callback_at=args.callback_at)
            #        norm(input[:, start:stop], undo=True, **args.misc_kwargs)
            #        start = stop
            if args.tgt_norms is not None:
                start = 0
                for norm, stop in zip(test_dataset.tgt_norms, np.cumsum(out_chan)):
                    norm = import_attr(norm, norms, callback_at=args.callback_at)
                    norm(output[:, start:stop], undo=True, **args.misc_kwargs)
                    norm(target[:, start:stop], undo=True, **args.misc_kwargs)
                    start = stop

            if args.test_tgt_patterns[0]:
                print("output range:", output.min(), output.max(), output.mean())
                print("target range:", target.min(), target.max(), target.mean())

                loss = criterion(output, target)
                average_loss += loss

                loss2 = logspectraldist(output, target)
                average_logspectraldist += loss2

                loss3 = powerL2E_non_diff(output, target)
                average_powerL2E_non_diff += loss3

                if i % 16 == 0:
                    print("loss:",loss )
                    print("loss2:",loss2)
                    print("loss3:",loss3)
                    print(f'sample {i}/{len(test_loader)}')

            #test_dataset.assemble('_in', in_chan, input,
            #                      data['input_relpath'])
            
            #test_dataset.assemble('_tgt', out_chan, target,
            #                      data['target_relpath'])

            if args.save_output:
                #anchor = data['anchor'][0].numpy() #anchor of the crop in the original field
                #put in the right place in dis_array and vel_array

                output_dis = output[0, 0:3].cpu().numpy() #first 3 channels are dis
                if len(args.test_in_patterns) == 2:
                    output_vel = output[0, 3:6].cpu().numpy() #next

                # complete here (use anchors)
                _, icrop = divmod(i, ncrop)
                anchor = anchors[icrop]

                anchor = data['anchor'][0].numpy()  # anchor of the crop in the original field

                #print(f'anchor: {anchor}, crop_start: {crop_start}, crop_stop: {crop_stop}')

                dis_array[:, anchor[0]*args.scale_factor:(anchor[0] + crop_step[0])*args.scale_factor, anchor[1]*args.scale_factor:(anchor[1]+crop_step[1])*args.scale_factor, anchor[2]*args.scale_factor:(anchor[2]+crop_step[2])*args.scale_factor] = output_dis
                if len(args.test_in_patterns) == 2:
                    vel_array[:, anchor[0]*args.scale_factor:(anchor[0] + crop_step[0])*args.scale_factor, anchor[1]*args.scale_factor:(anchor[1]+crop_step[1])*args.scale_factor, anchor[2]*args.scale_factor:(anchor[2]+crop_step[2])*args.scale_factor] = output_vel

            #save
            # assume args.test_in_patterns[0] is something like 'F/LR/*.npy', get 'F/SR' as output folder
        
        

        
            
            
        #for filename, assume args.load_state is something like 'method/state_10.pt', get 'method_'+'10' as identifier
        epoch = os.path.basename(args.load_state).replace('state_', '').replace('.pt', '') #this will be '10' for 'state_10.pt'
        
        print(f'epoch: {epoch}')

        #get state folder name and conocatenate with epoch
        state_folder = os.path.basename(os.path.dirname(args.load_state)) #this will be 'method' for 'method/state_10.pt'
        if args.save_output:
            identifier = f'{state_folder}_epoch_{epoch}_crop_step_{crop_step[0]}_crop_start_{crop_start[0]}_crop_stop_{crop_stop[0]}'
            print(f'identifier: {identifier}')

            out_folder = args.test_in_patterns[0].replace('LR', 'SR').rsplit('/', 2)[0] + '/SR' #NOTE: ugly but works
            print(f'out_folder: {out_folder}')
            os.makedirs(out_folder, exist_ok=True)

            #get original filename args.test_in_patterns[0] is a path like 'F/LR/seed_123456_dis.npy', get 'seed_123456_dis.npy'
            snap_name_dis = os.path.basename(args.test_in_patterns[0]).replace(".npy","") + "_" + identifier + args.suffix + '.npy'
            snap_name_vel = snap_name_dis.replace('dis', 'vel')
            
            final_path_dis = os.path.join(out_folder, snap_name_dis)
            final_path_vel = os.path.join(out_folder, snap_name_vel)

            print(f'saving to {final_path_dis} and {final_path_vel}')

            np.save(final_path_dis, dis_array)
            if len(args.test_in_patterns) == 2:
                np.save(final_path_vel, vel_array)

    average_loss /= len(test_loader)
    print(f"# total average loss({args.criterion}) for {state_folder}:",average_loss)

    average_logspectraldist /= len(test_loader)
    print(f"# total average logspectraldist for {state_folder}:",average_logspectraldist)

    average_powerL2E_non_diff /= len(test_loader)
    print(f"# total average average_powerL2E_non_diff for {state_folder}:",average_powerL2E_non_diff)

