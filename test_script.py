import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import torch.distributed as dist
import torch_xla.distributed.xla_backend
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_xla.experimental import pjrt
import torch_xla.experimental.pjrt_backend

import argparse
import datetime
import numpy as np
import time
#import torch.backends.cudnn as cudnn
import json
import os

use_tpu = True

def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)

    return parser.parse_args()



def main(rank, args):
    print('In main function')
    dist.init_process_group('xla', init_method='pjrt://')
    xm.mark_step()
    print('End of main function')
    

if __name__ == '__main__':
    opts = get_args()
    if use_tpu:
        print('Before xmp.spawn call')
        xmp.spawn(main, args=(opts,), nprocs=None)  # spawn
    else:
        main(0, opts)
