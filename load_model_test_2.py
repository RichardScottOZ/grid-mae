import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ >= "0.3.2"  # version check  #0.4.12 used here as per satmae repo
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.datasets import build_grid_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import models_mae_group_channels

def get_args_parser():
    parser = argparse.ArgumentParser('GridMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_type', default='group_c', choices=['group_c', 'vanilla'], help='Use channel model')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_channels', default=2, type=int, help='number of different grids')
    parser.add_argument('--input_size', default=96, type=int, help='images input size')
    parser.add_argument('--patch_size', default=8, type=int, help='patch embedding patch size')
    parser.add_argument('--mask_ratio', default=0.75, type=float, 
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--spatial_mask', action='store_true', default=False,
                        help='Whether to mask all channels of a spatial location. Only for indp c model')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.0001, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--train_path', default='dataset/grid/train.csv', type=str, help='Train .csv path')
    parser.add_argument('--data_dir', default='dataset/grid/grid/', type=str, help='Directory containing grid data files')
    parser.add_argument('--dataset_type', default='grid', choices=['rgb', 'sentinel', 'grid'],
                        help='Whether to use fmow rgb, sentinel, or other dataset.')
    parser.add_argument('--masked_bands', type=int, nargs='+', default=None,
                        help='Sequence of band indices to mask (with mean val) in sentinel dataset')
    parser.add_argument('--dropped_bands', type=int, nargs='+', default=None,
                        help="Which bands (0 indexed) to drop from sentinel data.")
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for GroupC mae")

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--wandb', type=str, default=None,help="Wandb project name, eg: sentinel_pretrain")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)  # prev default was -1
    parser.add_argument('--dist_on_itp', action='store_true')  #don't need distributed for test
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser

def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            # del checkpoint['model']['head.weight']
            # del checkpoint['model']['head.bias']
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")



def main(args):
    if args.model_type == 'group_c':
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            # Create default grouped_bands based on input_channels
            args.grouped_bands = misc.get_default_grouped_bands(args.input_channels)

        print(f"Grouping bands {args.grouped_bands}")
        model = models_mae_group_channels.__dict__[args.model](img_size=args.input_size,
                                                               patch_size=args.patch_size,
                                                               in_chans=args.input_channels,
                                                               channel_groups=args.grouped_bands,
                                                               spatial_mask=args.spatial_mask,
                                                               norm_pix_loss=args.norm_pix_loss)
    
    else:
        model = models_mae.__dict__[args.model](img_size=args.input_size,
                                                patch_size=args.patch_size,
                                                in_chans=dataset_train.in_c,
                                                norm_pix_loss=args.norm_pix_loss)
    #model.to(device)

    model_without_ddp = model

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    #optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    optimizer = torch.optim.AdamW(param_groups, lr=0.000001, betas=(0.9, 0.95))
    print("OPTIMIZER:",optimizer)
    loss_scaler = NativeScaler()   

    load_model(args,model_without_ddp, optimizer, loss_scaler)

    #from torchsummary import summary
    #import torchinfo
    #summary(model, input_size = (96, 96, 8), batch_size = -1)

    print(model_without_ddp)

    print("MODEL WITHOUT DDP")
    for name, param in model_without_ddp.named_parameters(): 
        print(name, param.shape) 

    print(model_without_ddp.cls_token)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
