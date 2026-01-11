import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ >= "0.3.2"  # version check  #0.4.12 used here as per satmae repo
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.datasets import build_grid_dataset, getRasterLayers
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import models_mae_group_channels

import torch.nn.functional as F

import rasterio as rio


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

def saveResultsFeatures(image, raster_name, outName, data_dir='dataset/grid/grid/', downscale=96, x0=0, y0=0, nodata=None):
    from util.misc import normalize_data_dir
    # Ensure data_dir ends with a slash
    data_dir = normalize_data_dir(data_dir)
    refName = data_dir + raster_name + ".tif"

    if len(image.shape)==2:
       outH,outW = image.shape
       outD = 1
    else:
       outH,outW,outD = image.shape
    with rio.open(refName, "r") as f:
        pix2Geo = f.transform
        crs = f.crs

    if downscale<1:
       downscale = 1
    w,s,e,n = rio.transform.array_bounds(1, 1, pix2Geo)
    gpx = pix2Geo[0]
    gpy = pix2Geo[4]
    left = w + x0*gpx
    top  = n + y0*gpy
    newPix2Geo = rio.transform.from_bounds(left, top+gpy*downscale, left+gpx*downscale, top, 1, 1)

    with rio.open(outName, 'w', width=outW,height=outH,
       driver="GTiff",count=outD,crs=crs,
       compress="lzw",
       tiled=True, blockxsize=256, blockysize=256, nodata=nodata,
       transform=newPix2Geo, dtype=image.dtype) as file:
            if outD==1:
                file.write(image.reshape(outH,outW), 1 )
            else:
                file.write(np.transpose(image,(2,0,1)), [i for i in range(1,outD+1)] )

    print("Wrote",outName)

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

    print(model_without_ddp.cls_token.mean(), model_without_ddp.cls_token.shape)

    dataset_inference = build_grid_dataset(is_train=False, args=args)

    data_dir = getattr(args, 'data_dir', 'dataset/grid/grid/')
    srcMeta, srcUseful = getRasterLayers(args.train_path, data_dir)

    input_shape = srcUseful.shape
    print("INPUT SHAPE:",input_shape)
    
    outputFeatures = model_without_ddp.cls_token.squeeze().shape[0]
    print("OUTPUT FEATURES:", model_without_ddp.cls_token.squeeze().shape[0]) #hardcode for now, get from model later.
    print("INPUT SIZE:",args.input_size)

    # this will need adjusting later to get nice predictions, borders, edges etc.
    resultsShape = srcUseful[::args.input_size,::args.input_size].shape
    result_height, result_width = resultsShape

    print("IntShape:",srcUseful[::args.input_size,::args.input_size].shape)
    print("IntSum:",srcUseful[::args.input_size,::args.input_size].sum())

    result = np.zeros( (result_height, result_width, outputFeatures), dtype=np.float32 )

    resultcls = np.zeros( (result_height, result_width, outputFeatures), dtype=np.float32 )

    print("RESULT SHAPE:",result.shape)

    batch = np.empty((args.batch_size,args.input_size,args.input_size, args.input_channels))
    #print("BATCH SHAPE:",batch.shape)
    batch_length = len(batch)
    #print("BATCH LENGTH:",len(batch))

    batch_count = 0
    targets = []

    tile_width = args.input_size
    tile_height = args.input_size    
    
    model.eval()

    normhook = {}
    def get_normhook(name):
        def hook(model, input, output):
            normhook[name] = output
        return hook


    def get_block11(name):
        def hook(model, input, output):
            normhook[name] = output
        return hook

    #def get_decembed(name):
        #def hook(model, input, output):
            #normhook[name + '_output'] = output
        #return hook

    model.norm.register_forward_hook(get_normhook("normhook"))
    #model.blocks[11].mlp.fc2.register_forward_hook(get_block11("block11hook"))
    model.blocks[-1].mlp.fc2.register_forward_hook(get_block11("block11hook"))
    #model.decoder_embed.register_forward_hook(get_decembed("decembedhook"))

    
    def flushTargets():
        #get pred here
            
        #mse_loss, l1_loss, _, _ = model(images, [images_up_2x, images_up_4x], mask_ratio=args.mask_ratio)
        #need list of upscale for this model

        img_as_tensor = dataset_inference.transform(batch[0])  # (c, h, w)

        img_dn_2x = F.interpolate(img_as_tensor.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
        img_dn_4x = F.interpolate(img_dn_2x.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)

        batch_p = np.empty((batch.shape[0],img_as_tensor.shape[0],img_as_tensor.shape[1],img_as_tensor.shape[2]), dtype=np.float32)
        batch_p_2x = np.empty((batch_p.shape[0],img_dn_2x.shape[0],img_dn_2x.shape[1],img_dn_2x.shape[2]), dtype=np.float32)
        batch_p_4x = np.empty((batch_p.shape[0],img_dn_4x.shape[0],img_dn_4x.shape[1],img_dn_4x.shape[2]), dtype=np.float32)

        batch_p = torch.tensor(batch_p)
        batch_p_2x = torch.tensor(batch_p_2x)
        batch_p_4x = torch.tensor(batch_p_4x)

        batch_orig = torch.tensor(batch.transpose(0,3,1,2).astype(np.float32))
        #print("BATCH ORIG:",batch_orig.shape)

        for b in range(batch_p.shape[0]):
            #print("B:",b)
            #print("B TRANSFORM SHAPE:",self.transform(batch[b]).shape )
            #print("BATCH P[0] SHAPE:",batch_p[b].shape)
            #print("BATCH B[0] SHAPE:",batch[b].shape)
            batch_p[b] = dataset_inference.transform(batch[b])

        for b in range(batch_p_2x.shape[0]):
            batch_p_2x[b] = F.interpolate(batch_p[b].unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
            #for b in range(batch_p_4x.shape[0]):
            batch_p_4x[b] = F.interpolate(batch_p_2x[b].unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)

        #predictions = model(batch_p_4x, [batch_p_2x, batch_p], mask_ratio=args.mask_ratio)
        #predictions = model(batch_orig, [batch_p_2x, batch_p], mask_ratio=args.mask_ratio)

        predictions = model(batch_orig, [batch_p_2x, batch_p], mask_ratio=args.mask_ratio)
        #print("PRED",predictions[2].shape)
        #print("MASK",predictions[3].shape)        

        #mt = predictions[2].detach().numpy()
        #print(mt.mean())
        #mt = mt.transpose(0,2,3,1)
        #plt.imshow(mt[0,:,:,0:1])
        #plt.show()

        #print("LENNORMHOOK:",len(normhook['normhook']))
        #print("LENBLOCK11HOOK:",len(normhook['block11hook']))

        #for o in normhook['normhook']:
            #print("OUTPUTLOOP:",o.shape)
        #print("RESULT MEAN BEFORE:",result.mean())

        for tileid, (x,y) in enumerate(targets):
            # work out borders and centres and things here

            #print("TARGETS ID, X, Y",tileid,(x,y),"TW:",tile_width,"TH:",tile_height)
            #pass
            #result[y:y+th, x:x+tw] = stuff from predictions yet to work out
            #loss, ms_loss, pred, mask = print(predictions)
            
            #print(normhook['block11hook'][tileid,:,:].detach().numpy().mean(axis=0))
            #print(normhook['block11hook'][tileid,:,:].detach().numpy().shape)
            result[y,x] = normhook['block11hook'][tileid,:,:].detach().numpy().mean(axis=0)

            resultcls[y,x] = normhook['block11hook'][tileid,1:,:].detach().numpy().mean(axis=0)

        #print("RESULT SHAPE:",result.shape)
        #print("RESULT MEAN AFTER:",result.mean())

        #quit()

    for y in tqdm(range(result_height)):
        yStart = y * tile_height
        for x in range(result_width):
            #xStart = x * tile_width
            xStart = x * tile_width
            #get tile with coords for batch[batch_count]
            dataset_inference.get_tile( xStart, yStart, batch[batch_count])
            #targets.append((xStart, yStart))
            targets.append((x, y))
            #print("x,y,xStart,yStart:",x,y,xStart,yStart)
            batch_count += 1

            if batch_count >= batch_length:
                flushTargets()
                #print(batch.mean())
                targets = []
                batch_count = 0


    #print(srcMeta)
    saveResultsFeatures(result, srcMeta[0]['name'], args.output_dir + '/result3.tif', data_dir=data_dir, downscale=args.input_size)

    #no cls version for comparison
    saveResultsFeatures(resultcls, srcMeta[0]['name'], args.output_dir + '/result3cls.tif', data_dir=data_dir, downscale=args.input_size)
    

    from sklearn.decomposition import PCA
    PCA_SIZE = 32

    validPcaPoints = srcUseful[::args.input_size,::args.input_size].flatten()
    pca = PCA(n_components=PCA_SIZE)
    pca.fit( result.reshape(-1,outputFeatures)[validPcaPoints] )

    vecs = pca.transform(  result.reshape(-1,outputFeatures) )
    vecs[validPcaPoints==False] = 0
    vecs = vecs.reshape( result_height,result_width, -1 )

    saveResultsFeatures(vecs, srcMeta[0]['name'], args.output_dir + '/vec3.tif', data_dir=data_dir, downscale=args.input_size)

    #no cls version for comparison
    pca.fit( resultcls.reshape(-1,outputFeatures)[validPcaPoints] )

    vecs = pca.transform(  resultcls.reshape(-1,outputFeatures) )
    vecs[validPcaPoints==False] = 0
    vecs = vecs.reshape( result_height,result_width, -1 )

    saveResultsFeatures(vecs, srcMeta[0]['name'], args.output_dir + '/veccls3.tif', data_dir=data_dir, downscale=args.input_size)
    #vec notes update

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
