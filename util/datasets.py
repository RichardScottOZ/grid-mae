# any modifications post 20240316 by Richard Scott

import os
import pandas as pd
import numpy as np
import warnings
import random
import json
import cv2
from glob import glob
from typing import Any, Optional, List
import rasterio
from rasterio import logging

import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


CATEGORIES = ["gravity", "magnetics"]


class GridDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)



#########################################################
# GRID DEFINITIONS
#########################################################


class GridNormalize:
    # need to change this to general
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """
    
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


def getRasterLayers(csv_path):
    df_grid = pd.read_csv(self.csv_path)

    grid_dict = {}
    self.mean = []
    self.std = []
    
    srcMeta = []
    
    for index, row in df_grid.iterrows():
        print("stats for:",row['category'])
        srcMeta.append({"name":row['category'], "loss":"mse"})

    for srcData in self.srcMeta:
        with rasterio.open('dataset/grid/grid/' + row['category'] + '.tif') as src:
            print(src.meta)
            data = src.read(1, masked=True)
            self.mean.append(np.nanmean(data))
            self.std.append(np.nanstd(data))
            
            srcData['mean'] = self.mean
            srcData['std'] = self.std
            srcData['min'] = np.nanmin(data)
            srcData['max'] = np.nanmax(data)
            srcData['range'] = np.nanmax(data)
            srcData['scale'] = 1.0/(srcData['max'] - srcData['min'] )
            srcData['data'] = data
            
    #print(self.srcMeta)
    usefulData = srcData['data'] != np.nan
    
    return srcMeta, usefulData


class GridIndividualImageDataset(GridDataset):
    #need to change this to general    
    label_types = ['value', 'one-hot']
    mean = []
    std = []
    #mean = [1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
            #1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            #1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117]
    #std = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
           #948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           #1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 years: Optional[List[int]] = [*range(2000, 2021)],
                 categories: Optional[List[str]] = None,
                 label_type: str = 'value',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands = [0, 9, 10],
                 batch_size = None):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.csv_path = csv_path
        self.base_path = '/'
        
        self.batch_size = batch_size
        self.batch_width = 224
        self.batch_height = 224  #hardcode a default start for now

        # extract base folder path from csv file path
        path_tokens = csv_path.split('/')
        for token in path_tokens:
            if '.csv' in token:
                continue
            self.base_path += token.strip() + '/'

        self.df = pd.read_csv(csv_path) \
            .sort_values(['category', 'location_id', 'timestamp'])

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:',
                ', '.join(self.label_types))
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)
            
        df_grid = pd.read_csv(self.csv_path)

        grid_dict = {}
        self.mean = []
        self.std = []
        
        self.srcMeta = []
        
        
        for index, row in df_grid.iterrows():
            print("stats for:",row['category'])
            self.srcMeta.append({"name":row['category'], "loss":"mse"})

        for srcData in self.srcMeta:
            with rasterio.open('dataset/grid/grid/' + row['category'] + '.tif') as src:
                print(src.meta)
                data = src.read(1, masked=True)
                self.mean.append(np.nanmean(data))
                self.std.append(np.nanstd(data))
                
                srcData['mean'] = self.mean
                srcData['std'] = self.std
                srcData['min'] = np.nanmin(data)
                srcData['max'] = np.nanmax(data)
                srcData['range'] = np.nanmax(data)
                srcData['scale'] = 1.0/(srcData['max'] - srcData['min'] )
                srcData['data'] = data
                
        #print(self.srcMeta)
        self.usefulData = srcData['data'] != np.nan
        self.height = self.usefulData.shape[0]
        self.width = self.usefulData.shape[1]
        
        perEpoch = int((self.usefulData[::self.batch_height,::self.batch_width]>0).sum())
        self.batch_count = perEpoch // batch_size
        self.perEpoch = self.batch_count * batch_size        
        
    def fill(self, batch):
        while True:
            xRest = np.random.randint(self.width - self.batch_width) #range to get sample from
            yRest = np.random.randint(self.height - self.batch_height) #range to get sample from
            if self.usefulData[yRest+(self.batch_height>>1), xRest+(self.batch_width >> 1)]:
                break
        
    
    def __len__(self):
        # this is the number of tiles
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor images, and integer label as a dict.
        """
        selection = self.df.iloc[idx]

        folder = 'fmow-sentinel/train'
        if 'val' in self.csv_path:
            folder = 'fmow-sentinel/val'
        elif 'test' in self.csv_path:
            folder = 'fmow-sentinel/test_gt'

        cat = selection['category']
        loc_id = selection['location_id']
        img_id = selection['image_id']
        image_path = '{0}/{1}_{2}/{3}_{4}_{5}.tif'.format(cat,cat,loc_id,cat,loc_id,img_id)

        abs_img_path = os.path.join(self.base_path, folder, image_path)

        images = self.open_image(abs_img_path)  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection['category'])

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]


        img_dn_2x = F.interpolate(img_as_tensor.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
        img_dn_4x = F.interpolate(img_dn_2x.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)

        return {'img_up_4x':img_as_tensor, 'img_up_2x':img_dn_2x, 'img':img_dn_4x, 'label':labels}

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(GridNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(GridNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)



###################################################################################################################

def build_grid_dataset(is_train: bool, args) -> GridDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    file_path = os.path.join(args.train_path if is_train else args.test_path)
    
    print(args)
    print("Train Path:",file_path)


    if args.dataset_type == 'grid':
        mean = GridIndividualImageDataset.mean
        std = GridIndividualImageDataset.std
        transform = GridIndividualImageDataset.build_transform(is_train, args.input_size*4, mean, std) # input_size*2 = 96*2 = 192
        print(args.batch_size)
        dataset = GridIndividualImageDataset(file_path, transform, masked_bands=args.masked_bands, dropped_bands=args.dropped_bands, batch_size=args.batch_size)

    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset.mean, dataset.std)

    return dataset
