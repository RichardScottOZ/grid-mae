# MAE

# Packages
- timm
- opencv/opencv-python
- rasterio

# Errors
- Widowns 10
## pillow 10.1
-  File "C:\Users\rscott\AppData\Local\miniconda3\envs\textgen\Lib\site-packages\PIL\Image.py", line 84, in <module>
    from . import _imaging as core
ImportError: DLL load failed while importing _imaging: The operating system cannot run %1.

- try Pillow=9.3

## rasterio 1.3.9
- from rasterio._version import gdal_version, get_geos_version, get_proj_version
ImportError: DLL load failed while importing _version: The operating system cannot run %1.

- Not a rasterio 1.2.10 problem
- gdal most likely
- start with 3.3?
- will need higher for python 3.11 however

# from scratch
- pytorch as pre instructions
- https://pytorch.org/get-started/locally/
- geopandas to get bonus gdal
- rasterio via conda-forge
- tensorboard via conda-forge
- pip install timm
- opencv

## Errors
```python
traceback (most recent call last):
  satmae_pp\main_pretrain.py", line 237, in <module>
    main(args)
  satmae_pp\main_pretrain.py", line 178, in main
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
AttributeError: module 'timm.optim.optim_factory' has no attribute 'add_weight_decay'
```python

## Environment advice
- from https://github.com/techmn/satmae_pp/issues/1
- Installing timm as advised 0.4.12
	- Runs until it can't find data as expected, so good.


# Channels
Group channels here in

class MaskedAutoencoderGroupChannelViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=96, patch_size=8, in_chans=10, spatial_mask=False,
                 channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
                 channel_embed=256, embed_dim=1024, depth=24, num_heads=16,
                 decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 proj_ratio=4):
        super().__init__()

    - need to change channels and groups to adapt

# Dataset Type Creation
 elif args.dataset_type == 'sentinel':
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size*4, mean, std) # input_size*2 = 96*2 = 192
        dataset = SentinelIndividualImageDataset(file_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands)

# Dataset Class
    class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

# Classes
- don't need if have generic training need

# Example data formats
- https://purl.stanford.edu/vg497cb6002

README.md
File 1

2.85 kB	 DownloadREADME.md
fmow-sentinel.tar.gz
File 2

77.50 GB	 Downloadfmow-sentinel.tar.gz
test_gt.csv
File 3

21.16 MB	 Downloadtest_gt.csv
train.csv
File 4

178.78 MB	 Downloadtrain.csv
val.csv
File 5

21.15 MB	 Download

Abstract
The Functional Map of the World - Sentinel-2 corresponding images (fMoW-Sentinel) dataset consists of image time series collected by the Sentinel-2 satellite, corresponding to locations from the Functional Map of the World (fMoW) dataset across several different times. The dataset follows the locations of the fMoW dataset, which are categorized by 62 different types of building/land uses. These images have a 10m spatial resolution, are created from cloud composites over 90 day intervals, and contain one channel for each of the 13 bands of the Sentinel-2 surface reflectance dataset. The dataset is split into train, validation, and test sets according to the original fMoW data splits (metadata is contained in train.csv, val.csv, test_gt.csv), with 712,874 training images, 84,939 validation
images, and 84,966 test images. The fMoW-Sentinel dataset is derived from two data sources with their own licenses: The Functional Map of the World Challenge Public License (https://raw.githubusercontent.com/fMoW/dataset/master/LICENSE) applies to the locations and categories of the images in the dataset (i.e. the data in the metadata CSV files), while the Sentinel-2 License (https://scihub.copernicus.eu/twiki/pub/SciHubWebPortal/TermsConditions/Sentinel_Data_Terms_and_Conditions.pdf) applies to the images themselves.

# Preprocessing
- From the satemae repository
Note that when loading the train.csv or val.csv files, you may have to preprocess a column called image_path. The image_path for any row can be constructed like this:

fmow-sentinel/<split>/<category>/<category>_<location_id>/<category>_<location_id>_<image_id>.tif

- original function https://github.com/fMoW/baseline/blob/master/code/data_ml_functions/dataFunctions.py#L107