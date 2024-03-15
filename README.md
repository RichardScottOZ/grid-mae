# GRID-MAE

- Investigate using multiscale grids in a Vision Transformer Masked Autoencoder.

# Question
- Will it be worth the computational requirements?

# Prior Art
## SATMAE
- https://github.com/techmn/satmae_pp
- A few notes : https://github.com/RichardScottOZ/satmae_pp
- LICENSE - Apache 2.0
	- https://github.com/techmn/satmae_pp/blob/main/LICENSE

### DATA
- Trained on https://github.com/fMoW/dataset [70GB tarball]
	- To invesigate structure
	- Presumably 3 band groupings for 10, 20 and 60m resolution patches around pictures of locations of interest - airports, zoos, etc.
	- Designed to classify these
### METADATA
- Metadata file - csv with location/polygon coordinates, class type etc.	
	- These are here https://raw.githubusercontent.com/fMoW/dataset/master/LICENSE

#### Example
	category	location_id	image_id	timestamp	polygon
0	airport	0	6	2015-07-25T08:45:14Z	POLYGON ((32.666164117900003 39.932541952376475, 32.711078120537337 39.932541952376475, 32.711078120537337 39.967113357199999, 32.666164117900003 39.967113357199999, 32.666164117900003 39.932541952376475))

### Instllation
- pytorch as pre instructions
	- https://pytorch.org/get-started/locally/
- geopandas to get bonus gdal
- rasterio via conda-forge
- tensorboard via conda-forge
- pip install timm
- pip install opencv-python
- [so far]
	
	
	
# Problem
## General
- Multiscale adaptation for segmentation based on general layers

### Geoscience
- Could be remote sensing, but any domain for geoscience, geophysics, geology, structure etc.

### Loss functions	
- might be continuous or one hot

### Simple Example
- To keep it in human finger space [and patch space]
- Take a set of geophysics grids at 100m resolution
- Take another set at 200m resolution

### Padding
- Planets, surface etc. - not rectangles

### Data Loader
- Likely want on the fly grid slicing into tiles, not directory structures full of sliced up grids in folders

### Overlap training tiles
- Is this useful for autoencoders here beyond smoothing reasons

### Input channels
- Needs to be general

### Groupings
- Resolution groupings - this is a satmae parameter already

	
	