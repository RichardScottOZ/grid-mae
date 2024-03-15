# GRID-MAE

- Investigate using multiscale grids in a Vision Transformer.

# Prior Art
## SATMAE
- https://github.com/techmn/satmae_pp
- A few notes : https://github.com/RichardScottOZ/satmae_pp

### DATA
- Trained on https://github.com/fMoW/dataset
	- To invesigate structure
	- Presumably 3 band groupings for 10, 20 and 60m resolution patches around pictures of locations of interest - airports, zoos, etc.
	- Designed to classify these
	
# Problem
## General
- Multiscale adaptation for segmentation based on general layers

### Geoscience
- Could be remote sensing, but any domain for geoscience, geophysics, geology, structure etc.

### Loss functions	
- might be continuous or one hot
	
	