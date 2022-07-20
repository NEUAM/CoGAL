# Image Matching via Global and Local Constraints 
## Overview
This method adds seed points fifiltering based on Delaunay triangulation, adaptive neighborhood radius selection based on regional feature density, and two-way ratio test based on AdaLAM
PROSAC. First incorporates global constraints into region-based image matching, filters the seed points by the topological consistency of the scene structure, and then performs outliers 
elimination in the neighborhoods near the seed points. Region mismatches caused by similar textures effectively removed in this way. The adaptive neighborhood radius reduces the redundant 
computation of overlapping parts and also makes the points in the texture sparse regions more taken into account. Two-way ratio test assigns more accurate scores to matches, thereby increasing 
the efficiency of PROSAC.

## Requirement
```angular2html
1. python 3.7
2. pytorch
3. tqdm
```


## Examples

The code to run the example is at 
```./examples/example.py. ```
Add images to match via ```--im1 --im2```.

## License
Our code improves upon AdaLAM's code. https://github.com/cavalli1234/AdaLAM
