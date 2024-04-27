# DeepFLEX  

## About

Deep learning-based single-cell analysis pipeline for FLuorescence multiplEX imaging via MELC (Multi-Epitope Ligand Cartography [1]).  

[Daria Lazic, Florian Kromp et al.  
**Landscape of Bone Marrow Metastasis in Human Neuroblastoma Unraveled by Transcriptomics and Deep Multiplex Imaging**](https://doi.org/10.3390/cancers13174311)

<img src="https://github.com/perlfloccri/DeepFLEX/blob/master/deepflex.jpg" alt="alt text" width="300">
    
**Contact:** Daria Lazic ([daria.lazic@ccri.at](mailto:daria.lazic@ccri.at))  

## Content

The pipeline is based on methods for:  
-image processing (registration, flat-field correction, retrospective multi-image illumination correction by CIDRE [2])    
-cell and nucleus segmentation by Mask R-CNN [3], [4]  
-feature extraction  
-normalization by negative control secondary antibodies and RESTORE [5]  
-single-cell analysis (Cytosplore [6], seaborn)  

A compiled release with all necessary dependencies pre-installed is available from [dockerhub](https://hub.docker.com/repository/docker/imageprocessing29092020/deepflex).	
Nvidia-docker is required to run the image (for tensorflow-gpu support).

## Requirements

All requirements can be found [here](https://github.com/perlfloccri/DeepFLEX/tree/master/required_prerequisites).

## Installation

For interactive and quantitative analysis of single-cell data generated by DeepFLEX, we used:  
- Cytosplore: an interactive tool for single-cell analysis (download [here](https://www.cytosplore.org/))
- [Seaborn](https://seaborn.pydata.org/): a python data visualization library 

## Start the pipeline

Navigate to the [code](https://github.com/perlfloccri/DeepFLEX/tree/master/code) folder and run the pipeline.sh script.
## Data availability

Download the MELC imaging data of our 8 samples [here](https://doi.org/10.5281/zenodo.5906989).  

## References

<a id="1">[[1]](https://www.nature.com/articles/nbt1250)</a> 
Schubert, W. et al. (2006). 
Analyzing proteome topology and function by automated multidimensional fluorescence microscopy.
Nature Biotechnology.    
<a id="1">[[2]](https://www.nature.com/articles/nmeth.3323)</a> 
Smith, K. et al. (2015). 
CIDRE: An illumination-correction method for optical microscopy. 
Nat. Methods, 12, 404-406.  
<a id="1">[[3]](https://ieeexplore.ieee.org/document/9389742)</a> 
Kromp, F. et al. (2019). 
Evaluation of Deep Learning Architectures for Complex Immunofluorescence Nuclear Image Segmentation. 
IEEE.  
<a id="1">[[4]](https://www.nature.com/articles/s41597-020-00608-w)</a> 
Kromp, F. et al. (2020). 
An annotated fluorescence image dataset for training nuclear segmentation methods. 
Scientific Data, 7, 262.  
<a id="1">[[5]](https://www.nature.com/articles/s42003-020-0828-1)</a> 
Chang, Y.H. et al. (2020). 
RESTORE: Robust intEnSiTy nORmalization mEthod for multiplexed imaging. 
Commun. Biol., 3, 1-9.  
<a id="1">[[6]](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.12893)</a> 
Höllt, T. et al. (2016). 
Cytosplore: Interactive Immune Cell Phenotyping for Large Single-Cell Datasets. 
Comput. Graph., 35, 171-180.  
