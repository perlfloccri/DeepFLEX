# DeepFLEX  

## About

Deep learning-based single-cell analysis pipeline for FLuorescence multiplEX imaging via MELC (Multi-Epitope Ligand Cartography [1]).  

[Daria Lazic, Florian Kromp et al.  
**Single-cell landscape of bone marrow metastases in human neuroblastoma unraveled by deep multiplex imaging**](https://www.biorxiv.org/content/10.1101/2020.09.30.321539v1)

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

## Requirements

All requirements can be found [here](https://github.com/perlfloccri/DeepFLEX/tree/master/required_prerequisites).

## Installation

For interactive and quantitative analysis of single-cell data generated by DeepFLEX, we used:  
- Cytosplore: an interactive tool for single-cell analysis (download [here](https://www.cytosplore.org/))
- [Seaborn](https://seaborn.pydata.org/): a python data visualization library 

## Data availability

Download the MELC imaging data of our 8 samples [here](https://cloud.stanna.at/sharing/qiN0u9QPO).  

## References

<a id="1">[[1]](https://www.nature.com/articles/nbt1250)</a> 
Schubert, W. et al. (2006). 
Analyzing proteome topology and function by automated multidimensional fluorescence microscopy.
Nature Biotechnology.    
<a id="1">[[2]](https://www.nature.com/articles/nmeth.3323)</a> 
Smith, K. et al. (2015). 
CIDRE: An illumination-correction method for optical microscopy. 
Nat. Methods, 12, 404-406.  
<a id="1">[[3]](https://arxiv.org/abs/1907.12975)</a> 
Kromp, F. et al. (2019). 
Deep Learning architectures for generalized immunofluorescence based nuclear image segmentation. 
arXiv.  
<a id="1">[[4]](https://www.nature.com/articles/s41597-020-00608-w)</a> 
Kromp, F. et al. (2020). 
An annotated fluorescence image dataset for training nuclear segmentation methods. 
Scientific Data, 7, 262.  
<a id="1">[[5]](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.20426)</a> 
Chang, Y.H. et al. (2020). 
RESTORE: Robust intEnSiTy nORmalization mEthod for multiplexed imaging. 
Commun. Biol., 3, 1-9.  
<a id="1">[[6]](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.12893)</a> 
Höllt, T. et al. (2016). 
Cytosplore: Interactive Immune Cell Phenotyping for Large Single-Cell Datasets. 
Comput. Graph., 35, 171-180.  
