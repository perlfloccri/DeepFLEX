from Classes.Helper import SVGTools;
import matplotlib.pyplot as plt;
from tifffile import tifffile
import glob
import os
diagnosis = ['normal','Neuroblastoma', 'Ganglioneuroma']
tool = SVGTools();
for diag in diagnosis:
    ids_images = glob.glob(os.path.join(r"\\chubaka\home\florian.kromp\settings\desktop\nucleusanalyzer\FFG COIN VISIOMICS\Ongoing\Image groundtruth curation",diag,'*_svg.svg'))
    for img_path in ids_images:
        mask = tool.transformSVGToMaskNew(img_path)
        tifffile.imsave(img_path.replace('_svg.svg','_mask.tif'), mask)

#plt.imshow(tool.transformSVGToMaskNew(r"\\chubaka\home\florian.kromp\settings\desktop\nucleusanalyzer\FFG COIN VISIOMICS\Ongoing\Image groundtruth curation\Neuroblastoma\4471_svg.svg"))