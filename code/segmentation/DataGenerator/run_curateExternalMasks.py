from Classes.Config import Config
from Classes.Helper import Tools,SVGTools
from Classes.Image import AnnotatedImage,AnnotatedObjectSet, ArtificialAnnotatedImage
from matplotlib import pyplot as plt
import scipy.misc
import random
import numpy as np
from tifffile import tifffile
import argparse
import glob
import os
from random import randint
import matplotlib.pyplot as plt
from shutil import copyfile
import cv2

def main():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--tissue', help='select tissue to train.', default=None)
    parser.add_argument('--inputFolder', help='Select input folder.', default=None)
    parser.add_argument('--outputFolder', help='select output folder', default=None)
    parser.add_argument('--nr_images', help='select number of images to create', default=None)
    parser.add_argument('--overlapProbability', help='select overlapProbability', default=None)
    parser.add_argument('--samplingrate', help='how fine the contour shall be sampled', default=None)
    parser.add_argument('--ending', help='how fine the contour shall be sampled', default=None)
    args = parser.parse_args()

    config = Config
    if args.tissue:
        config.diagnosis = [args.tissue]

    if args.outputFolder:
        config.outputFolder = args.outputFolder

    #print(config.diagnosis)
    print(args.ending)
    print(args.inputFolder)
    tools = Tools()
    svg_tools = SVGTools(samplingrate=int(args.samplingrate))
    folder=args.inputFolder #=r"\\chubaka\home\florian.kromp\settings\desktop\nucleusanalyzer\FFG COIN VISIOMICS\Ongoing\EvaluationMetrics\HaCat_grown\10"
    print(os.path.join(folder,"*[!mask]."+args.ending))
    #images = glob.glob(os.path.join(folder,"*[!mask]."+args.ending))
    images = glob.glob(os.path.join(folder,"*."+args.ending))

    masks = [x.replace('.' + os.path.basename(x).split('.')[1],"_mask.TIF") for x in images] 
    print (images)
    print(masks)
    for index,elem in enumerate(images):

        img = AnnotatedImage()
        img.readFromPath(images[index], masks[index])
        cv2.imwrite(os.path.join(folder, os.path.basename(elem).replace('.'+os.path.basename(elem).split('.')[1],'_raw.jpg')),(img.getRaw() * 255.0).astype(np.uint8))
        svg_tools.openSVG(img.getRaw().shape[0],img.getRaw().shape[1])
        svg_tools.addRawImage(name='Raw image', img_path=(os.path.basename(elem).replace('.'+os.path.basename(elem).split('.')[1],'_raw.jpg')))
        svg_tools.addMaskLayer(img.getMask(),'Single nuclei','#00FF00',0.5)
        svg_tools.closeSVG()


        print("Path to SVG: " + os.path.join(folder, os.path.basename(elem).replace('.' + os.path.basename(elem).split('.')[1],'_svg.svg')))
        svg_tools.writeToPath(os.path.join(folder, os.path.basename(elem).replace('.' + os.path.basename(elem).split('.')[1],'_svg.svg')))
        #tools.writeSVGToPath(os.path.join(config.outputFolder, config.diagnosis[0], str(elem[0]) + '_svg.svg'),img.getSVGMask(img_path=os.path.join(config.outputFolder, config.diagnosis[0], str(elem[0]) + '_raw.jpg')))
main()