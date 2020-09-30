from Classes.DBTools import TisQuantExtract
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
    args = parser.parse_args()
    tisquant = TisQuantExtract()
    config = Config
    if args.tissue:
        config.diagnosis = [args.tissue]

    if args.outputFolder:
        config.outputFolder = args.outputFolder

    print(config.diagnosis)
    tools = Tools()
    svg_tools = SVGTools(samplingrate=args.samplingrate)

    ids_paths = tisquant.dbconnector.execute(query=tisquant.getLevel3AnnotatedImagesByDiagnosis_Query(diagnosis = config.diagnosis,magnification = config.magnification, staining_type = config.staining_type, staining = config.staining, segmentation_function = config.segmentation_function, annotator = config.annotator, device = config.device))

    for index,elem in enumerate(ids_paths):
        groundtruth_path_l3 = tisquant.dbconnector.execute(tisquant.getLevel3AnnotationByImageIdUsingMaxExperience_Query(elem[0], config.annotator))[0]
        groundtruth_path_l2 = tisquant.dbconnector.execute(tisquant.getLevel2AnnotationByImageIdUsingMaxExperience_Query(elem[0], config.annotator))[0]
        #copyfile(tools.getLocalDataPath(elem[1], 1),os.path.join(config.outputFolder, config.diagnosis[0], str(elem[0]) + '.tif'))
        img = AnnotatedImage()
        img.readFromPath(tools.getLocalDataPath(elem[1], 1), tools.getLocalDataPath(groundtruth_path_l2[0], 3))
        cv2.imwrite(os.path.join(config.outputFolder, config.diagnosis[0], str(elem[0]) + '_raw.jpg'),(img.getRaw() * 255.0).astype(np.uint8))
        svg_tools.openSVG(img.getRaw().shape[0],img.getRaw().shape[1])
        #svg_tools.addRawImage(name='Raw image',img_path=os.path.join(config.outputFolder, config.diagnosis[0], str(elem[0]) + '_raw.jpg'))
        svg_tools.addRawImage(name='Raw image', img_path=(str(elem[0]) + '_raw.jpg'))
        svg_tools.addMaskLayer(img.getMask()[:,:,0], 'Not annotated', '#0000FF', 0.5)
        svg_tools.addMaskLayer(img.getMask()[:, :, 2], 'Clumps', '#FF0000', 0.5)
        img.readFromPath(tools.getLocalDataPath(elem[1], 1), tools.getLocalDataPath(groundtruth_path_l3[0], 3))
        svg_tools.addMaskLayer(img.getMask(),'Single nuclei','#00FF00',0.5)
        svg_tools.closeSVG()

        svg_tools.writeToPath(os.path.join(config.outputFolder, config.diagnosis[0], str(elem[0]) + '_svg.svg'))
        #tools.writeSVGToPath(os.path.join(config.outputFolder, config.diagnosis[0], str(elem[0]) + '_svg.svg'),img.getSVGMask(img_path=os.path.join(config.outputFolder, config.diagnosis[0], str(elem[0]) + '_raw.jpg')))
main()