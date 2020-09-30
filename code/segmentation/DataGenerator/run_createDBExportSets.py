from Classes.DBTools import TisQuantExtract
from Classes.Config import Config
from Classes.Helper import Tools
from Classes.Image import AnnotatedImage,AnnotatedObjectSet, ArtificialAnnotatedImage
from matplotlib import pyplot as plt
from shutil import copyfile
import argparse
import os
import random

def main():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--tissue', help='select tissue to train.', default=None)
    parser.add_argument('--outputFolder', help='select output folder', default=None)
    args = parser.parse_args()
    tisquant = TisQuantExtract()
    config = Config
    if args.tissue:
        config.diagnosis = [args.tissue]

    if args.outputFolder:
        config.outputFolder = args.outputFolder

    print(config.diagnosis)
    print(config.outputFolder)
    tools = Tools()

    annotated_nuclei = AnnotatedObjectSet()
    for fold in ['train','val','test','train_and_val']:
        path_train = os.path.join(config.outputFolder,fold)
        if not os.path.exists(path_train):
            os.makedirs(path_train)
        path_output = os.path.join(config.outputFolder,fold,config.diagnosis[0])
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        path_output_images = os.path.join(config.outputFolder,fold,config.diagnosis[0],'images')
        if not os.path.exists(path_output_images):
            os.makedirs(path_output_images)
        path_output_masks = os.path.join(config.outputFolder,fold,config.diagnosis[0],'masks')
        if not os.path.exists(path_output_masks):
            os.makedirs(path_output_masks)

    # Create dataset for training the pix2pix-network based on image pairs
    fold = 'train'
    running_ind = 0

    ids_paths = tisquant.dbconnector.execute(query=tisquant.getLevel3AnnotatedImagesByDiagnosis_Query(diagnosis = config.diagnosis,magnification = config.magnification, staining_type = config.staining_type, staining = config.staining, segmentation_function = config.segmentation_function, annotator = config.annotator, device = config.device))
    random.shuffle(ids_paths)
    
    nr_trainval = round(ids_paths.__len__()*0.8)
    nr_val = round(nr_trainval*0.8)

    for index,elem in enumerate(ids_paths):

        groundtruth_paths = tisquant.dbconnector.execute(tisquant.getLevel3AnnotationByImageIdUsingMaxExperience_Query(elem[0], config.annotator))
        if (index < nr_val):
            folds = ['train','train_and_val']
        elif (index < nr_trainval):
            folds = ['val','train_and_val']
        else:
            folds = ['test']
        for fold in folds:
            path_output_images = os.path.join(config.outputFolder,fold,config.diagnosis[0],'images')
            path_output_masks = os.path.join(config.outputFolder, fold, config.diagnosis[0], 'masks')
            for groundtruth_path in groundtruth_paths:
                copyfile (tools.getLocalDataPath(elem[1],1),os.path.join(path_output_images,config.diagnosis[0] + '_' + str(running_ind) + '.tif'))
                print ('Copying rawimage ' + tools.getLocalDataPath(elem[1],1) + '...')
                copyfile (tools.getLocalDataPath(groundtruth_path[0],3),os.path.join(path_output_masks,config.diagnosis[0] + '_' + str(running_ind) + '.tif'))
                print('Copying mask ' + tools.getLocalDataPath(groundtruth_path[0],3) + '...')
        running_ind = running_ind + 1


main()