from Classes.Config import Config
from Classes.Helper import Tools
from Classes.Image import AnnotatedImage,AnnotatedObjectSet, ArtificialAnnotatedImage
from matplotlib import pyplot as plt
import os
import argparse
import glob
import pandas
import csv
import numpy as np
from Classes.Image import Image
from tifffile import imread, imsave
import h5py
import pickle
import cv2

def main():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--scale', help='select output folder', default=None)
    parser.add_argument('--resultfile', help='select result file', default=None)
    parser.add_argument('--predictionfile', help='select result file', default=None)
    parser.add_argument('--net', help='describe net', default=None)
    parser.add_argument('--overlap', help='select output folder', default=None)
    parser.add_argument('--dilate', help='select output folder', default=False)
    args = parser.parse_args()

    config = Config

    if args.scale == '1':
        config.scale=True
    if args.resultfile:
        config.resultfile=args.resultfile
    else:
        print("No result file provided")
        exit()
    if args.net:
        config.net=args.net
    if args.overlap:
        config.overlap=int(args.overlap)

    path_to_img = []
    tiles = []
    images = []
    scales = []
    scales_new = []
    path_to_save = []
    start=1
    with open(args.resultfile) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            path_to_img.append(row[0])
            if start==1:
                path_to_save.append(row[0])
                start=0
            else:
                if path_to_save[-1] != row[0]:
                    path_to_save.append(row[0])
            scales.append(float(row[1]))
            tiles.append(int(row[2]))

    print(config.scale)
    tools = Tools()
    print ("Loading predictions ...")
    predictions = h5py.File(args.predictionfile, 'r')['predictions']

    tile_ind = 0
    for i in range (0, tiles.__len__()):
        if (tiles[i] == 0):
            if (os.path.basename(path_to_img[i]).split('.')[1] == 'tif') or (os.path.basename(path_to_img[i]).split('.')[1] == 'TIF'):
                img = imread(path_to_img[i])
            else:
                img = cv2.imread(path_to_img[i])
            images.append(Image.pre_process_img(img, color='gray'))
            scales_new.append(scales[i])
    # Create and save the reconstructed images
    print ("Reconstruct images ...")
    reconstructed_predictions, reconstructed_masks = tools.reconstruct_images(images=images,predictions=predictions,scales=scales_new,rescale=config.scale,overlap=config.overlap,config=config,label_output=True,dilate_objects=int(args.dilate))
    for index,i in enumerate(reconstructed_masks):
        print(path_to_save[index].replace('.TIF','_mask.TIF'))
        #imsave(path_to_save[index].replace('.TIF','_mask.TIF'),i)        
        #imsave(path_to_save[index].replace('.tif','_mask.TIF'),i)
        print (path_to_save[index].replace('.' + os.path.basename(path_to_save[index]).split('.')[1],'_mask.TIF'))
        imsave(path_to_save[index].replace('.' + os.path.basename(path_to_save[index]).split('.')[1],'_mask.TIF'),i)
    #pickle.dump(({"masks":reconstructed_masks, "predictions":reconstructed_predictions}),open(os.path.join(os.path.dirname(args.predictionfile), os.path.basename(args.predictionfile).replace('.h5','_reconstructed.pkl')),"wb"))
main()