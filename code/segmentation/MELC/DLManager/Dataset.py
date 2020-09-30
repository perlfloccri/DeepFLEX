# --------------------------------------------------------
# Multi-Epitope-Ligand Cartography (MELC) phase-contrast image based segmentation pipeline
#
#
# Written by Filip Mivalt
# --------------------------------------------------------

import os
import cv2
import json
import pandas as pd
import numpy as np
import MELC.utils.Files as myF

import tifffile
from MELC.utils.ptr import Pointer
from os.path import join, isfile, exists
from copy import deepcopy
from MELC.utils.JSON_coco import JSONDataset
from MELC.Client.Dataset import MELCSynthesiser

import json
from MELC.utils.JSON_coco import JSONDataset
import numpy as np
from tifffile import imread, imsave
import pickle
import MELC.utils.Files as myF
from tqdm import tqdm

import os

from config import *


import pycocotools.mask as mask

class SuperMELCDataset:
    # toto bude dataset, ktery se podiva, kolik je runnu, vsechny je spocita.
    # pro kazdy run zkontroluje, jestli jsou nejake anotace pro training, pro testing
    # inicalizuje datasety
    # ulozi data pro trenovani kompatabilne do json a do slozky, ze ktere budem trenovat
    # ai a real data se budou generovat zvlast
    # training se budou augmentovat, test ne

    def __init__(self, raw_dataset_path, annotations_path):
        #        'C:\Users\Public\Filip\Projects\MELC-Phase Contrast\data\annotations_SVG'

        runs_path = raw_dataset_path
        train_path = join(annotations_path, 'train')
        test_path = join(annotations_path, 'test')
        validation_path = join(annotations_path, 'validation')

        folder_runs_items = os.listdir(runs_path)
        folder_train_items = os.listdir(train_path)
        folder_test_items = os.listdir(test_path)
        folder_validation_items = os.listdir(validation_path)

        runs = dict()
        for k in range(len(folder_runs_items)):
            if not 'Thumbs' in folder_runs_items[k]:
                current_run = folder_runs_items[k]
                # print(current_run)
                runs[current_run] = dict()
                runs[current_run]['path'] = join(raw_dataset_path, current_run)

                runs[current_run]['train'] = dict()
                runs[current_run]['train']['exists'] = False
                runs[current_run]['train']['path'] = ''
                # runs[current_run]['train']['dataset'] = None
                for train_folder in folder_train_items:
                    if current_run in train_folder:
                        runs[current_run]['train']['exists'] = True
                        runs[current_run]['train']['path'] = join(train_path, train_folder)
                        # print('train')
                        # runs[current_run]['train']['dataset'] = MELCSynthesiser(runs[current_run]['path'], runs[current_run]['train']['path'])

                runs[current_run]['test'] = dict()
                runs[current_run]['test']['exists'] = False
                runs[current_run]['test']['path'] = ''
                # runs[current_run]['test']['dataset'] = None
                for test_folder in folder_test_items:
                    if current_run in test_folder:
                        runs[current_run]['test']['exists'] = True
                        runs[current_run]['test']['path'] = join(test_path, test_folder)
                        # print('test')
                        # runs[current_run]['test']['dataset'] = MELCSynthesiser(runs[current_run]['path'], runs[current_run]['test']['path'])

                runs[current_run]['validation'] = dict()
                runs[current_run]['validation']['exists'] = False
                runs[current_run]['validation']['path'] = ''
                # runs[current_run]['validation']['dataset'] = None
                for validation_folder in folder_validation_items:
                    if current_run in validation_folder:
                        runs[current_run]['validation']['exists'] = True
                        runs[current_run]['validation']['path'] = join(validation_path, validation_folder)
                        # print('validation')
                        # runs[current_run]['validation']['dataset'] = MELCSynthesiser(runs[current_run]['path'], runs[current_run]['validation']['path'])

        self.runs = runs

    def generate_validation(self, path_images='', path_json=''):
        try:
            myF.remove_folder(path_images)
        except:
            pass
        myF.create_folder(path_images)


        self.JSONDataset_val = JSONDataset(path_images)

        domain = 'validation'
        json_path = join(path_json, 'instances_val' + '.json')
        if exists(json_path):
            myF.remove_folder(json_path)

        number_annotated_runs = 0
        for cur_run in self.runs.keys():
            if self.runs[cur_run][domain]['exists']: number_annotated_runs += 1

        cntr = -1
        print('Generating validation data')
        for cur_run in tqdm(self.runs.keys()):
            if self.runs[cur_run][domain]['exists']:
                cntr += 1
                self.runs[cur_run][domain]['path']
                print(cur_run)
                Dataset = MELCSynthesiser(self.runs[cur_run]['path'], self.runs[cur_run][domain]['path'])

                for k in range(Dataset.__len__()):
                    image, annotations, fid = Dataset[k]
                    fid = 'Real_' + fid
                    fid_real = fid
                    self.JSONDataset_val.add_image(image, annotations, fid)

        print(json_path)
        self.JSONDataset_val.write_json(json_path)

        ############################33# TRAIN ###################################

    def generate_train(self, path_images='', path_json='', real=True, augmented=0, synthetic=0):
        try:
            myF.remove_folder(path_images)
        except:
            pass
        myF.create_folder(path_images)

        self.JSONDataset_train = JSONDataset(path_images)

        domain = 'train'
        json_path = join(path_json, 'instances_' + domain + '.json')

        if exists(json_path):
            myF.remove_folder(json_path)


        number_annotated_runs = 0
        print('Generate training data')
        for cur_run in self.runs.keys():
            if self.runs[cur_run][domain]['exists']: number_annotated_runs += 1

        distribution_augmented = np.zeros(number_annotated_runs)
        distribution_synthetic = np.zeros(number_annotated_runs)

        for k in range(number_annotated_runs):
            distribution_augmented[k] = np.round(augmented / number_annotated_runs)
            distribution_synthetic[k] = np.round(synthetic / number_annotated_runs)
            if k == number_annotated_runs - 1:
                distribution_augmented[k] = augmented - (
                        k * np.round(augmented / number_annotated_runs))

                distribution_synthetic[k] = synthetic - (
                        k * np.round(augmented / number_annotated_runs))

        # distribution is vector with length of the number of annotated runs, if there are 4
        # it is [2, 2, 2, 4]
        # it means that it generates 2 images from first run, .... and 4 for the last run, last run takes all rounding innacuracies

        cntr = -1
        for cur_run in tqdm(self.runs.keys()):
            if self.runs[cur_run][domain]['exists']:
                cntr += 1
                self.runs[cur_run]['train']['path']
                print(cur_run)
                Dataset = MELCSynthesiser(self.runs[cur_run]['path'], self.runs[cur_run]['train']['path'])

                print('Real images')
                for k in range(Dataset.__len__()):
                    image, annotations, fid = Dataset[k]
                    fid = 'Real_' + fid
                    fid_real = fid
                    if real:
                        self.JSONDataset_train.add_image(image, annotations, fid)

                if augmented > 0:
                    print('Augmented real images')
                    for k in range(int(distribution_augmented[cntr])):
                        fid = f'{k:06d}'
                        image, annotations, fid2 = Dataset.generate_augmented_real()
                        fid = 'Augmented_' + fid + '_' + fid2
                        self.JSONDataset_train.add_image(image, annotations, fid)

                if synthetic > 0:
                    print('Synthetic images')
                    for k in range(int(distribution_synthetic[cntr])):
                        fid = f'{k:06d}'
                        fid = 'Synthetic_' + fid_real + '_' + fid
                        image, annotations = Dataset.generate_synthetic_image((256, 256))
                        self.JSONDataset_train.add_image(image, annotations, fid)

        self.JSONDataset_train.write_json(json_path)


class DLPredictionImage:

    def __init__(self, path_prediction_pickle):
        self.overlap_threshold = 0.5
        self.confidence_threshold = 0.5  # prediction confidence threshold
        self.confidence_threshold = 0.5  # prediction confidence threshold

        self.predictions = self.load_predictions(path_prediction_pickle)

        num_pred_before = np.Inf
        num_pred_after = 0
        while not num_pred_before == num_pred_after:
            num_pred_before = self.predictions.__len__()
            self.mutual_matrix, self.distance_matrix = self.get_cell_distance_matrices()  # from predictions
            # estimates mutual_matrix, at first -> compares radius of 2 cells and their distance, if R1+R2 < D, there is
            # probability, they are overlapping. This cells are checked, if at least one point of one bb2 lies in bb1
            # if yes, then the percentage overlap of the cells is computed and inserted into the distance matrix
            # pos[k1, k2] is percentage of overlap for k1 and pos[k2, k1] denotes % overlap for k2 cell

            # there are values of metric 1 with radius, however, it is already processed, thus not neccesary for next
            # run of the algorithm mutual matrix

            self.cell_list = self.create_dependencies_from_matrix(self.distance_matrix)
            # creates dictionary of cells, which are in some ration of overlap

            self.clump_parents = self.get_clump_parents(self.cell_list)
            # returns the list of indexes of all cells which has only children and no parents - are on the top of the trees

            self.clump_child_index, self.clump_child_deepness, self.clump_cells_index = self.create_clumps(
                self.cell_list, self.clump_parents)

            self.clump_child_index, self.clump_child_deepness, self.clump_cells_index = self.concatenate_relating_clumps()
            # output - list of lists; each field in the top list denotes one cell - clump:
            #           cells in some realtion in the matter of overlaps
            #
            # clump_child_index - list of lists containing children of each clump. there are presented only children
            #                       without any other children

            # clump_cells_index - deepness for each corresponding child in the clump_child_index
            #                       represents the length of the path in the graph from parent to the child

            # list_clump_index - represents the list of all cells in the clump

            self.mark_removal_candidates()
            self.remove_marked()

            num_pred_after = self.predictions.__len__()

    def remove_marked(self):
        # erase them
        k1 = 0
        while k1 < self.predictions.__len__():
            if self.predictions[k1]['solved'] == True:
                del self.predictions[k1]
                k1 = 0
            else:
                k1 += 1

    def mark_removal_candidates(self):
        # match the lowest confidence in clumps to erase
        for cell_clump in self.clump_cells_index:
            cell_confidences = []
            for cell in cell_clump:
                cell_confidences.append(self.predictions[cell]['prediction'])
            pos_min = np.argmin(cell_confidences)
            self.predictions[cell_clump[pos_min]]['solved'] = True

    def concatenate_relating_clumps(self):
        clump_child_index = self.clump_child_index
        clump_child_deepness = self.clump_child_deepness
        clump_cells_index = self.clump_cells_index

        break_wheel = True
        if clump_child_index.__len__() < 3:
            break_wheel = False

        while break_wheel:
            break_bool = False
            for k1 in range(clump_cells_index.__len__() - 1):
                if break_bool: break
                for k2 in range(1 + k1, clump_cells_index.__len__()):
                    if break_bool: break
                    for cell in clump_cells_index[k2]:
                        if cell in clump_cells_index[k1]:
                            clump_cells_index[k1] = clump_cells_index[k1] + clump_cells_index[k2]
                            clump_child_deepness[k1] = clump_child_deepness[k1] + clump_child_deepness[k2]
                            clump_child_index[k1] = clump_child_index[k1] + clump_child_index[k2]
                            del clump_cells_index[k2]
                            del clump_child_deepness[k2]
                            del clump_child_index[k2]

                            break_bool = True
                            break

                            # return
                            # merge k2 to k1 and erase k2
                    # print(k1, k2)
                    if k1 == clump_cells_index.__len__() - 2 and k2 == clump_cells_index.__len__() - 1:
                        break_wheel = False

        return clump_cells_index, clump_child_deepness, clump_cells_index

    def create_clumps(self, cell_list, clump_parents):
        index_list_clumps = []
        deepness_list_clumps = []
        cell_list_clumps = []
        for parent_idx in clump_parents:
            index_temp, deepness_temp, all_clump_indexes = self.get_child_of_childs(cell_list, parent_idx)
            index_list_clumps.append(index_temp)
            deepness_list_clumps.append(deepness_temp)
            cell_list_clumps.append(list(np.unique(all_clump_indexes)))

        return index_list_clumps, deepness_list_clumps, cell_list_clumps

    def load_predictions(self, path_prediction_pickle):
        # loads prediction pickle file
        objects = []
        with (open(path_prediction_pickle, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break

        objects = objects[0]
        segments = objects['all_segms']
        bboxes = objects['all_boxes'][1]

        predictions = []
        for k in range(segments.__len__()):
            segm = segments[k]
            bbox = bboxes[k]

            segm = np.array(segm)
            segm = segm.reshape(segm.size)
            contour = np.zeros((int(segm.shape[0] / 2), 1, 2))
            contour[:, 0, 0] = segm[::2]
            contour[:, 0, 1] = segm[1::2]
            contour = contour.astype(np.int64)



            bbox_contour = np.zeros((4, 1, 2))
            bbox_contour[0, 0, :] = np.array([bbox[0], bbox[1]]).round()
            bbox_contour[1, 0, :] = np.array([bbox[2], bbox[1]]).round()
            bbox_contour[2, 0, :] = np.array([bbox[2], bbox[3]]).round()
            bbox_contour[3, 0, :] = np.array([bbox[0], bbox[3]]).round()
            bbox_contour = bbox_contour.astype(np.int64)

            bbox_orig = np.array(bbox)[:-1]
            center = np.array([0.5 * (bbox_orig[0] + bbox_orig[2]), 0.5 * (bbox_orig[1] + bbox_orig[3])])

            bbox_orig[2] = bbox_orig[2] - bbox_orig[0]
            bbox_orig[3] = bbox_orig[3] - bbox_orig[1]
            stick1 = bbox_orig[2] / 2
            stick2 = bbox_orig[3] / 2
            stick = np.array([stick1, stick2])

            radius = np.sqrt(bbox_orig[3] ** 2 + bbox_orig[2] ** 2)

            prediction = {
                'contour': contour,
                'bbox': bbox_orig,
                'bbox_contour': bbox_contour,
                'prediction': bbox[4],
                'center': center,
                'radius': radius,
                'stick': stick,
                'solved': False,
            }

            if prediction['prediction'] > self.confidence_threshold and contour.shape[0] > 5: # CONTROL WHETHER WORKS AS IT SHOULD
                predictions.append(prediction)

        return predictions

    def get_cell_distance_matrices(self):
        predictions = self.predictions

        mutual_matrix = np.zeros((predictions.__len__(), predictions.__len__()))
        distance_matrix = np.zeros((predictions.__len__(), predictions.__len__()))

        for k1 in range(mutual_matrix.shape[0]):
            for k2 in range(k1):
                pred1 = predictions[k1]
                pred2 = predictions[k2]

                distance = np.sqrt(((pred1['center'][0] - pred2['center'][0]) ** 2 + \
                                    (pred1['center'][1] - pred2['center'][1]) ** 2))
                Radius_b = pred1['radius'] + pred2['radius']
                temp_dist = distance - Radius_b  # where < 0 - potential overlap

                mutual_matrix[k1, k2] = temp_dist

                if temp_dist < 0:  # there are close each other they could overlap
                    square_overlap = False
                    pts_p1 = pred1['center'] + pred1['stick']
                    pts_m1 = pred1['center'] - pred1['stick']

                    pts_p2 = pred2['center'] + pred2['stick']
                    pts_m2 = pred2['center'] - pred2['stick']

                    for k3 in range(len(pred2['bbox_contour'])):
                        pt1 = pred2['bbox_contour'][k3, 0, :]
                        pt2 = pred1['bbox_contour'][k3, 0, :]
                        if pt1[0] <= pts_p1[0] and pt1[0] >= pts_m1[0] and pt1[1] <= pts_p1[1] and pt1[1] >= pts_m1[1]:
                            square_overlap = True
                        if pt2[0] <= pts_p2[0] and pt2[0] >= pts_m2[0] and pt2[1] <= pts_p2[1] and pt2[1] >= pts_m2[1]:
                            square_overlap = True

                    if square_overlap == False:
                        temp_shape = max([pred1['bbox_contour'].max(), pred2['bbox_contour'].max()])
                        temp_img1 = np.zeros((temp_shape, temp_shape), dtype=np.uint8)
                        temp_img2 = np.zeros((temp_shape, temp_shape), dtype=np.uint8)

                        temp_img1 = cv2.drawContours(temp_img1, [pred1['contour']], -1, 1, -1)
                        temp_img2 = cv2.drawContours(temp_img2, [pred2['contour']], -1, 1, -1)
                        if (temp_img1 * temp_img2).sum() > 0:
                            square_overlap = True

                    # if square_overlap is TRUE, then at least 1 point of the prediction 1 lies in the bbox of prediction 2
                    # there is even higher possibility of mask overlap

                    if square_overlap:  # now compare masks and compute real mask overlap
                        img_shape = max([pred1['contour'].max(), pred2['contour'].max()])
                        mask1 = np.zeros((img_shape, img_shape), dtype=np.uint8)
                        mask2 = np.zeros((img_shape, img_shape), dtype=np.uint8)
                        mask1 = cv2.drawContours(mask1, [pred1['contour']], -1, 1, -1)
                        mask2 = cv2.drawContours(mask2, [pred2['contour']], -1, 1, -1)

                        #mask1 = cv2.drawContours(mask1, [pred1['bbox_contour']], -1, 1, 1)
                        #mask2 = cv2.drawContours(mask2, [pred2['bbox_contour']], -1, 1, 1)

                        mask_sum = mask1 + mask2
                        mask_overlap = mask_sum
                        mask_overlap[mask_overlap == 1] = 0
                        mask_overlap[mask_overlap == 2] = 1

                        mask1_overlap = mask_overlap * mask1
                        mask2_overlap = mask_overlap * mask2

                        ovlp1_val = mask1_overlap.sum() / mask1.sum()
                        ovlp2_val = mask2_overlap.sum() / mask2.sum()

                        distance_matrix[k1, k2] = ovlp1_val
                        distance_matrix[k2, k1] = ovlp2_val

        return mutual_matrix, distance_matrix

    def create_dependencies_from_matrix(self, distance_matrix):
        cell_list = dict()

        ## BODY OF THE FUNCTION
        for k1 in range(self.predictions.__len__()):
            for k2 in range(k1):
                if distance_matrix[k1, k2] > 0 and distance_matrix[k2, k1] > 0:
                    cell_list = self.add_cell_dependence(cell_list, k1, k2, distance_matrix[k1, k2],
                                                         distance_matrix[k2, k1])
                    # adding is checked in the function and only overlaps higher than threshold are considered

        return cell_list

    def add_cell_dependence(self, cell_list, k1, k2, dist12, dist21):
        if max([dist12, dist21]) > self.overlap_threshold:  # only overlaps higher than 0.5 are considered

            if dist12 <= dist21:  # then k1 is dominant - has lower amount of overlap; k1 is parent of k2
                dominant = k1
                recessive = k2
            else:
                dominant = k2
                recessive = k1

            dom_str = str(dominant)
            rec_str = str(recessive)

            # add child, into parent, if exists. if not, create parent
            if dom_str in cell_list.keys():  # if parent do not exist, create
                cell_list[dom_str]['child'].append(recessive)
            else:
                cell_temp = {
                    'parent': list(),
                    'child': [recessive],
                    'index': dominant
                }
                cell_list[dom_str] = cell_temp

            # add parent to child, if child do not exist -> create
            if rec_str in cell_list.keys():
                cell_list[rec_str]['parent'].append(dominant)
            else:
                cell_temp = {
                    'parent': [dominant],
                    'child': [],
                    'index': recessive
                }
                cell_list[rec_str] = cell_temp

        return cell_list

    def get_clump_parents(self, cell_list):
        # find all parents
        clump_parents = []
        for key in cell_list.keys():
            cell = cell_list[key]
            if cell['parent'].__len__() == 0:
                clump_parents.append(cell['index'])

        return clump_parents

    def get_child_of_childs(self, cell_list, parent_id, index_list=[], deepness_list=[], all_clump_indexes=[],
                            deepness=-1):  #
        index_list = deepcopy(index_list)
        deepness_list = deepcopy(deepness_list)
        all_clump_indexes = deepcopy(all_clump_indexes)
        # recursively goes into all corners of the tree/graph and returns, deepness index and child of childs, where
        # I can start overlap solving
        # doesn't count with possibility of infinite loop
        deepness += 1
        parent_str = str(parent_id)
        parent = cell_list[parent_str]
        all_clump_indexes.append(parent['index'])

        if parent['child'].__len__() > 0:
            for child_idx in parent['child']:
                # if multiple childs, then find childs for all of them
                temp_index, temp_deepness, all_clump_indexes = self.get_child_of_childs(cell_list, child_idx,
                                                                                        index_list,
                                                                                        deepness_list,
                                                                                        all_clump_indexes,
                                                                                        deepness)

                if type(temp_index) == type(list()):
                    for k in range(len(temp_index)):
                        # in case there is already array, then extends array and
                        deepness_list.append(temp_deepness[k])
                        index_list.append(temp_index[k])

                else:
                    # if return was only a number, appends the number into a list
                    deepness_list.append(temp_deepness)
                    index_list.append(temp_index)

            return index_list, deepness_list, all_clump_indexes

        else:
            # if there are no other childs, returns index and deepness
            index = parent['index']
            return index, deepness, all_clump_indexes


class DLPredictionInference:

    def __init__(self, path_predictions, path_images=''):
        self.path_predictions = path_predictions
        self.path_images = path_images

        pfiles = myF.get_files(path_predictions, ('pkl'))
        ifiles = myF.get_files(path_images, ('tif'))

        files_pd = pd.DataFrame(pfiles)

        def get_FID(x):
            """
            separates file id from the MELC Run path

            :param x: pandas.DataFrame (row)
            :return fid: str
            """
            temp = x['prediction'].split(SEPARATOR)
            return temp[len(temp) - 1][0:-4]

        files_pd = files_pd.rename(columns={0: "prediction"})
        files_pd['fid'] = files_pd.apply(lambda x: get_FID(x), axis=1)
        files_pd['image'] = ""
        for k1 in range(files_pd.__len__()):
            for k2 in range(files_pd.__len__()):
                if files_pd['fid'][k1] in ifiles[k2]:
                    files_pd.at[k1, 'image'] = ifiles[k2]
                    break

        self.files_pd = files_pd

    def __len__(self):
        return self.files_pd.__len__()

    def __getitem__(self, item):
        path_im = self.files_pd['image'][item]
        path_pred = self.files_pd['prediction'][item]
        image = imread(path_im)
        predObj = DLPredictionImage(path_pred)
        return {'image': image, 'predictions': predObj.predictions}

        # find all predictions, find matching files, input is path to the pickle predictions and to the image files used for prediction

    def print_contours(self, item):
        pred = self.__getitem__(item)
        img = pred['image']
        if len(img) > 0:
            predictions = pred['predictions']
            img = img - img.min()
            img = img / img.max()

            nimg = np.zeros((img.shape[0], img.shape[1], 3))
            nimg[:, :, 0] = img
            nimg[:, :, 1] = img
            nimg[:, :, 2] = img
            img = nimg

            predImg = np.zeros(img.shape, dtype=np.uint8)

            for pred in predictions:
                predImg = cv2.drawContours(predImg, [pred['contour']], -1,
                                           (np.random.randint(255), np.random.randint(255), np.random.randint(255)), 1)

            predImg = predImg.astype(np.float64) / 255
            imS = deepcopy(img)
            imS[predImg > 0] = predImg[predImg > 0]
            return imS



class DLPredictionTest(DLPredictionInference):

    def __init__(self, path_predictions, path_annotations, path_images):
        super(DLPredictionTest, self).__init__(path_predictions, path_images)
        self.path_annotations = path_annotations

        fid = open(path_annotations, "rb")
        annot_txt = fid.read()
        fid.close()
        print(annot_txt)
        annot_json = json.loads(annot_txt)
        self.AnnotationDataset = JSONDataset.fromjson(annot_json, path_images)

        self.pxScoreEval = ScoreCounter()
        self.objScoreEval = ScoreCounter()
        self.evaluateDataset()


        # annot_contour = annotD.ImagesObj[idx].COCO_annotations[0].contour
        # annot_bbox = annotD.ImagesObj[idx].COCO_annotations[0].bbox

        # uses annotation file, as shown in mainWin, somewhere lower.
        # create annotation and to each JSON annotation just add matching prediction
        # thus for each image may be implemented desired metric

    def __getitem__(self, item):


        path_im = self.files_pd['image'][item]
        path_pred = self.files_pd['prediction'][item]
        image = imread(path_im)
        predObj = DLPredictionImage(path_pred)

        fid = path_im.split(SEPARATOR)[-1].split('.')[0]
        if fid in self.AnnotationDataset.ImagesObj[item].file_name:
            return {
                'image': image,
                'predictions': predObj.predictions,
                'annotations': self.AnnotationDataset.ImagesObj[item]
            }
        else:
            for ImgCoco in self.AnnotationDataset.ImagesObj:
                if fid in ImgCoco.file_name:
                    return {
                        'image': image,
                        'predictions': predObj.predictions,
                        'annotations': ImgCoco
                    }

    def evaluate_sample(self, item):
        temp_outp = self.__getitem__(item)
        img = temp_outp['image']
        preds = temp_outp['predictions']
        annots = temp_outp['annotations']

        predictions = []
        for prediction in preds:
            temp_dict = {
                'contour': prediction['contour'],
                'bbox':  prediction['bbox'],
                'img_shape': img.shape
            }
            predictions.append(temp_dict)

        annotations = []
        for COCO_annot in annots.COCO_annotations:
            temp_dict = {
                'contour': COCO_annot.contour.astype(np.int64),
                'bbox': COCO_annot.bbox,
                'img_shape': img.shape
            }
            annotations.append(temp_dict)

        predImg = DLPredictionTestImage(annotations, predictions)

        self.objScoreEval.OS += predImg.OS
        self.objScoreEval.US += predImg.US
        self.objScoreEval.TP += predImg.TP
        self.objScoreEval.FP += predImg.FP
        self.objScoreEval.FN += predImg.FN

        self.pxScoreEval.TP += predImg.TPpx
        self.pxScoreEval.FP += predImg.FPpx
        self.pxScoreEval.FN += predImg.FNpx

    def evaluateDataset(self):
        for k in tqdm(range(self.__len__())):
            self.evaluate_sample(k)

    def get_scores(self):
        return {
            'object': self.objScoreEval.get_obj_scores(),
            'pixel': self.pxScoreEval.get_px_scores()
        }


class DLPredictionTestImage:

    def __init__(self, annotations, predictions):
        # FOR OBJECT LEVEL

        self.Annotations = self.get_credentials(annotations)
        self.Predictions = self.get_credentials(predictions)
        # I tried to filterout predictions and cells at the edge of the map. but I should do that somewhere else.
        # if I filter separately predictions and separately
        # annotations it may happen, that I filter out annotation which is touching corner, however it leaves
        # prediction, which may be close to corner but not really, bcs of artefacts
        '''
        cntr = 0
        while not cntr == len(self.Annotations):
            contour = self.Annotations[cntr]['contour']
            shape = self.Annotations[cntr]['img_shape']

            if 0 in contour or \
                    shape[0] in contour[:, 0, 0] or \
                    shape[1] in contour[:, 0, 1] :
                del self.Annotations[cntr]
            else:
                cntr += 1

        cntr = 0
        while not cntr == len(self.Predictions):
            contour = self.Predictions[cntr]['contour']
            shape = self.Predictions[cntr]['img_shape']

            if 0 in contour or \
                    shape[0] in contour[:, 0, 0] or \
                    shape[1] in contour[:, 0, 1]:
                del self.Predictions[cntr]
            else:
                cntr += 1

        '''


        self.IoU_matrix, self.overlap_annot_matrix, self.overlap_pred_matrix = \
            self.get_IoU_matrix(self.Annotations, self.Predictions)
        # I remove predictions and annotations which are touching cell borders, if in the related cells at least one is touching border, all of them are erased
        to_delete_pred = []
        to_delete_annot = []
        for k1 in range(self.IoU_matrix.shape[0]):
            temp_IoU = self.IoU_matrix[k1, :]
            temp_pred = self.overlap_pred_matrix[k1, :]
            temp_annot = self.overlap_annot_matrix[k1, :]

            related_cells = (temp_IoU > 0.5) | (temp_pred > 0.5) | (temp_annot > 0.5)
            related_cells = np.where(related_cells == True)[0]

            annot_cntr = self.Annotations[k1]['contour']
            annot_shape = self.Annotations[k1]['img_shape']

            delete_idx = 0 in annot_cntr or annot_shape[0] in annot_cntr[:, 0, 0] or annot_shape[1] in annot_cntr[:, 0, 1]

            for cell_idx in related_cells:
                pred_cntr = self.Predictions[cell_idx]['contour']
                #if 0 in pred_cntr or annot_shape[0] in pred_cntr[:, 0, 0] or annot_shape[1] in pred_cntr[:, 0, 1]:
                if \
                    (pred_cntr <= 5).sum() or \
                    (predictions[1]['contour'] <= 5).sum() or \
                    (pred_cntr[:, 0, 0] >= annot_shape[0]-5).sum() or \
                    (pred_cntr[:, 0, 1] >= annot_shape[1]-5).sum():

                    delete_idx = True

            if delete_idx == True:
                to_delete_pred = np.append(to_delete_pred, related_cells)
                to_delete_annot = np.append(to_delete_annot, k1)


        to_delete_annot = np.unique(to_delete_annot)
        to_delete_pred = np.unique(to_delete_pred)

        for del_pred_idx in to_delete_pred[::-1]:
            del self.Predictions[int(del_pred_idx)]

        for del_annot_idx in to_delete_annot[::-1]:
            del self.Annotations[int(del_annot_idx)]

        # estimate matrices again without edge-cells
        self.IoU_matrix, self.overlap_annot_matrix, self.overlap_pred_matrix = \
            self.get_IoU_matrix(self.Annotations, self.Predictions)


        self.IoU_matrix[self.IoU_matrix < 0.5] = 0

        # objects based values

        self.TP = None
        self.FN = None
        self.FP = None
        self.numObjects = None
        self.OS = None
        self.US = None

        self.get_obj_based_metrics()

        # pixel based values

        self.TPpx = None
        self.TPpx = None
        self.FNpx = None
        self.FPpx = None
        self.numpx = None
        #self.intersectionpx = 0 # TPpx
        #self.unionpx = 0 # FN + FP + TP

        self.get_px_based_metrix()

    def get_px_based_metrix(self):
        self.TPpx = 0
        self.TPpx = 0
        self.FNpx = 0
        self.FPpx = 0
        #self.numpx = 0
        #self.intersectionpx = 0 # TPpx
        #self.unionpx = 0 # FN + FP + TP


        max_val = 0
        for pred in self.Predictions:
            bbox = pred['bbox']
            temp = bbox[0:2] + bbox[2:]
            mtemp = temp.max()
            if mtemp > max_val: max_val = mtemp

        for annot in self.Annotations:
            bbox = annot['bbox']
            temp = bbox[0:2] + bbox[2:]
            mtemp = temp.max()
            if mtemp > max_val: max_val = mtemp

        self.annotation_img = np.zeros((max_val, max_val), dtype=np.uint8)
        self.prediction_img = np.zeros((max_val, max_val), dtype=np.uint8)

        for pred in self.Predictions:
            self.prediction_img = cv2.drawContours(self.prediction_img, [pred['contour']], -1, 1, -1)

        for annot in self.Annotations:
            self.annotation_img = cv2.drawContours(self.annotation_img, [annot['contour']], -1, 1, -1)

        sumImg = 2*self.prediction_img + self.annotation_img


        self.TPpx = sumImg[sumImg == 3].size
        self.FNpx = sumImg[sumImg == 1].size
        self.FPpx = sumImg[sumImg == 2].size
        #self.numspx = sumImg.size
        #self.intersectionpx = 0 # TPpx
        #self.unionpx = 0 # FN + FP + TP

    def get_obj_based_metrics(self):

        self.TP = 0
        self.FN = 0
        self.OS = 0
        self.US = 0
        self.FP = 0
        self.numObjects = self.Annotations.__len__()

        for k1 in range(self.Annotations.__len__()):
            IoU_annot = self.IoU_matrix[k1, :]
            num_IoU = IoU_annot[IoU_annot > 0.5].size

            # Ovlp_annot = predImg.overlap_annot_matrix[k1, :]
            # num_Ovlp_annot = Ovlp_annot[Ovlp_annot > 0.5].size

            Ovlp_pred = self.overlap_pred_matrix[k1, :]
            num_Ovlp_pred = Ovlp_pred[Ovlp_pred > 0.5].size

            if num_IoU == 1:
                self.TP += 1
            elif num_IoU == 0:
                self.FN += 1

            if num_Ovlp_pred > 1:
                self.OS += 1


        for k2 in range(self.Predictions.__len__()):
            IoU_annot = self.IoU_matrix[:, k2]
            num_IoU = IoU_annot[IoU_annot > 0.5].size

            Ovlp_annot = self.overlap_annot_matrix[:, k2]
            num_Ovlp_annot = Ovlp_annot[Ovlp_annot > 0.5].size

            Ovlp_pred = self.overlap_pred_matrix[:, k2]
            num_Ovlp_pred = Ovlp_pred[Ovlp_pred > 0.5].size

            if num_IoU == 0:
                self.FP += 1

            if num_Ovlp_annot > 1:
                self.US += 1


    def get_credentials(self, annpreds):
        for k in range(annpreds.__len__()):
            contour = annpreds[k]['contour'].astype(np.int64)
            bbox = np.array(annpreds[k]['bbox']).round().astype(np.int64)

            bbox_contour = np.zeros((4, 1, 2))
            bbox_contour[0, 0, :] = np.array([bbox[0], bbox[1]]).round()
            bbox_contour[1, 0, :] = np.array([bbox[2], bbox[1]]).round()
            bbox_contour[2, 0, :] = np.array([bbox[2], bbox[3]]).round()
            bbox_contour[3, 0, :] = np.array([bbox[0], bbox[3]]).round()
            bbox_contour = bbox_contour.astype(np.int64)


            center = np.array([0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])])
            radius = np.sqrt(bbox[3] ** 2 + bbox[2] ** 2)

            annpreds[k]['contour'] = contour
            annpreds[k]['bbox'] = bbox
            annpreds[k]['bbox_contour'] = bbox_contour
            annpreds[k]['center'] = center
            annpreds[k]['radius'] = radius
        
        return annpreds

    def get_IoU(self, mask1, mask2):
        mask1[mask1 > 0] = 1
        mask2[mask2 > 0] = 1
        mask_sum = mask1 + mask2
        union = mask_sum[mask_sum > 0].size
        intersection = mask_sum[mask_sum == 2].size
        return intersection / union

    def get_IoU_matrix(self, annotations, predictions):
        mutual_matrix = np.zeros((annotations.__len__(), predictions.__len__()))
        # gets candidates, which are close to each other and thus there is possibility of overlap
        # for potential candidates which are close to each other is computed IoU (small decrease of computational expense)
        IoU_matrix = np.zeros((annotations.__len__(), predictions.__len__()))
        overlap_annot_matrix = np.zeros((annotations.__len__(), predictions.__len__()))  # overlaps
        overlap_pred_matrix = np.zeros((annotations.__len__(), predictions.__len__()))  # overlaps

        for k1 in range(annotations.__len__()):
            for k2 in range(predictions.__len__()):
                annot = annotations[k1]
                pred = predictions[k2]

                distance = np.sqrt(((annot['center'][0] - pred['center'][0]) ** 2 + \
                                    (annot['center'][1] - pred['center'][1]) ** 2))
                Radius_b = annot['radius'] + pred['radius']
                temp_dist = distance - Radius_b  # where < 0 - potential overlap
                mutual_matrix[k1, k2] = temp_dist

                if temp_dist < 0:  # there are close each other they could overlap

                    img_shape = max([annot['contour'].max(), pred['contour'].max()])
                    mask_annot = np.zeros((img_shape, img_shape), dtype=np.uint8)
                    mask_pred = np.zeros((img_shape, img_shape), dtype=np.uint8)
                    mask_annot = cv2.drawContours(mask_annot, [annot['contour']], -1, 1, -1)
                    mask_pred = cv2.drawContours(mask_pred, [pred['contour']], -1, 1, -1)

                    # IoU
                    IoU = self.get_IoU(mask_annot, mask_pred)
                    IoU_matrix[k1, k2] = IoU

                    # OVERLAP
                    mask_sum = mask_annot + mask_pred
                    mask_overlap = mask_sum
                    mask_overlap[mask_overlap == 1] = 0
                    mask_overlap[mask_overlap == 2] = 1

                    mask_annot_overlap = mask_overlap * mask_annot
                    mask_pred_overlap = mask_overlap * mask_pred

                    ovlp_annot_val = mask_annot_overlap.sum() / mask_annot.sum()
                    ovlp_pred_val = mask_pred_overlap.sum() / mask_pred.sum()

                    overlap_annot_matrix[k1, k2] = ovlp_annot_val
                    overlap_pred_matrix[k1, k2] = ovlp_pred_val

        return IoU_matrix, overlap_annot_matrix, overlap_pred_matrix


class ScoreCounter:

    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

        self.US = 0
        self.OS = 0

        self.overall = 0

    def get_px_scores(self):
        intersection = self.TP
        union = self.FP + self.FN + self.TP
        AJI = intersection / union


        Precision = self.TP / (self.TP + self.FN)
        Recall = self.TP / (self.TP + self.FP)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

        return {
            'AJI': AJI,
            'Precision': Precision,
            'Recall': Recall,
            'F1': F1
        }

    def get_obj_scores(self):
        Precision = self.TP / (self.TP + self.FN)
        Recall = self.TP / (self.TP + self.FP)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        OS = self.OS / (self.TP + self.FN)
        US = self.US / (self.TP + self.FN)

        return {
            'Precision': Precision,
            'Recall': Recall,
            'F1': F1,
            'OS': OS,
            'US': US
        }


class TestCocoCells:


    def __init__(self, path_images, path_annotations_json):
        files_png = myF.get_files(path_images, ('tif', 'TIF'))
        files_pd = pd.DataFrame(files_png)

        def get_FID(x):
            """
            separates file id from the MELC Run path

            :param x: pandas.DataFrame (row)
            :return fid: str
            """
            temp = x['path'].split(SEPARATOR)
            return temp[len(temp) - 1][0:-4]






