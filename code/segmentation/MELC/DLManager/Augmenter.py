# --------------------------------------------------------
# Multi-Epitope-Ligand Cartography (MELC) phase-contrast image based segmentation pipeline
#
#
# Written by Filip Mivalt
# --------------------------------------------------------

import cv2
import numpy as np
from numpy.random import randint, normal
from copy import deepcopy

from numpy.random import randint, uniform

class ImageAugmenter():


    @classmethod
    def get_random_transform(cls, img, annotations, expand=True):

        #tp = img.dtype
        #img = img.astype(np.float64)
        #img = img + randint(-100, 100)
        #mn = img.mean()
        #img = (img - mn) * uniform(0.95, 1.05) + mn
        #img = img.round().astype(tp)


        if expand == True:
            resRat = 1 + (0.6*randint(0, 10)/10)

        elif expand == False:
            resRat = 0.7 + (0.3 * randint(0, 10) / 10)

        temp = np.zeros(img.shape, dtype=np.uint8)
        temp = cv2.drawContours(temp, annotations, -1, 255, -1)
        background_mean = np.median(img[temp == 0])
        background_std = np.std(img[temp == 0])*0.15
        rot_effect_threshold = background_mean - 300

        tImg = cv2.resize(img, (0, 0), fx=resRat, fy=resRat)
        tAnnotations = list()
        for annotation in annotations:
            temp_annot = deepcopy(annotation)
            temp_annot[:, 0, 0] = temp_annot[:, 0, 0] * resRat
            temp_annot[:, 0, 1] = temp_annot[:, 0, 1] * resRat
            tAnnotations.append(temp_annot.round().astype(np.int32))




        ########### ROTATE ##############
        #M = cls._get_rotation(tImg)
        #tImg, tAnnotations = cls._transform(tImg, tAnnotations, M)
        #tImg[tImg < rot_effect_threshold] = normal(background_mean, background_std, tImg[tImg < rot_effect_threshold].shape)



        if resRat > 1:
            ########### ROTATE ##############
            M = cls._get_rotation(tImg)
            tImg, tAnnotations = cls._transform(tImg, tAnnotations, M)
            tImg[tImg < rot_effect_threshold] = normal(background_mean, background_std,
                                                       tImg[tImg < rot_effect_threshold].shape)



            tAnnotations_2 = list()
            s_orig = img.shape
            s_res = tImg.shape
            shift = randint(0, s_res[1] - s_orig[1], 2, dtype=np.int32)

            idx1 = np.array([0, s_orig[0]]) + shift[0]
            idx2 = np.array([0, s_orig[1]]) + shift[1]



            oImg = tImg[idx1[0] : idx1[1], idx2[0] : idx2[1]]



            for annotation in tAnnotations:
                temp_annot = deepcopy(annotation)
                temp_annot[:, 0, 0] = temp_annot[:, 0, 0] - shift[1]
                temp_annot[:, 0, 1] = temp_annot[:, 0, 1] - shift[0]


                # is IN?
                # is OUT?
                # the rest

                min_cell_to_min_border_1 = temp_annot[:, 0, 0].min() > 0
                min_cell_to_max_border_1 = temp_annot[:, 0, 0].min() < oImg.shape[0]
                max_cell_to_min_border_1 = temp_annot[:, 0, 0].max() > 0
                max_cell_to_max_border_1 = temp_annot[:, 0, 0].max() < oImg.shape[0]

                min_cell_to_min_border_2 = temp_annot[:, 0, 1].min() > 0
                min_cell_to_max_border_2 = temp_annot[:, 0, 1].min() < oImg.shape[1]
                max_cell_to_min_border_2 = temp_annot[:, 0, 1].max() > 0
                max_cell_to_max_border_2 = temp_annot[:, 0, 1].max() < oImg.shape[1]

                bools_idx1 = np.array([
                    min_cell_to_min_border_1,
                    min_cell_to_max_border_1,
                    max_cell_to_min_border_1,
                    max_cell_to_max_border_1
                ])

                bools_idx2 = np.array([
                    min_cell_to_min_border_2,
                    min_cell_to_max_border_2,
                    max_cell_to_min_border_2,
                    max_cell_to_max_border_2
                ])

                inside_bcombination = np.array([True, True, True, True])

                upper_outside_bcombination = np.array([True, False, True, False])
                bottom_outside_bcombination = np.array([False, True, False, True])

                upper_crossing_bcombination = np.array([True, True, True, False])
                bottom_crossing_bcombination = np.array([False, True, True, True])

                if all(bools_idx1 == inside_bcombination) and all(bools_idx2 == inside_bcombination):
                    tAnnotations_2.append(temp_annot.round().astype(np.int32))

                elif (all(bools_idx1 == upper_outside_bcombination) or all(bools_idx2 == upper_outside_bcombination)) and \
                   (all(bools_idx1 == bottom_outside_bcombination) or all(bools_idx2 == bottom_outside_bcombination)):
                    # is completely outside nothing is going on
                    pass

                else:
                    idx = np.zeros(temp_annot.shape[0])
                    for k in range(idx.shape[0]):
                        if not(temp_annot[k, 0, 0] < 0 or temp_annot[k, 0, 0] < oImg.shape[0] or \
                                temp_annot[k, 0, 1] < oImg.shape[0] or temp_annot[k, 0, 1] < oImg.shape[1]):
                            idx[k] = 1

                    temp_annot = temp_annot[idx==1, :, :]
                    # is provably in crossing
                    #temp_annot_1 = temp_annot[:, 0, 0]
                    #temp_annot_2 = temp_annot[:, 0, 1]

                    #temp_annot_1[temp_annot_1 < 0] = 0
                    #temp_annot_1[temp_annot_1 > idx1[1]] = oImg.shape[0]

                    #temp_annot_2[temp_annot_2 < 0] = 0
                    #temp_annot_2[temp_annot_2 > idx2[1]] = oImg.shape[1]

                    #temp_annot[:, 0, 0] = temp_annot_1
                    #temp_annot[:, 0, 1] = temp_annot_2
                    if temp_annot.__len__() > 3:
                        tAnnotations_2.append(temp_annot.round().astype(np.int32))



            tAnnotations = tAnnotations_2

        elif resRat < 1:
            oImg = normal(background_mean, background_std*0.8, img.shape)


            tImg_c1 = tImg.shape[0] / 2.0
            tImg_c2 = tImg.shape[1] / 2.0

            oImg_c1 = oImg.shape[0] / 2.0
            oImg_c2 = oImg.shape[1] / 2.0

            idx1 = np.array([0, tImg.shape[0]])
            idx2 = np.array([0, tImg.shape[1]])

            idx1 = idx1 - tImg_c1 + oImg_c1
            idx2 = idx2 - tImg_c2 + oImg_c2
            idx1 = idx1.astype(np.int16)
            idx2 = idx2.astype(np.int16)

            oImg[idx1[0] : idx1[1], idx2[0] : idx2[1]] = tImg

            for k in range(len(tAnnotations)):
                tAnnotations[k][:, 0, 0] = tAnnotations[k][:, 0, 0] - tImg_c1 + oImg_c1
                tAnnotations[k][:, 0, 1] = tAnnotations[k][:, 0, 1] - tImg_c2 + oImg_c2


            ########### ROTATE ##############
            M = cls._get_rotation(oImg)
            oImg, tAnnotations = cls._transform(oImg, tAnnotations, M)
            oImg[oImg < rot_effect_threshold] = normal(background_mean, background_std,
                                                        oImg[oImg < rot_effect_threshold].shape)

        else:
            return img, annotations

        return oImg, tAnnotations

    @classmethod
    def _get_rotation(cls, img):
        return CellObjectAugmenter._get_rotation(img)

    @classmethod
    def _transform(cls, img, annotations, M):
        return CellObjectAugmenter._transform(img, annotations, M)


class CellObjectAugmenter():
    @classmethod
    def get_random_transform(cls, img, annotations):
        M = cls._get_random_transformation_matrix(img)
        return cls._transform(img, annotations, M)

    @classmethod
    def _transform(cls, img, annotations, M):
        tImg = cv2.warpPerspective(img, M, img.shape)
        tAnnotations = list()

        for annotation in annotations:
            temp_contour = np.ones((annotation.shape[0], 3))
            temp_contour[:, :2] = annotation.squeeze()[:, :]
            x = np.dot(M, temp_contour.transpose())
            y = np.zeros(annotation.shape)

            y[:, 0, 0] = x[0, :]
            y[:, 0, 1] = x[1, :]
            y[:, 0, 0] = y[:, 0, 0]/x[2, :]
            y[:, 0, 1] = y[:, 0, 1]/x[2, :]
            tAnnotations.append(y.round().astype(np.int32))

        return tImg, tAnnotations


    @classmethod
    def get_random_resize(cls, img, annotations):
        resRat_x = 0.7 + (0.4*randint(0, 1000)/1000)
        resRat_y = 0.7 + (0.4*randint(0, 1000)/1000)

        tImg = cv2.resize(img, (0, 0), fx=resRat_x, fy=resRat_y)
        tAnnotations = list()
        for annotation in annotations:
            temp_annot = deepcopy(annotation)
            temp_annot[:, 0, 0] = temp_annot[:, 0, 0] * resRat_x
            temp_annot[:, 0, 1] = temp_annot[:, 0, 1] * resRat_y

            tAnnotations.append(temp_annot.round().astype(np.int32))
        return tImg, tAnnotations

    @classmethod
    def _get_random_transformation_matrix(cls, img):
        M_rot = cls._get_rotation(img)
        M_persp = cls._get_perspective_transformation(img)
        return np.dot(M_rot, M_persp)

    @classmethod
    def _get_rotation(cls, img):
        degree = np.random.randint(360)# in degrees
        rows, cols = img.shape
        M1 = cv2.getRotationMatrix2D((cols/2,rows/2), degree, 1)
        M01 = np.zeros((3, 3))
        M01[:-1,:] = M1
        M01[-1,-1] = 1
        return M01

    @classmethod
    def _get_perspective_transformation(cls, img):
        pts2 = np.array(
            [
                [
                    0, 0
                ],
                [
                    img.shape[1], 0
                ],
                [
                    0, img.shape[0]
                ],
                [
                    img.shape[1], img.shape[0]
                ]
            ]
        )

        pts1 = deepcopy(pts2)
        for k1 in range(pts1.shape[0]):
            for k2 in range(pts1.shape[1]):
                temp_rand = np.random.randint(-img.shape[k2] / 6, img.shape[k2] / 6)
                if pts1[k1, k2] == 0:
                    pts1[k1, k2] = pts1[k1, k2] + temp_rand
                else:
                    pts1[k1, k2] = pts1[k1, k2] - temp_rand

        M2 = cv2.getPerspectiveTransform(pts2.astype(np.float32), pts1.astype(np.float32))
        return M2

