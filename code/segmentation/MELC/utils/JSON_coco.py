# --------------------------------------------------------
# Multi-Epitope-Ligand Cartography (MELC) phase-contrast image based segmentation pipeline
#
#
# Written by Filip Mivalt
# --------------------------------------------------------

import json
import tifffile
import cv2
import pandas as pd
import numpy as np
from os.path import join

# MELC packages
from MELC.utils.ptr import Pointer


class JSONDataset:
    """
        Description of really cool class

        ...

        Attributes
        ----------
        attr1 : str
            this is very cool attribute

        Methods
        -------
        __init__(data_path)
            very cool init function

    """
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.COCO_structure = {
            'info': [],
            'licences': [],
            'images': [],
            'annotations': [],
            'categories': []
        }

        self.COCO_info = {
            'info': [],
            'licences': [],
            'images': [],
            'annotations': [],
            'categories': []
        }

        self.COCO_categories = {
            'subcategory': '',
            'id': 0,
            'name': ''
        }

        self.COCO_licences = list()

        self.COCO_images = list()

        self.number_of_annotations = 0

        self.ImagesObj = list()


    @classmethod
    def fromdata(cls, image_path):
        return cls(image_path)

    @classmethod
    def fromjson(cls, json_dict, image_path):
        clsObj = cls(image_path)
        annotations_pd = pd.DataFrame(json_dict['annotations'])

        for image in json_dict['images']:
            temp = JSONImage.fromjson(image, annotations_pd.loc[annotations_pd['image_id'] == image['id']].sort_values('id'))
            clsObj.ImagesObj.append(temp)

        return clsObj




    def add_image(self, image, annotations, file_name):
        ImageInputDict = {
            'image': image,
            'annotations': annotations,
            'filename': file_name + '.tif',
            'img_id': len(self.ImagesObj)
        }

        image = image.round().astype(np.uint16)
        #print(join(self.image_path, ImageInputDict['filename']))
        #print(image.dtype)
        #print(image.shape)
        #print(image)
        tifffile.imsave(join(self.image_path, ImageInputDict['filename']), image)


        JImg = JSONImage.fromdata(ImageInputDict)
        JImg.set_annotation_base_id_ptr(self.number_of_annotations)
        self.number_of_annotations += JImg.COCO_annotations.__len__()
        self.ImagesObj.append(JImg)



    def write_json(self, path_json_file):
        self.COCO_structure['info'] = self.COCO_info
        self.COCO_structure['licences'] = self.COCO_licences
        self.COCO_structure['categories'] = [{"subcategory": "", "id": 1, "name": "phaseCell", "supercategory": "cell"}]

        for ImgObj in self.ImagesObj:
            img_json, annot_json = ImgObj.get_JSON()
            self.COCO_structure['images'].append(img_json)
            self.COCO_structure['annotations'] = self.COCO_structure['annotations'] + annot_json

        with open(path_json_file, 'w') as f:
            json.dump(self.COCO_structure, f)

        return self.COCO_structure




    #    with open(json_file, 'w') as f:
    #        json.dump(json_file, f)


class JSONImage:
    """
        Description of really cool class

        ...

        Attributes
        ----------
        attr1 : str
            this is very cool attribute

        Methods
        -------
        __init__(data_path)
            very cool init function

    """

    def __init__(self):
        #self._init_from_data(Image)
        pass

    @classmethod
    def fromjson(cls, Image_json = {
        'licenses': '',
        'file_name': None,
        'coco_url': '',
        'height': None,
        'width': None,
        'date_captured': '',
        'flickr_url': '',
        'id': None
    }, annotations_pdDataFrame = pd.DataFrame
                  ):

        clsObj = cls()
        clsObj.license = Image_json['license']
        clsObj.file_name = Image_json['file_name']
        clsObj.coco_url = Image_json['coco_url']
        clsObj.height = Image_json['height']
        clsObj.width = Image_json['width']
        clsObj.date_captured = Image_json['date_captured']
        clsObj.flick_url = Image_json['flickr_url']
        clsObj.PTR_id = Pointer(Image_json['id'])

        clsObj.PTR_annotation_base_id = Pointer(annotations_pdDataFrame['id'].min())
        clsObj.annotation_relative_id = annotations_pdDataFrame['id'].__len__()

        clsObj.COCO_annotations = list()

        for annotation in annotations_pdDataFrame.iterrows():
            input_annotation_dict = {
                'annotation_pdDataFrame': annotation[1],
                'PTR_annotation_base_id': clsObj.PTR_annotation_base_id,
                'annotation_relative_id': annotation[1]['id'] - clsObj.PTR_annotation_base_id.get(),
                'PTR_image_id': clsObj.PTR_id,
            }

            clsObj.COCO_annotations.append(JSONAnnotation.fromjson(input_annotation_dict))

        return clsObj

    @classmethod
    def fromdata(cls, Image={
        'image': np.array([]),
        'filename': '',
        'img_id': 0,
        'annotations': []
    }
                        ):
        clsObj = cls()

        clsObj.license = 0
        clsObj.file_name = Image['filename']
        clsObj.coco_url = ""
        clsObj.height = Image['image'].shape[0]
        clsObj.width = Image['image'].shape[1]
        clsObj.date_captured = ""
        clsObj.flick_url = ""

        clsObj.PTR_id = Pointer(Image['img_id'])
        clsObj.PTR_annotation_base_id = Pointer(0)
        clsObj.annotation_relative_id = -1

        clsObj.COCO_annotations = list()

        for contour in Image['annotations']:
            if contour.shape[0] > 6:
                clsObj.annotation_relative_id += 1
                annotation = {
                    'contour': contour,
                    'PTR_annotation_base_id': clsObj.PTR_annotation_base_id,
                    'annotation_relative_id': clsObj.annotation_relative_id,
                    'PTR_image_id': clsObj.PTR_id,
                    'category_id': 1,
                    'id': 0,
                    'img_shape': Image['image'].shape
                }
                clsObj.COCO_annotations.append(JSONAnnotation.fromdata(annotation))

        return clsObj

    def set_annotation_base_id_ptr(self, value):
        self.PTR_annotation_base_id.set(value)

    def get_annotation_base_id_ptr(self):
        return self.PTR_annotation_base_id.get()

    def set_image_id_PTR(self, value):
        self.PTR_id.set(value)

    def get_image_id_PTR(self):
        return self.PTR_id.get()

    def get_JSON(self):
        annotations = [annot.get_JSON() for annot in self.COCO_annotations]

        image = {
            'license': '',
            'file_name': self.file_name,
            'coco_url': self.coco_url,
            'height': self.height,
            'width': self.width,
            'date_captured': self.date_captured,
            'flickr_url': self.flick_url,
            'id': self.PTR_id.get()
        }

        return image, annotations


class JSONAnnotation:
    """
        Object representing COCO JSON annotation with all parameters defined in COCO file format.
        It is possible to initialize this object from raw data, while there is input of numpy array format shape = [N, 1, 2]
        or to initialise object from annotation from json file.

        ...

        Attributes
        ----------
        contour : np.ndarray
            Contour as is returned by the function cv2.getContours. The shape is (N, 1, 2), where N represents length of
            the contour and X and Y are represented by 3rd dimension. The type of contour MUST be np.int32!!!

        area : float
            Identificator for the contour for the consistency check.

        iscrowd : int
            whether annotation entity represents group of multiple objects or not (0, 1). There is implemented only 0,
            value in this work. There may be implemented 1 in possible future work for cell clamps.

        bbox : list [int, int, int, int]
            bounding box for each annotation given by contour.

            bbox[0] = contour[:, 0, 0].min()
            bbox[2] = contour[:, 0, 0].max() - contour[:, 0, 0].min()

            bbox[1] = contour[:, 0, 1].min()
            bbox[3] = contour[:, 0, 1].max() - contour[:, 0, 1].min()

        category_id : int
            id of detected categhory specified in the file. For cell it will be probably always 1

        PTR_image_id : MELC.utils.Pointer()
            Image id of corresponding image. The value may be changed or set after

        relative_id : int
            id of annotation for given image. If image0 has 30 annotations, then relative_id are on the range from 0-29

        PTR_base_id : int
            denotes offset of annotation. if image1 has 30 annotations with relative ids 0-29 and in image0, there were
            20 annotations, then offset for image1 is 20. Thus, absolute annotation id for image1 are 20+relative_id -> 20-49

        Class Methods
        -------------
        fromdata(annotation)
            annotation = {
                            'contour': [],  # cv2 format contour
                            'PTR_annotation_base_id': Pointer(),    # pointer to the annotation index offset for given image
                            'annotation_relative_id': None,         # number of the annotation in the image
                            'PTR_image_id': Pointer(),      # pointer to the id of corresponding image
                            'category_id': None,        #id of category of the annotation - for cell 1
                            'img_shape': (None, None)
                        }

        fromjson(annotation)
            annotation = {
                            'contour': [],
                            'PTR_annotation_base_id': Pointer(),
                            'annotation_relative_id': None,
                            'PTR_image_id': Pointer(),
                            'category_id': None,
                            'id': None,
                            'img_shape': (None, None)
                        }

        Methods
        -------
        __init__()
            initialisation function without any further functions. It is overloaded by fromjson and fromdata functions

        get_JSON()
            returns json dictionary in COCO format for COCO dataset of given annotation

    """

    def __init__(self):
        pass
        #self._init_from_file(annotation)


    @classmethod
    def fromjson(cls, annotation = {
        'annotation_pdDataFrame': [],
        'PTR_annotation_base_id': Pointer(),
        'annotation_relative_id': None,
        'PTR_image_id': Pointer(),
    }
                 ):

        clsObj = cls()

        ##### contour #####
        annotation['annotation_pdDataFrame']
        temp_contour = annotation['annotation_pdDataFrame']['segmentation'][0] #TADYYYYY ADDED [0]
        temp_contour = np.array(temp_contour)
        clsObj.contour = np.zeros((int(temp_contour.__len__()/2), 1, 2))
        clsObj.contour[:, 0, 0] = temp_contour[0::2]
        clsObj.contour[:, 0, 1] = temp_contour[1::2]

        ##### area #####
        clsObj.area = annotation['annotation_pdDataFrame']['area']

        ##### iscrowd #####
        clsObj.iscrowd = annotation['annotation_pdDataFrame']['iscrowd']

        ##### bbox #####
        clsObj.bbox = annotation['annotation_pdDataFrame']['bbox']

        ##### category_id #####
        clsObj.category_id = annotation['annotation_pdDataFrame']['category_id']

        ##### image_id #####
        clsObj.PTR_image_id = annotation['PTR_image_id']

        ##### id #####
        clsObj.relative_id = annotation['annotation_relative_id']
        clsObj.PTR_base_id = annotation['PTR_annotation_base_id']
        # self.id = self.relative_id_annotation.get() + self.PTR_base_id ### this will be in the output

        return clsObj

    @classmethod
    def fromdata(cls, annotation = {
        'contour': [],
        'PTR_annotation_base_id': Pointer(),
        'annotation_relative_id': None,
        'PTR_image_id': Pointer(),
        'category_id': None,
        'id': None,
        'img_shape': (None, None)
    }
                 ):

        clsObj = cls()
        img_shape = annotation['img_shape']

        ##### contour #####
        clsObj.contour = annotation['contour']
        clsObj.contour[clsObj.contour < 0] = 0
        temp1 = clsObj.contour[:, 0, 0]
        temp2 = clsObj.contour[:, 0, 1]
        temp2[temp2 > img_shape[1]] = img_shape[1]
        temp1[temp1 > img_shape[0]] = img_shape[0]
        temp2[temp2 < 0] = 0
        temp1[temp1 < 0] = 0

        clsObj.contour[:, 0, 0] = temp1
        clsObj.contour[:, 0, 1] = temp2


        ##### area #####
        clsObj.area = cv2.moments(clsObj.contour)['m00']

        ##### iscrowd #####
        clsObj.iscrowd = 0

        ##### bbox #####
        clsObj.bbox = [0, 0, 0, 0]
        clsObj.bbox[0] = clsObj.contour[:, 0, 0].min()
        clsObj.bbox[2] = clsObj.contour[:, 0, 0].max() - clsObj.contour[:, 0, 0].min()

        clsObj.bbox[1] = clsObj.contour[:, 0, 1].min()
        clsObj.bbox[3] = clsObj.contour[:, 0, 1].max() - clsObj.contour[:, 0, 1].min()

        clsObj.bbox[0] = clsObj.bbox[0] - 3
        clsObj.bbox[1] = clsObj.bbox[1] - 3
        clsObj.bbox[2] = clsObj.bbox[2] + 3
        clsObj.bbox[3] = clsObj.bbox[3] + 3

        if clsObj.bbox[0] < 0: clsObj.bbox[0] = 0
        if clsObj.bbox[1] < 0: clsObj.bbox[1] = 0
        if clsObj.bbox[2] > img_shape[1]: clsObj.bbox[2] = img_shape[1]
        if clsObj.bbox[3] > img_shape[0]: clsObj.bbox[3] = img_shape[0]


        for k in range(4):
            clsObj.bbox[k] = int(clsObj.bbox[k])


        ##### image_id #####
        clsObj.PTR_image_id = annotation['PTR_image_id']

        ##### category_id #####
        clsObj.category_id = annotation['category_id']

        ##### id #####
        clsObj.relative_id = annotation['annotation_relative_id']
        clsObj.PTR_base_id = annotation['PTR_annotation_base_id']
        # self.id = self.relative_id_annotation.get() + self.PTR_base_id ### this will be in the output

        return clsObj


    def get_JSON(self):

        contour = np.zeros(2 * self.contour.shape[0])
        contour[0::2] = self.contour[:, 0, 0]
        contour[1::2] = self.contour[:, 0, 1]

        return {
            'segmentation': [contour.tolist()],  # [[h1, w1, h2, w2, ....]] # TADYYYYYYYYYYYY added []
            'area': self.area,
            'bbox': self.bbox,
            'iscrowd': 0,  #
            'image_id': int(self.PTR_image_id.get()),  # id of image
            'category_id': int(self.category_id),
            'id': int(self.relative_id + self.PTR_base_id.get())
        }


