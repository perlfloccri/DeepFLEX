import cv2
import numpy as np
from tifffile import imread
import random
import os

class AnnotatedObject:
    mean_x = 0
    mean_y = 0
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    img_ind = 0

    def __init__(self,img_ind = 0,obj_ind = 0, mean_x = None, mean_y = None, min_x = None, min_y = None, max_x = None, max_y = None):
        self.img_ind = img_ind
        self.obj_ind = obj_ind
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

class Image:

    raw = 0

    def pre_process_img(img, color='gray'):
        if color is 'gray':
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                e=1 #print("Error in Conversion to grayscale: maybe already grayscale?")
        elif color is 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            pass
        if ((img.dtype == 'int16') or (img.dtype == 'uint16')):
            print (img.dtype)
            max_097_value = np.percentile(img,99.5)
            print("Image type 16 Bit \n")
            print("Max value: " + str(max_097_value) + "\n")
            img_new = img.astype(np.float32) / float(max_097_value)
            print (img_new.dtype)
            img = img_new * 255.0
            img = img.astype(np.float32)
        else:
            img = img.astype(np.float32)
            img /= 255.0
        cv2.imwrite(r"/root/flo/tmp/test.jpg",img)
        return img

    def getRaw(self):
        return self.raw

class AnnotatedImage(Image):

    mask = 0

    def readFromPath(self,image_path,mask_path):
        print("Reading image " + image_path)
        if os.path.basename(image_path).split('.')[1] == 'jpg':
            self.raw = Image.pre_process_img(cv2.imread(image_path),color='gray')
        else:
            self.raw = Image.pre_process_img(imread(image_path),color='gray')
        if os.path.basename(mask_path).split('.')[1] == 'jpg':
            self.mask = cv2.imread(mask_path)
        else:
            self.mask = imread(mask_path)

    def readFromPathOnlyImage(self,image_path):
        if os.path.basename(image_path).split('.')[1] == 'jpg':
            self.raw = Image.pre_process_img(cv2.imread(image_path),color='gray')
        else:
            self.raw = Image.pre_process_img(imread(image_path),color='gray')
        self.mask = np.zeros_like(self.raw,dtype=np.uint8)

    def createWithArguments(self,image,mask):
        self.raw = image
        self.mask = mask
    def getMask(self):
        return self.mask
    def getCroppedAnnotatedImage(self,annotation):
        tmp = AnnotatedImage()
        tmp.createWithArguments(self.raw[annotation.min_x:annotation.max_x,annotation.min_y:annotation.max_y] * (self.mask[annotation.min_x:annotation.max_x,annotation.min_y:annotation.max_y]==annotation.obj_ind).astype(np.uint8),self.mask[annotation.min_x:annotation.max_x,annotation.min_y:annotation.max_y]==annotation.obj_ind)
        return tmp
    def getMeanMaskObjectSize(self):
        total_sum = np.uint64(0);
        for i in range(self.mask.max()):
           total_sum = total_sum + np.square((self.mask==(i+1)).sum())
        A = np.sqrt(total_sum / self.mask.max())
        return (2*np.sqrt(A/np.pi)).astype(np.int16)

class ArtificialAnnotatedImage(AnnotatedImage):

    running_mask = 0
    number_nuclei = 0
    griddy = 0

    def __init__(self,width=None,height=None, number_nuclei = None, probabilityOverlap=0):
        self.raw = np.zeros((height,width))
        self.mask = np.zeros((height,width))
        self.number_nuclei = number_nuclei
        self.griddy = gridIterable(width=width,height=height,nrObjects=number_nuclei,probabilityOverlap=probabilityOverlap)

    def addImageAtRandomPosition(self,image):
        rand_x = random.randint(0,int(self.raw.shape[0]-image.getRaw().shape[0]))
        rand_y = random.randint(0,int(self.raw.shape[1] - image.getRaw().shape[1]))

        [x, y] = np.where(image.getMask() == 1)
        self.raw[x+rand_x, y+rand_y] = image.getRaw()[x, y]
        self.running_mask = self.running_mask + 1
        self.mask[x+rand_x, y+rand_y] = image.getMask()[x, y] + self.running_mask

    def addImageAtGridPosition(self,image):
        pos = self.griddy.next()
        #print('minx: ' + str(pos.minx) + ', maxx: ' + str(pos.maxx) + ', miny: ' + str(pos.miny) + ', maxy: ' + str(pos.maxy))
        rand_x = random.randint(pos.minx,pos.maxx)
        rand_y = random.randint(pos.miny,pos.maxy)
        [x, y] = np.where(image.getMask() == 1)
        tmp = image.getRaw()
        tmp_mask = image.getMask()
        for i in range(0,x.__len__()):
            try:
                if (((x[i] + rand_x) > 0 ) & ((y[i] + rand_y) > 0 )):
                    self.raw[x[i] + rand_x, y[i] + rand_y] = tmp[x[i], y[i]]
                    self.mask[x[i] + rand_x, y[i] + rand_y] = tmp_mask[x[i], y[i]] + self.running_mask
            except:
                e=0
        self.running_mask = self.running_mask + 1
        return 1
        #try:
        #    self.raw[x+rand_x, y+rand_y] = image.getRaw()[x, y]
        #    self.running_mask = self.running_mask + 1
        #    self.mask[x+rand_x, y+rand_y] = image.getMask()[x, y] + self.running_mask
        #    return 1
        #except:
        #    return 0

    def transformToArtificialImage(image=None,useBorderObjects=False):
        raw = np.zeros((image.getRaw().shape[0],image.getRaw().shape[1]))
        mask = np.zeros((image.getRaw().shape[0],image.getRaw().shape[1]))
        running_mask = 0
        #for i in range(1, image.getMask().max()+1):
        for i in np.unique(image.getMask()):
            if i > 0:
                [x, y] = np.where(image.getMask() == i)
                #if (x.__len__() > 0):
                #if (~useBorderObjects): # border objects excluding could be implemented
                #    if ~((x.min() == 0) | (y.min() == 0) | (x.max() == image.getRaw().shape[0]) | (y.max() == image.getRaw().shape[1])):
                raw[x, y] = image.getRaw()[x, y]
                running_mask = running_mask + 1
                mask[x, y] = image.getMask()[x, y] + running_mask
        ret_img = AnnotatedImage()
        ret_img.createWithArguments(raw,mask)
        return ret_img

class AnnotatedObjectSet:
    images = []
    objects = []
    path_to_imgs = []

    def addObjectImage(self,image=None,useBorderObjects=False, path_to_img=None):
        self.images.append(image)
        if path_to_img:
            self.path_to_imgs.append(path_to_img)
        curr_img_index = self.images.__len__() - 1
        for i in range(1,image.getMask().max()+1):
            [x, y] = np.where(cv2.dilate((image.getMask() == i).astype(np.uint8),np.ones((5,5),np.uint8),iterations = 1))
            if (x.__len__() > 0):
                if (~useBorderObjects):
                    if ~((x.min() == 0) | (y.min() == 0) | (x.max() == image.getRaw().shape[0]) | (y.max() == image.getRaw().shape[1])):
                        self.objects.append(AnnotatedObject(img_ind = curr_img_index, obj_ind = i, mean_x = x.mean(),mean_y = y.mean(),min_x = x.min(), min_y = y.min(),max_x = x.max(), max_y = y.max()))

    def returnArbitraryObject(self):
        rand_int = random.randint(0,self.objects.__len__()-1)
        return self.images[self.objects[rand_int].img_ind].getCroppedAnnotatedImage(self.objects[rand_int])

class gridIterable:

    def __init__(self,width=0,height=0,nrObjects=0,probabilityOverlap = 0):
        self.width=width
        self.height = height
        self.nrObjects = nrObjects
        self.nr_x = self.nr_y = np.sqrt(nrObjects)
        self.stepX = self.width / (self.nr_x-1)
        self.stepY = self.height / (self.nr_y-1)
        self.probabilityOverlap = 1-probabilityOverlap
        self.curr_ind = 0

    def __iter__(self):
        return self

    def next(self):
        if (self.curr_ind <= (self.nrObjects)):
            self.curr_ind = self.curr_ind + 1
            row = np.ceil(self.curr_ind / self.nr_x)
            column = self.curr_ind - self.nr_x * (row-1)
            a = Rectangle(minx=round(0+(column-1)*self.stepX + (self.stepX/2) * self.probabilityOverlap),maxx=round(0+column*self.stepX - (self.stepX/2) * self.probabilityOverlap-1),miny=round(0+(row-1)*self.stepY + (self.stepY/2) * self.probabilityOverlap),maxy=round(0+row*self.stepY - (self.stepY/2) * self.probabilityOverlap-1))

            return a
        else:
            raise StopIteration()

class Rectangle:
    def __init__(self,minx=None,maxx=None,miny=None,maxy=None):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
