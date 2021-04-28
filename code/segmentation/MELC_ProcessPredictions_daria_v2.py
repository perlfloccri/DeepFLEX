import sys
from os.path import join
from config import *
project_folder = '/workspace/code/MELC_pipeline/maskrcnn'
from skimage.measure import label as lb

#sys.path.append(join(project_folder, 'config'))
#sys.path.append(join(project_folder, 'lib'))
#sys.path.append(join(project_folder, 'MELC'))

from tqdm import tqdm

import os
import numpy as np
from os.path import join
#from myUtils.imageTiling import *
from scipy.io import savemat, loadmat
from scipy.signal  import medfilt2d
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
import matplotlib
matplotlib.use('Qt5Agg')




from MELC.utils.Files import get_files
import tifffile
from MELC.Client.Annotation import SVGAnnot
import argparse

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Run Inference of Mask R-CNN')

    parser.add_argument(
        '--phase_pred_path', dest='phase_pred_path', required=True,
        help='Config file for training (and optionally testing)')

    parser.add_argument(
        '--fluor_pred_path', dest='fluor_pred_path', required=True)

    parser.add_argument(
        '--phase_source_path', dest='phase_source_path', required=True)

    parser.add_argument(
        '--output_path', dest='output_path', required=True)

    return parser.parse_args()

def print_contours(img, contours):
    masks = np.zeros_like(img, dtype=np.uint16)
    cntr256 = 0
    cntr_laps = 0

    temp_masks = np.zeros_like(img, dtype=np.uint8)
    for contour in contours:
        cntr256 += 1
        temp_masks = cv2.drawContours(temp_masks, [contour], -1, cntr256, -1)

        if cntr256 == 255 or np.array_equal(contour, contours[contours.__len__() - 1]):
            temp_masks = temp_masks.astype(np.uint16)
            temp_masks[temp_masks > 0] = temp_masks[temp_masks > 0] + cntr_laps * 255
            cntr256 = 0
            cntr_laps += 1

            masks[temp_masks > 0] = temp_masks[temp_masks > 0]
            temp_masks = np.zeros_like(img, dtype=np.uint8)

    return masks



def find_nuclei_file(predictions):
    for k in range(predictions.__len__()):
        if 'iodide' in predictions[k]:
            return k
    return None




def main():
    args = parse_args()
    path_phase = args.phase_source_path
    print("Pfad: " + path_phase)
    path_phase_preds = args.phase_pred_path
    path_fluor_preds = args.fluor_pred_path
    path_nuclei_preds = get_files(path_fluor_preds, 'svg')[0]

    predictions = os.listdir(path=path_phase_preds)
    nuclei_idx = find_nuclei_file(predictions)
    print (predictions[nuclei_idx])
    print ("Path PC iodide mask:" + join(path_phase_preds, predictions[nuclei_idx].split('.')[0].replace("_svg","_mask.TIF")))
    print ("Path nuclear iodide mask:" + join(path_fluor_preds, predictions[nuclei_idx].split('.')[0].replace("_svg","_mask.TIF")))
    # Phase contrast mask of Propidiodide staining
    PI_PC_mask = tifffile.imread(join(path_phase_preds, predictions[nuclei_idx].split('.')[0].replace("_svg","_mask.TIF")))
    Combined_PC_masks = np.ones_like(PI_PC_mask, dtype=np.bool)
	
    # Read nucleus mask
    PI_Nuclei_mask = tifffile.imread(join(path_fluor_preds, predictions[nuclei_idx].split('.')[0].replace("_svg","_mask.TIF")))
    
    for idx, pred in enumerate(predictions):
        if pred.endswith("_mask.TIF"):
            masks = tifffile.imread(join(path_phase_preds,pred))
            # Combination of all phase contrast masks (binary multiplication)
            Combined_PC_masks = Combined_PC_masks * masks.astype(np.bool) * PI_Nuclei_mask.astype(np.bool)

    # Filter (extend) mask containing phase contrast objects of all Stainings
    Combined_PC_masks = medfilt2d(Combined_PC_masks.astype(np.float)).astype(np.bool)

    # Filter cells according to size and relabel
    cell_th = 40
    nucleus_th = 30

    # Keep only intact objects present PI PC mask
    cell_labels_to_keep = np.unique(PI_PC_mask.astype(np.uint16) * Combined_PC_masks.astype(np.uint16))
    nuclear_labels_to_keep = np.unique(PI_Nuclei_mask.astype(np.uint16) * Combined_PC_masks.astype(np.uint16))

    print ("Filtering PC mask ...")
    for celllabel in tqdm(np.unique(PI_PC_mask)):
        if celllabel > 0:
            if (((PI_PC_mask == celllabel).sum() < cell_th) or (celllabel not in cell_labels_to_keep)):
                PI_PC_mask[PI_PC_mask == celllabel] = 0
    PI_PC_mask = lb(PI_PC_mask > 0,connectivity=1)
    print("Remaining cell number: " + str(np.unique(PI_PC_mask).__len__()))

    print("Filtering nuclear mask ...")
    for nuclearlabel in tqdm(np.unique(PI_Nuclei_mask)):
        if nuclearlabel > 0:
            if (((PI_Nuclei_mask == nuclearlabel).sum() < nucleus_th) or (nuclearlabel not in nuclear_labels_to_keep)):
                PI_Nuclei_mask[PI_Nuclei_mask == nuclearlabel] = 0
    PI_Nuclei_mask = lb(PI_Nuclei_mask > 0,connectivity=1)
    print("Remaining nucleus number: " + str(np.unique(PI_Nuclei_mask).__len__()))

    # Generate list of cell/nucleus pairs and save it
    cell_list = []
    print ("Generating cell/nuclei pairs...")
    combined_cell_mask = np.zeros_like(PI_PC_mask, dtype=np.uint16)
    combined_nucleus_mask = np.zeros_like(PI_Nuclei_mask, dtype=np.uint16)
    running_label = 1
    for celllabel in tqdm(np.unique(PI_PC_mask)):
        if celllabel > 0:
            max_overlap = 0
            max_label = 0
            tmp = (PI_PC_mask == celllabel)
            labels_nuclei_remaining = np.unique(tmp.astype(np.uint16) * PI_Nuclei_mask)
            for nuclearlabel in labels_nuclei_remaining:
                if (nuclearlabel > 0):
                    if (tmp * (PI_Nuclei_mask == nuclearlabel)).sum() > max_overlap:
                        max_overlap = (tmp * (PI_Nuclei_mask == nuclearlabel)).sum()
                        max_label = nuclearlabel
            if (max_overlap > 0):
                # Add cell/nucleus to combined mask
                combined_cell_mask[PI_PC_mask == celllabel] = 0
                combined_cell_mask += (PI_PC_mask==celllabel).astype(np.uint16) * running_label
                combined_nucleus_mask[PI_Nuclei_mask == nuclearlabel] = 0
                combined_nucleus_mask += (PI_Nuclei_mask==nuclearlabel).astype(np.uint16) * running_label
                running_label +=1
    print ("Number of remaining cell/nucleus pairs: " + str(running_label))

    save_file = join(args.output_path, 'PREDICTIONS.mat')
    savemat(save_file, {'cells': combined_cell_mask, 'nuclei': combined_nucleus_mask})

if __name__ == '__main__':
    main()