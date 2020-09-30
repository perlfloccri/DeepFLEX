#!/bin/bash

PIPELINE_BASE_PATH=/workspace/code/segmentation

PATH_DATAGEN=$PIPELINE_BASE_PATH/DataGenerator  
PATH_TISQUANTPIPELINE=$PIPELINE_BASE_PATH/Kaggle
PATH_MASKRCNN=$PIPELINE_BASE_PATH/Mask_RCNN

PYTHON_MASKRCNN=python
PYTHON_DATAGENERATOR=python
DIAGNOSIS='normal Neuroblastoma Ganglioneuroma'

MAX_EPOCHS_UNET=5
MAX_EPOCHS_MASK_RCNN=5
MAX_EPOCHS_DEEPCELL=5
COMBINED_TRAINING=0
NR_IMAGES=10000
NR_IMAGES_VAL=2000
OVERLAP_TRAINVAL=20
OVERLAP_TEST=50 # Set normal to 50, for test now to 80

DO_EVALUATION=1
USE_MASKRCNN=1
USE_DEEPCELL=0
USE_UNETTISQUANT=0
USE_UNETCLASSIC=0
GENERATE_SVGS=1

SAMPLE_FOLDERS=$1
ENDING=$2
DILATION=$3
SCALESIZE=$4
SAMPLINGRATE=$5
WEIGHTS=$6
SCALE_IMAGES=$7

if [ "$SCALE_IMAGES" -eq 1 ]; then
	SCALE_TEXT=scaled
else
	SCALE_TEXT=notscaled
fi
echo $SCALE_TEXT

RESULTS_FILE_PATH=$PIPELINE_BASE_PATH/Results/results_$SCALE_TEXT.csv
rm $RESULTS_FILE_PATH
rm $PIPELINE_BASE_PATH/tmp/*.*
TILE_FOLDER=tiles_$SCALE_TEXT

echo "Sample Folder: " 
echo $SAMPLE_FOLDERS
rm $SAMPLE_FOLDERS/*_mask.TIF

# Create tiles

python "$PATH_DATAGEN/run_createNaturalTilesForSampleSegmentation.py" --ending "$ENDING" --scalesize "$SCALESIZE" --tissue Ganglioneuroma --outputFolder "$PIPELINE_BASE_PATH/tmp" --inputFolder "$SAMPLE_FOLDERS" --scale "$SCALE_IMAGES" --mode test --resultsfile "$RESULTS_FILE_PATH" --overlap $OVERLAP_TEST


#### MASK RCNN ####
if [ $USE_MASKRCNN -eq 1 ]; then
	if [ $DO_EVALUATION -eq 1 ]; then
		# Set settings for Net Prediction
		$PYTHON_DATAGENERATOR "$PATH_TISQUANTPIPELINE/Config/Config.py" --startup 0 --net_description "$WEIGHTS" --dataset segmentSample --results_folder "$PIPELINE_BASE_PATH/tmp" --traintestmode test --netinfo maskrcnn --dataset_dirs_test "$PIPELINE_BASE_PATH/tmp"
		
		# Run maskRCNN
		$PYTHON_MASKRCNN "$PATH_MASKRCNN/segment_sample.py"

		# Reconstruct results
		WEIGHTS+="_predictions.h5" 
		$PYTHON_DATAGENERATOR $PATH_DATAGEN/run_reconstructResultMasksForSampleSegmentation.py --scale $SCALE_IMAGES --dilate "$DILATION" --resultfile $RESULTS_FILE_PATH --predictionfile "$PIPELINE_BASE_PATH/tmp/$WEIGHTS" --net maskrcnn --overlap $OVERLAP_TEST
                if [ $GENERATE_SVGS -eq 1 ]; then
		       $PYTHON_DATAGENERATOR $PATH_DATAGEN/run_curateExternalMasks.py --ending "$ENDING" --inputFolder $SAMPLE_FOLDERS --samplingrate $SAMPLINGRATE
                fi
	fi
fi
