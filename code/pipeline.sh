#!/bin/bash 

# Set Path to code and data
PIPELINE_BASE_PATH=/workspace/code
DATA_DIR=/workspace/data



# Activate/deactivate parts of the pipeline
RUN_CLEANUP=false
RUN_PREPROCESSING=false
RUN_PC_SEGMENTATION=false
RUN_NUCLEI_SEGMENTATION=false
RUN_POSTPROCESSING=false
SELECT_ROIS=false
EXTRACT_SINGLECELLOBJECTS=false
RUN_FEATURE_EXTRACTION=false
RUN_NORMALIZATION=true

if [ $RUN_CLEANUP = true ]
then
	# Clean up previous predictions
	echo "Cleaning up ..."
	ANALYSIS="$DATA_DIR/analysis"
	if [ -d $ANALYSIS ]; 
	then 
		rm -Rf $ANALYSIS; 
	fi
	re="[0-9]_FoV"
	for DATA in $(find $DATA_DIR -maxdepth 2 -type d)
	do
	 	if [[ $DATA =~ $re ]]; then
			if [ -d "$DATA/processed" ]; then
				rm -Rf "$DATA/processed";
			fi
			if [ -d "$DATA/w_raw" ]; then
				rm -Rf "$DATA/w_raw";
			fi	
		fi
	done
fi

if [ $RUN_PREPROCESSING = true ]
then
	echo "Run preprocessing ..."
	# RAW DATA PROCESSING - DARIA
	cd /workspace/code/preprocessing

	python process_images.py --path $DATA_DIR

	python combine_all_images.py --in_path $DATA_DIR --out_path /workspace/processing_tmp

	# Run Cidre
	/workspace/cidre_tmp/cidre/cidre/trunk/compiled/linux64bit/run_cidre.sh /usr/local/MATLAB/MATLAB_Compiler_Runtime/v81 '/workspace/processing_tmp/*.tif' /workspace/cidre_tmp/destination 0 0.17 0.1 16 0.25 10000
	
	python structure_cidre_images.py --in_path /workspace/cidre_tmp/destination --out_path $DATA_DIR
	echo "Preprocessing finished."
fi

DATA_DIR_BASE=/workspace/data
re="[0-9]_FoV"

for DATA_DIR in $(find $DATA_DIR_BASE -maxdepth 2 -type d)
do

    	if [[ $DATA_DIR =~ $re ]]; then 
		echo "Processing $DATA_DIR..."
		if [ $RUN_PC_SEGMENTATION = true ]
		then
			# PHASE CONTRAST SEGMENTATION
			NUCLEI_FILE_FOLDER="$DATA_DIR/processed/cut/phase"
			PREDICT_NUCLEI_TO="$DATA_DIR/processed/phase_preds"
			PREPARE_FOR_PC_SEGM="$PIPELINE_BASE_PATH/segmentation/MELC_PreparePCSegmentation.py"
			# prepare - copy phase contrast image into a single folder to be able run the pipeline
			python $PREPARE_FOR_PC_SEGM --input_path $NUCLEI_FILE_FOLDER --output_path $PREDICT_NUCLEI_TO
			# segment nuclei
			ENDING="tif"
			EXPAND_MASKS=1 # for 63x objective approx. 10, for 10 or 20x approx. 3
			SCALESIZE=0.1 # For 63x objective, approximately 0.8, for 10 or 20x approximately 0.2
			SAMPLINGRATE=5 # for 63x objective approx. 12, for 10 or 20x approx. 5
			/workspace/code/segmentation/analyze_sample.sh "$PREDICT_NUCLEI_TO" "$ENDING" "$EXPAND_MASKS" "$SCALESIZE" "$SAMPLINGRATE" "maskrcnn_melc_naturalnuclei_dataset_notscaled_gold" 0
		fi
		if [ $RUN_NUCLEI_SEGMENTATION = true ]
		then
			#NUCLEI SEGMENTATION
			NUCLEI_FILE_FOLDER="$DATA_DIR/processed/cut/fluor_cidre"
			PREDICT_NUCLEI_TO="$DATA_DIR/processed/fluor_preds"
			PREPARE_FOR_NUC_SEGM="$PIPELINE_BASE_PATH/segmentation/MELC_PrepareNucleiSegmentation.py"
			# prepare - copy nuclear image into a single folder to be able run the pipeline
			python $PREPARE_FOR_NUC_SEGM --input_path $NUCLEI_FILE_FOLDER --output_path $PREDICT_NUCLEI_TO
			# segment nuclei
			ENDING="tif"
			EXPAND_MASKS=1 # for 63x objective approx. 10, for 10 or 20x approx. 3
			SCALESIZE=0.11 # For 63x objective, approximately 0.8, for 10 or 20x approximately 0.2
			SAMPLINGRATE=5 # for 63x objective approx. 12, for 10 or 20x approx. 5
        		ENV_NAME=maskrcnn
			/workspace/code/segmentation/analyze_sample.sh "$PREDICT_NUCLEI_TO" "$ENDING" "$EXPAND_MASKS" "$SCALESIZE" "$SAMPLINGRATE" "specific_maskrcnn_artificialandnaturalnuclei_dataset_scaled_10000_2000" 1
		fi
		if [ $RUN_POSTPROCESSING = true ]
		then
			#PHASE CLEANING AND PAIRING
			FLUOR_PREDICTIONS="$DATA_DIR/processed/fluor_preds"
			PHASE_PREDICTIONS="$DATA_DIR/processed/phase_preds"
			PHASE_SOURCE="$DATA_DIR/processed/cut/phase"
			EXPORT_FILTERED_PREDICTIONS_TO="$DATA_DIR/processed"
			PROCESS_PREDICTIONS="$PIPELINE_BASE_PATH/segmentation/MELC_ProcessPredictions_daria_v2.py"
			python $PROCESS_PREDICTIONS --phase_pred_path $PHASE_PREDICTIONS --fluor_pred_path $FLUOR_PREDICTIONS --phase_source_path $PHASE_SOURCE --output_path $EXPORT_FILTERED_PREDICTIONS_TO
		fi
		if [ $SELECT_ROIS = true ]
		then
			SELECT_ROIS="$PIPELINE_BASE_PATH/segmentation/select_RoI.py"
			python $SELECT_ROIS --path $DATA_DIR_BASE --path_preselected_roi $DATA_DIR_BASE/RoI
		fi
	fi     
done


re="BM"
for DATA_DIR in $(find $DATA_DIR_BASE -maxdepth 1 -type d)
do
    	if [[ $DATA_DIR =~ $re ]]; then 		
		if [ $EXTRACT_SINGLECELLOBJECTS = true ]
		then
			GENERATE_SINGLECELLOBJECTS="$PIPELINE_BASE_PATH/feature_extraction/generation_multi_channel_sc_objects.py"
			python $GENERATE_SINGLECELLOBJECTS --path $DATA_DIR
		fi
	fi
done

if [ $RUN_FEATURE_EXTRACTION = true ]
then
	echo "Run feature extraction ..."
	EXTRACT_FEATURES="$PIPELINE_BASE_PATH/feature_extraction/extract_features.py"
	python $EXTRACT_FEATURES --path $DATA_DIR_BASE
	echo "Feature extraction finished."
fi


if [ $RUN_NORMALIZATION = true ]
then
	echo "Run normalization ..."
	NORMALIZE_FEATURES="$PIPELINE_BASE_PATH/normalization/normalize.py"
	python $NORMALIZE_FEATURES --path_feature_matrix $DATA_DIR_BASE/analysis/feature_matrix_large.csv --path_marker_status $DATA_DIR_BASE/marker_status.csv --output_path $DATA_DIR_BASE/analysis #--thresholds_predicted
	echo "Normalization finished."
	echo "Pipeline finished. Generated FCS feature files are located here: $DATA_DIR/analysis"
fi

