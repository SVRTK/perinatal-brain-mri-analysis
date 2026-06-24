#!/usr/bin/env bash -l


#
# AI tools for perinatal brain MRI analysis
#
# Copyright 2026 - King's College London
#
# The auto SVRTK code and all scripts are distributed under the terms of the
# [GNU General Public License v3.0: 
# https://www.gnu.org/licenses/gpl-3.0.en.html. 
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation version 3 of the License. 
# 
# This software is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# See the GNU General Public License for more details.
#


src=/home/perinatal-brain-mri-analysis
mirtk=/bin/MIRTK/build/lib/tools


n4_run=$1
org_t2=$2
proc=$3
out_lab=$4


if [[ $# -ne 4 ]] ; then

    echo
    echo "------------------------------------------------------------"
    echo
    echo "Usage: please use the following format ..."
    echo "bash /home/perinatal-brain-mri-analysis/run-fetal-brain-bet-extraction-2026-general.sh [0/1: 0 - no N4 bias correction, 1 - with N4 bias correction] [full_path_to_input_t2w_recon.nii.gz] [full_path_to_folder_for_tmp_processing] [full_path_to_output_label.nii.gz]"
    echo
    echo "------------------------------------------------------------"
    echo
    exit

fi 



echo 
echo "------------------------------------------------------------"
echo
echo " - SCRIPT FOR FETAL BRAIN EXTRACTION PROCESSING ... "
echo
echo "------------------------------------------------------------"
echo 
  


echo
echo "------------------------------------------------------------"
echo
echo " - input t2 : " ${org_t2}
echo " - processing folder : " ${proc}
echo
echo "------------------------------------------------------------"
echo
echo " - RUNNING PREPROCESSING ... "
echo
echo "------------------------------------------------------------"
echo

if [[ ! -f ${org_t2} ]];then
    echo
    echo "------------------------------------------------------------"
    echo
    echo "ERROR: NO INPUT FILE ..."
    echo
    echo "------------------------------------------------------------"
    echo
    exit
fi



if [[ ! -d ${proc} ]];then
    mkdir ${proc}
fi

if [[ ! -d ${proc} ]];then
    echo
    echo "------------------------------------------------------------"
    echo
    echo "ERROR: CANNOT CREATE PROCESSING FOLDER ..."
    echo
    echo "------------------------------------------------------------"
    echo
    exit
fi




${mirtk}/convert-image ${org_t2} ${proc}/org-t2.nii.gz

${mirtk}/extract-image-region ${proc}/org-t2.nii.gz ${proc}/org-t2.nii.gz -Rt1 0 -Rt2 0

${mirtk}/nan ${proc}/org-t2.nii.gz 100000


${mirtk}/threshold-image ${proc}/org-t2.nii.gz ${proc}/m-t2.nii.gz 0.1 > ${proc}/t.txt

${mirtk}/extract-connected-components ${proc}/m-t2.nii.gz ${proc}/m-t2.nii.gz

${mirtk}/crop-image ${proc}/org-t2.nii.gz ${proc}/m-t2.nii.gz ${proc}/crop-org-t2.nii.gz

${mirtk}/edit-image ${src}/templates/ref-bounti/ref-082025.nii.gz ${proc}/tmp-ref.nii.gz -copy-origin ${proc}/crop-org-t2.nii.gz

${mirtk}/transform-image ${proc}/org-t2.nii.gz ${proc}/tr-org-t2.nii.gz -target ${proc}/tmp-ref.nii.gz -interp Linear

${mirtk}/nan ${proc}/tr-org-t2.nii.gz 100000

${mirtk}/crop-image ${proc}/tr-org-t2.nii.gz ${proc}/m-t2.nii.gz ${proc}/crop-tr-t2.nii.gz

${mirtk}/pad-3d ${proc}/crop-tr-t2.nii.gz ${proc}/pad-crop-tr-t2-128.nii.gz 128 1

if [[ $n4_run -ne 1 ]] ; then

	echo " ... "
	
else 

	echo
	echo " - n4 ..."
	echo

	${src}/bin/N4BiasFieldCorrection -i ${proc}/pad-crop-tr-t2-128.nii.gz  -o ${proc}/n4-pad-crop-tr-t2-128.nii.gz  > ${proc}/t.txt
	cp ${proc}/n4-pad-crop-tr-t2-128.nii.gz ${proc}/pad-crop-tr-t2-128.nii.gz
	
	
fi 



echo
echo " - brain extraction ..."
echo

#w_bet=${src}/models/atunet_bet_brain_fetal_1lab_best_metric_model.pth

w_bet=${src}/models/atunet_bet_brain_153t_svr_2026_128_best_metric_model.pth

#unset PYTHONPATH ;
python3 ${src}/src/run_monai_patch_atunet_segmentation_1case-2026-gpu.py 128 1 ${w_bet} ${proc}/pad-crop-tr-t2-128.nii.gz ${proc}/bet-lab-pad-crop-t2-128.nii.gz

${mirtk}/erode-image ${proc}/bet-lab-pad-crop-t2-128.nii.gz ${proc}/bet-lab-pad-crop-t2-128.nii.gz 

${mirtk}/extract-connected-components ${proc}/bet-lab-pad-crop-t2-128.nii.gz ${proc}/bet-lab-pad-crop-t2-128.nii.gz

${mirtk}/dilate-image ${proc}/bet-lab-pad-crop-t2-128.nii.gz ${proc}/bet-lab-pad-crop-t2-128.nii.gz 

${mirtk}/transform-image ${proc}/bet-lab-pad-crop-t2-128.nii.gz ${out_lab} -labels -target ${proc}/org-t2.nii.gz



if [[ ! -f ${out_lab} ]];then
    echo 
    echo "------------------------------------------------------------"
    echo
    echo "ERROR - LABEL FILE WAS NOT GENERATED ..."
    echo 
    echo "------------------------------------------------------------"
    echo
    exit
    
else

    # rm -r 

    echo
    echo "------------------------------------------------------------"
    echo
    echo " - output label : " ${out_lab}
    echo
    echo "------------------------------------------------------------"


fi


chmod 777 -R ${proc} ${out_lab}

 
