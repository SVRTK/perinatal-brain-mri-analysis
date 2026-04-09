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
    echo "bash /home/perinatal-brain-mri-analysis/run-multi-bounti-neo-brain-segmentation-2026.sh [0/1: 0 - no N4 bias correction, 1 - with N4 bias correction] [full_path_to_input_t2w_recon.nii.gz] [full_path_to_folder_for_tmp_processing] [full_path_to_output_label.nii.gz]"
    echo
    echo "------------------------------------------------------------"
    echo
    exit

fi 



echo 
echo "------------------------------------------------------------"
echo
echo " - SCRIPT FOR MULTI-BOUNTI BRAIN PROCESSING ... "
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

${mirtk}/threshold-image ${proc}/org-t2.nii.gz ${proc}/m-t2.nii.gz 0.5 > ${proc}/t.txt

${mirtk}/extract-connected-components ${proc}/m-t2.nii.gz ${proc}/m-t2.nii.gz

${mirtk}/crop-image ${proc}/org-t2.nii.gz ${proc}/m-t2.nii.gz ${proc}/crop-org-t2.nii.gz

${mirtk}/edit-image ${src}/templates/ref-bounti/ref-082025.nii.gz ${proc}/tmp-ref.nii.gz -copy-origin ${proc}/crop-org-t2.nii.gz

${mirtk}/transform-image ${proc}/org-t2.nii.gz ${proc}/tr-org-t2.nii.gz -target ${proc}/tmp-ref.nii.gz -interp Linear

${mirtk}/nan ${proc}/tr-org-t2.nii.gz 100000

${mirtk}/crop-image ${proc}/tr-org-t2.nii.gz ${proc}/m-t2.nii.gz ${proc}/crop-tr-t2.nii.gz

${mirtk}/pad-3d ${proc}/crop-tr-t2.nii.gz ${proc}/pad-crop-tr-t2-128.nii.gz 128 1

echo
echo " - brain extraction ..."
echo

w_bet=${src}/models/atunet_bet_brain_2025_neo_1lab_best_metric_model.pth

#unset PYTHONPATH ;
python3 ${src}/src/run_monai_patch_atunet_segmentation_1case-2026-gpu.py 128 1 ${w_bet} ${proc}/pad-crop-tr-t2-128.nii.gz ${proc}/bet-lab-pad-crop-t2-128.nii.gz

${mirtk}/extract-connected-components ${proc}/bet-lab-pad-crop-t2-128.nii.gz ${proc}/bet-lab-pad-crop-t2-128.nii.gz

${mirtk}/dilate-image ${proc}/bet-lab-pad-crop-t2-128.nii.gz ${proc}/dl-bet-lab-pad-crop-t2-128.nii.gz -iterations 2

${mirtk}/transform-image ${proc}/bet-lab-pad-crop-t2-128.nii.gz ${proc}/bet-lab-crop-t2.nii.gz -labels -target ${proc}/crop-tr-t2.nii.gz

if [[ $n4_run -ne 1 ]] ; then

	cp  ${proc}/crop-tr-t2.nii.gz ${proc}/n4-crop-tr-t2.nii.gz
	
else 

	echo
	echo " - n4 ..."
	echo

	${src}/bin/N4BiasFieldCorrection -i ${proc}/crop-tr-t2.nii.gz  -x ${proc}/bet-lab-crop-t2.nii.gz -o ${proc}/n4-crop-tr-t2.nii.gz > ${proc}/t.txt
	
fi 


${mirtk}/mask-image ${proc}/n4-crop-tr-t2.nii.gz ${proc}/dl-bet-lab-pad-crop-t2-128.nii.gz ${proc}/masked-n4-crop-tr-t2.nii.gz

${mirtk}/crop-image ${proc}/masked-n4-crop-tr-t2.nii.gz ${proc}/dl-bet-lab-pad-crop-t2-128.nii.gz ${proc}/masked-n4-crop-tr-t2.nii.gz

${mirtk}/pad-3d ${proc}/masked-n4-crop-tr-t2.nii.gz ${proc}/pad-masked-n4-crop-tr-t2-128.nii.gz 128 1

echo
echo " - l/r extraction & reorientation ..."
echo

w_lr=${src}/models/atunet_lr_brain_neo_2lab_best_metric_model.pth

#unset PYTHONPATH ;
python3 ${src}/src/run_monai_patch_atunet_segmentation_1case-2026-gpu.py  128 2 ${w_lr} ${proc}/pad-masked-n4-crop-tr-t2-128.nii.gz ${proc}/lr-lab-pad-masked-crop-t2-128.nii.gz


${mirtk}/edit-image ${src}/templates/ref-bounti/dl-ref-t2.nii.gz ${proc}/tmp-dl-ref.nii.gz -copy-origin ${proc}/masked-n4-crop-tr-t2.nii.gz

${mirtk}/edit-image ${src}/templates/ref-bounti/lr-lab.nii.gz ${proc}/tmp-lr-ref.nii.gz -copy-origin ${proc}/masked-n4-crop-tr-t2.nii.gz

${mirtk}/mask-image ${proc}/lr-lab-pad-masked-crop-t2-128.nii.gz ${proc}/bet-lab-pad-crop-t2-128.nii.gz ${proc}/lr-lab-pad-masked-crop-t2-128.nii.gz 


${mirtk}/register ${proc}/tmp-lr-ref.nii.gz ${proc}/lr-lab-pad-masked-crop-t2-128.nii.gz -model Affine -dofin ${src}/templates/ref-bounti/i.dof -dofout ${proc}/aff-d.dof -v 0


${mirtk}/transform-image ${proc}/bet-lab-pad-crop-t2-128.nii.gz ${proc}/dl-bet-org-t2.nii.gz -target ${proc}/crop-org-t2.nii.gz -labels

${mirtk}/dilate-image ${proc}/dl-bet-org-t2.nii.gz ${proc}/dl-bet-org-t2.nii.gz -iterations 2 

${mirtk}/mask-image ${proc}/crop-org-t2.nii.gz ${proc}/dl-bet-org-t2.nii.gz ${proc}/masked-crop-org-t2.nii.gz

${mirtk}/crop-image ${proc}/masked-crop-org-t2.nii.gz ${proc}/dl-bet-org-t2.nii.gz ${proc}/masked-crop-org-t2.nii.gz

${mirtk}/transform-image ${proc}/dl-bet-org-t2.nii.gz ${proc}/dl-bet-org-t2.nii.gz -target ${proc}/masked-crop-org-t2.nii.gz -labels

if [[ $n4_run -ne 1 ]] ; then

	cp  ${proc}/masked-crop-org-t2.nii.gz  ${proc}/n4-masked-crop-org-tr-t2.nii.gz 

else 

	echo
	echo " - n4 ..."
	echo
	
	${src}/bin/N4BiasFieldCorrection -i ${proc}/masked-crop-org-t2.nii.gz  -x ${proc}/dl-bet-org-t2.nii.gz -o ${proc}/n4-masked-crop-org-tr-t2.nii.gz > ${proc}/t.txt

fi 


${mirtk}/transform-image ${proc}/n4-masked-crop-org-tr-t2.nii.gz ${proc}/reo-n4-masked-t2.nii.gz  -target ${proc}/tmp-dl-ref.nii.gz -dofin ${proc}/aff-d.dof

${mirtk}/threshold-image ${proc}/reo-n4-masked-t2.nii.gz ${proc}/m.nii.gz 0.5 > ${proc}/t.txt

${mirtk}/crop-image ${proc}/reo-n4-masked-t2.nii.gz ${proc}/m.nii.gz ${proc}/reo-n4-masked-t2.nii.gz

${mirtk}/nan ${proc}/reo-n4-masked-t2.nii.gz  50000

${mirtk}/pad-3d ${proc}/reo-n4-masked-t2.nii.gz ${proc}/pad-reo-n4-masked-t2-256.nii.gz 256 1

echo 
echo "------------------------------------------------------------"
echo
echo " - RUNNING MULTI-BOUNTI SEGMENTATION ... "
echo
echo "------------------------------------------------------------"
echo


w_bounti=${src}/models/patch_atunet_new_multi_43_brain_neo_aff_fix_022026_best_metric_model.pth

python3 ${src}/src/run_monai_patch_atunet_segmentation_1case-2026-gpu.py 128 43 ${w_bounti} ${proc}/pad-reo-n4-masked-t2-256.nii.gz ${proc}/multi-lab-reo-t2.nii.gz


${mirtk}/transform-image ${proc}/multi-lab-reo-t2.nii.gz ${proc}/tr-multi-lab-reo-t2.nii.gz -target ${proc}/org-t2.nii.gz  -dofin_i ${proc}/aff-d.dof -labels

${mirtk}/thin-cortex-thick ${proc}/tr-multi-lab-reo-t2.nii.gz ${out_lab} > ${proc}/t.txt 



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

 
