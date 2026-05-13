Automated analysis tools for fetal and neonatal brain MRI
====================

This repository contains DL pipelines for [MONAI](https://github.com/Project-MONAI/MONAI)-based automated analysis for fetal and neonatal brain MRI.


- The repository, scripts and models were designed and created at the Department of Early Life Imaging, King's College London.

  
- Please email alena.uus (at) kcl.ac.uk if in case of any questions.



Development of these processing and analysis tools was supported by projects led by Prof Mary Rutherford, Dr Lisa Story, Prof Tomoki Arichi, Prof David Edwards and Prof Jo Hajnal.



<img src="info/multi-bounti-3t-full.jpg" alt="AUTOSVRTKEXAMPLE" height="400" align ="center" />




Auto processing scripts 
------------------------


**The automated docker tags are _fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd_ (AMD systems only)**


**AUTOMATED Multi-BOUNTI SEGMENTATION FOR 3D T2W BRAIN MRI:**

*Input data requirements:*
- sufficient SNR and image quality, no extreme shading artifacts
- good quality 3D SVR
- fetal TE=250ms - dHCP protocol 
- full ROI coverage
- standard radiological space
- 25-45 weeks PMA: neonatal
- 22-39 weeks GA: fetal
- no extreme structural anomalies
- 3T
- **0.5mm resolution (please resample or run with 0.5mm reconstructi for all images before running segmentation)**
  
Note: you will need >16GB GPU


**PLEASE RUN IT DIRECTLY VIA OUR DOCKER:**


```bash

docker pull fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd


#auto Multi-BOUNTI brain tissue segmentation: fetal
docker run --rm --gpus all --mount type=bind,source=LOCATION_ON_YOUR_MACHINE,target=/home/data  fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd sh -c ' bash /home/perinatal-brain-mri-analysis/scripts/run-multi-bounti-fetal-brain-segmentation-2026.sh [/home/data/path_to_t2w_recon.nii.gz] [/home/data/path_to_tmp_processing_folder] [/home/data/path_to_output_multi_tissue_bounti_label.nii.gz]  ; '


#auto Multi-BOUNTI brain tissue segmentation: neonatal
docker run --rm --gpus all --mount type=bind,source=LOCATION_ON_YOUR_MACHINE,target=/home/data  fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd sh -c ' bash /home/perinatal-brain-mri-analysis/scripts/run-multi-bounti-neo-brain-segmentation-2026.sh [/home/data/path_to_t2w_recon.nii.gz] [/home/data/path_to_tmp_processing_folder] [/home/data/path_to_output_multi_tissue_bounti_label.nii.gz]  ; '


#volumetry reporting for Multi-BOUNTI in .html: fetal
docker run --rm  --mount type=bind,source=LOCATION_ON_YOUR_MACHINE,target=/home/data  fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd sh -c ' python3 /home/perinatal-brain-mri-analysis/scripts/scripts/auto-reporting-multi-bounti-brain-volumetry-fetal.py CASE_ID GA DATE /home/data/vol-test/brain-svr-file.nii.gz /home/data/brain-tissue-segmenation-file.nii.gz /home/data/name-for-volumetry-report.html ; chmod 777 /home/data/name-for-volumetry-report.html  '


#volumetry reporting for Multi-BOUNTI in .html: neonatal
docker run --rm  --mount type=bind,source=LOCATION_ON_YOUR_MACHINE,target=/home/data  fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd sh -c ' python3 /home/perinatal-brain-mri-analysis/scripts/scripts/auto-reporting-multi-bounti-brain-volumetry-neo.py CASE_ID GA DATE /home/data/vol-test/brain-svr-file.nii.gz /home/data/brain-tissue-segmenation-file.nii.gz /home/data/name-for-volumetry-report.html ; chmod 777 /home/data/name-for-volumetry-report.html  '


```




**AUTOMATED SVR RECONSTRUCTION FOR NEONATAL T2W BRAIN MRI:**

*Input data requirements:*
- 2-4 T2w stacks
- no extreme motion artifacts (better to exclude low quality stacks)
- sufficient ROI oversampling
- template selection: best quality stack with full brain coverage
- please run with 0.5mm output resolution

**PLEASE RUN IT DIRECTLY VIA OUR DOCKER:**



```bash

docker pull fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd


#auto neonatal brain SVR reconstruction (Kuklisova-Murgasova,2012)

docker run --rm --mount type=bind,source=LOCATION_ON_YOUR_MACHINE,target=/home/data  fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd sh -c ' cd /home/data ; mkdir out ; cd out ; mirtk reconstruct ../name_for_output_svr.nii.gz [number_of_stacks; e.g., 3] ../input_stack1.nii.gz ../input_stack2.nii.gz ../input_stackN.nii.gz -default_thickness [slice_thickness; e.g., 3.0 -resolution 1.0 -iterations 2 -sr_iterations 3 -remove_black_background -svr_only -template ../template_stack.nii.gz ; chmod 777 ../name_for_output_svr.nii.gz ;  '


```

License
-------

The code and model weights are distributed under the terms of the
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation version 3 of the License. 

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.


Citation and acknowledgements
-----------------------------

In case you found this repository useful please give appropriate credit to the software.


**Multi-BOUNTI segmentation:**
> Uus, A., Fukami-Gartner, A., Kyriakopoulou, V., Cromb, D., Morgan, T., Arulkumaran, S., Egloff Collado, A., Luis, A., Bos, R., Makropoulos, A., Schuh, A., Robinson, E., Sousa, H., Deprez, M., Cordero-Grande, L., Bradshaw, C., Colford, K., Hutter, J., Price, A., O’Muircheartaigh, J., Hammers, A., Rueckert, D., Counsell, S., McAlonan, G., Arichi, T., Edwards, A. D., Hajnal, J. V., Rutherford, M. A., Story, L. (2026). Multi-BOUNTI: Multi-lobe Brain vOlUmetry and segmeNtation for feTal and neonatal MRI. medRxiv, 2026.04.21.26351376. https://doi.org/10.64898/2026.04.21.26351376

**Neonatal brain reconstruction:**
> Kuklisova-Murgasova, M., Quaghebeur, G., Rutherford, M. A., Hajnal, J. V., & Schnabel, J. A. (2012). Reconstruction of fetal brain MRI with intensity matching and complete outlier removal. Medical Image Analysis, 16(8), 1550–1564.: https://doi.org/10.1016/j.media.2012.07.004


Disclaimer
-------

This software has been developed for research purposes only, and hence should not be used as a diagnostic tool. In no event shall the authors or distributors be liable to any direct, indirect, special, incidental, or consequential damages arising of the use of this software, its documentation, or any derivatives thereof, even if the authors have been advised of the possibility of such damage.

