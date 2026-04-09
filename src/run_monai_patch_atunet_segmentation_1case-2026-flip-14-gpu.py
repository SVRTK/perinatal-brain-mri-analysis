

import numpy as np
from scipy.ndimage import zoom
import nibabel as nib

#import skimage
import matplotlib.pyplot as plt

from scipy import ndimage
#from skimage.measure import label, regionprops

import sys
import os

import torch
import monai
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, AttentionUnet
from monai.transforms import Flip

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

#to_tensor = ToTensor()
#to_numpy = ToNumpy()



res = int(sys.argv[1])

cl_num = int(sys.argv[2])

model_weights_path_unet = sys.argv[3]

input_img_name=sys.argv[4]

output_lab_name=sys.argv[5]


#print(" - loading image")

global_img = nib.load(input_img_name)

input_matrix_image_data = global_img.get_fdata()

input_image = torch.tensor(input_matrix_image_data).unsqueeze(0)

scaler = monai.transforms.ScaleIntensity(minv=0.0, maxv=1.0)
final_image = scaler(input_image)



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = torch.device('cpu')
#map_location = torch.device('cpu')
 

#print(" - defining the model")


segmentation_model = AttentionUnet(spatial_dims=3,
                    in_channels=1,
                    out_channels=cl_num+1,
                    channels=(32, 64, 128, 256, 512),
                    strides=(2,2,2,2),
                    kernel_size=3,
                    up_kernel_size=3,
                    dropout=0.5).to(device)


#print(" - loading the model")

with torch.no_grad():
  segmentation_model.load_state_dict(torch.load(model_weights_path_unet), strict=False)
  #segmentation_model.to(device)
  segmentation_model.eval()


print(" - running segmentation : ", output_lab_name)

flp_run = Flip(1)

segmentation_inputs = final_image.unsqueeze(0).cuda()

fl_segmentation_inputs = torch.unsqueeze(final_image, 1).cuda()
fl_segmentation_inputs = flp_run(fl_segmentation_inputs)



def replace_dhcp(fl_val_outputs):
    
        org_fl_val_outputs = fl_val_outputs.clone();

        i_org = 1 ; i_fl = 2 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 2 ; i_fl = 1 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 3 ; i_fl = 4 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 4 ; i_fl = 3 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 5 ; i_fl = 6 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 6 ; i_fl = 5 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 7 ; i_fl = 8 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 8 ; i_fl = 7 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 9 ; i_fl = 10 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 10 ; i_fl = 9 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 11 ; i_fl = 12 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 12 ; i_fl = 11 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 13 ; i_fl = 14 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 14 ; i_fl = 13 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        

        return org_fl_val_outputs


with torch.no_grad():

    segmentation_output = sliding_window_inference(segmentation_inputs, (128, 128, 128), 4, segmentation_model, overlap=0.8)
    
    fl_segmentation_output = sliding_window_inference(fl_segmentation_inputs, (128, 128, 128), 4, segmentation_model, overlap=0.8)
    fl_segmentation_output_tmp = flp_run(fl_segmentation_output.clone())
    fl_segmentation_output_final = replace_dhcp(fl_segmentation_output_tmp.clone())
    
    sum_segmentation_outputs = (segmentation_output.clone() + fl_segmentation_output_final.clone()) / 2.0
    
#    segmentation_output = segmentation_model(segmentation_inputs)


label_output = torch.argmax(sum_segmentation_outputs, dim=1).detach().cpu()[0, :, :, :]
label_matrix = label_output.cpu().numpy()


#print(" - saving results")

img_tmp_info = nib.load(input_img_name)
out_lab_nii = nib.Nifti1Image(label_matrix, img_tmp_info.affine, img_tmp_info.header)
nib.save(out_lab_nii, output_lab_name)


