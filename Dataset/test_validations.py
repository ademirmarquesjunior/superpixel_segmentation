# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:13:12 2022

@author: adejunior
"""


from sklearn.metrics import f1_score

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import glob


base_images = glob.glob('*.tif')


wd = "D:/Coding/SLIC_tree_segmentation/Testes superpixels"

folder1 = wd + "/O1_interpretations/"
folder2 = wd + "/O2_interpretations/"
folder3 = wd + "/O3_interpretations/"



images1 = glob.glob(folder1 + '*.png')
image_list_1 = []

for ref in images1:
    base_image = Image.open(ref)
    image_array = np.asarray(base_image)
    
    mask = np.uint8(image_array[:,:,0]/255)
    image_list_1.append(mask)


images2 = glob.glob(folder2 + '*.png')
image_list_2 = []

for ref in images2:
    base_image = Image.open(ref)
    image_array = np.asarray(base_image)
    
    mask = np.uint8(image_array[:,:,0]/255)
    image_list_2.append(mask)
    
    
    
images3 = glob.glob(folder3 + '*.png')
image_list_3 = []

for ref in images3:
    base_image = Image.open(ref)
    image_array = np.asarray(base_image)
    
    mask = np.uint8(image_array[:,:,0]/255)
    image_list_3.append(mask)
    
   
f1_score_1_2 = []
for i in range(0, 20):
    f1_score_1_2.append(f1_score(image_list_1[i].flatten(), image_list_2[i].flatten(), average='weighted'))
    
f1_score_1_3 = []
for i in range(0, 20):
    f1_score_1_3.append(f1_score(image_list_1[i].flatten(), image_list_3[i].flatten(), average='weighted'))
    

f1_score_2_3 = []
for i in range(0, 20):
    f1_score_2_3.append(f1_score(image_list_2[i].flatten(), image_list_3[i].flatten(), average='weighted'))
    
    
print("F1 score user 1 and 2" + str(np.mean(f1_score_1_2)))
print("F1 score user 1 and 3" + str(np.mean(f1_score_1_3)))
print("F1 score user 2 and 3" + str(np.mean(f1_score_2_3)))

print("Standard deviation F1 score user 1 and 2" + str(np.std(f1_score_1_2)))
print("Standard deviation F1 score user 1 and 3" + str(np.std(f1_score_1_3)))
print("Standard deviation F1 score user 2 and 3" + str(np.std(f1_score_2_3)))


f1_score_1_res = []
f1_score_2_res = []
f1_score_3_res = []
for image_id in range(0,20):
    image_sum = np.uint8(image_list_1[image_id])+np.uint8(image_list_2[image_id])+np.uint8(image_list_3[image_id])
    res_image = np.uint8((image_sum>2)*255)
    Image.fromarray(res_image).save(wd + "/consolidated_dataset/" + str(image_id)+'.png')
    f1_score_1_res.append(f1_score(res_image.flatten(), image_list_1[image_id].flatten(), average='weighted'))
    f1_score_2_res.append(f1_score(res_image.flatten(), image_list_2[image_id].flatten(), average='weighted'))
    f1_score_3_res.append(f1_score(res_image.flatten(), image_list_3[image_id].flatten(), average='weighted'))
    

plt.imshow((image_sum>2)*1)


print("F1 score user 1 and consolidated data" + str(np.mean(f1_score_1_res)))
print("F1 score user 2 and consolidated data" + str(np.mean(f1_score_2_res)))
print("F1 score user 3 and consolidated data" + str(np.mean(f1_score_3_res)))

print("Standard deviation F1 score user 1 and consolidated data" + str(np.std(f1_score_1_res)))
print("Standard deviation F1 score user 2 and consolidated data" + str(np.std(f1_score_2_res)))
print("Standard deviation F1 score user 3 and consolidated data" + str(np.std(f1_score_3_res)))
        