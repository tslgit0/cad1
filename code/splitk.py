import numpy
import openslide
import os
from PIL import Image
import pandas as pd
import numpy as np
import csv
kimage_csv_path='/home/cad429/code/data/k.csv'


k_image_path='/home/cad429/code/data/train_images'
k_image_label_path='/home/cad429/code/data/train_label_masks'
k_image_patch_path='/home/cad429/code/data/splitk'
cut_size=512
space_size=512
area=cut_size*cut_size*3/4
def cut(image_path,mask_path,cut_size,space_size,patch_path,image_name):
    mask=openslide.OpenSlide(mask_path)
    slide=openslide.OpenSlide(image_path)
    IM_rows = slide.dimensions[1]  # 图片的高
    Im_cols = slide.dimensions[0]  # 图片的宽
    Image.MAX_IMAGE_PIXELS = None

    mask_rows = mask.dimensions[1]  # mask 的高
    mask_cols = mask.dimensions[0]  # mask的宽

    mask_data = mask.read_region((0, 0), 0, (mask_cols, mask_rows))
    mask_array = np.asarray(mask_data)[:, :, 0]

    for i in range(0, IM_rows - cut_size, space_size):
        for j in range(0, Im_cols - cut_size, space_size):
            tumor_mask_np = mask_array[i:i + cut_size, j:j + cut_size]
            if (tumor_mask_np == 1).sum() >= area:#健康
                save_name = '{}_{}_{}_{}.png'.format(image_name, i, j, cut_size)
                save_path1 = os.path.join(patch_path, '1', save_name)
                try:
                    img = np.array(slide.read_region((j, i), 0, (cut_size, cut_size)))
                except:
                    slide.close()
                    mask = openslide.OpenSlide(mask_path)
                    continue
                Image.fromarray(img).save(save_path1)
            elif (tumor_mask_np == 2).sum() >= area:#有病
                save_name = '{}_{}_{}_{}.png'.format(image_name, i, j, cut_size)
                save_path2 = os.path.join(patch_path, '2', save_name)
                try:
                    img = np.array(slide.read_region((j, i), 0, (cut_size, cut_size)))
                except:
                    slide.close()
                    mask = openslide.OpenSlide(mask_path)
                    continue
                Image.fromarray(img).save(save_path2)

            elif(tumor_mask_np == 0).sum() >= area:#背景
                save_name = '{}_{}_{}_{}.png'.format(image_name, i, j, cut_size)
                save_path0 = os.path.join(patch_path, '0', save_name)
                try:
                    img = np.array(slide.read_region((j, i), 0, (cut_size, cut_size)))
                except:
                    slide.close()
                    mask = openslide.OpenSlide(mask_path)
                    continue
                Image.fromarray(img).save(save_path0)



with open(kimage_csv_path,'r') as csvfile:
    reader = csv.reader(csvfile)
    for i,rows in enumerate(reader):
        if i==0:  #  第一行为lie
            continue
        #print(rows[0])
        if i==3:
            image_path=os.path.join(k_image_path,rows[0]+'.tiff')
            image_mask=os.path.join(k_image_label_path,rows[0]+'_mask.tiff')
            print(image_mask)
            print(image_path)
            cut(image_path=image_path,mask_path=image_mask,cut_size=cut_size,space_size=space_size,patch_path=k_image_patch_path,image_name=rows[0])
