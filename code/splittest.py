import cv2
import numpy as np
import os
import openslide
from PIL import Image
# dir='/home/cad429/code/data'
# train_dir=os.path.join(dir,'train_images')
# train_mask_dir=os.path.join(dir,'train_label_masks')
# print(train_mask_dir)
#数据来源为radboud的一个数据的地址，切割该图片
#test_Image=os.path.join(train_dir,'fbc0c3145008a420b3b038e8820aba1b'+'.tiff')
#test_label=os.path.join(train_mask_dir,'fbc0c3145008a420b3b038e8820aba1b'+'.tiff')

# def cut(type,image_dir,mask_dir,patch_dir,cut_size,spacing_size):
#     slide = openslide.OpenSlide(test_Image)
#     IM_rows=slide.dimensions[1] # 图片的高
#     Im_cols=slide.dimensions[0] #图片的宽
#     Image.MAX_IMAGE_PIXELS = None
#
#     mask=openslide.OpenSlide(test_label)
#     mask_rows=mask.dimensions[1] #mask 的高
#     mask_cols=mask.dimensions[0] # mask的宽
#     region=np.array(mask.read_region((0,0),0,(mask_cols,mask_rows)))
#
#
#
#     print(IM_rows,Im_cols)
#     print(region.size())
# cut('radboud',test_Image,test_label,splist_dir,512,512)
#label
mask=openslide.OpenSlide('/home/cad429/code/data/train_label_masks/000920ad0b612851f8e01bcc880d9b3d_mask.tiff')
#image
slide=openslide.OpenSlide('/home/cad429/code/data/train_images/000920ad0b612851f8e01bcc880d9b3d.tiff')
#%%

IM_rows=slide.dimensions[1] # 图片的高
Im_cols=slide.dimensions[0] #图片的宽
Image.MAX_IMAGE_PIXELS = None

mask_rows=mask.dimensions[1] #mask 的高
mask_cols=mask.dimensions[0] # mask的宽
print(Im_cols,IM_rows,mask_cols,mask_rows)


#加载了image和label之后开始切图片，将mask转换成对应的数组。

mask_data=mask.read_region((0,0),0,(mask_cols,mask_rows))
mask_array=np.asarray(mask_data)[:,:,0]
#print(mask_array)
print(mask_array.shape)
cut_size=512
space_size=512
area=cut_size*cut_size*3/4
split_path='/home/cad429/code/data/splittest'
# #开始切割图片：
for i in range(0,IM_rows-cut_size,space_size):
    for j in range(0,Im_cols-cut_size,space_size):
        tumor_mask_np=mask_array[i:i+cut_size,j:j+cut_size]
        if (tumor_mask_np == 1).sum()>= area:
            save_name='{}_{}_{}_{}.png'.format('4',i,j,cut_size)
            save_path0=os.path.join(split_path,'0',save_name)
            try:
                img=np.array(slide.read_region((j,i),0,(cut_size,cut_size)))
            except:
                slide.close()
                mask=openslide.OpenSlide('/home/cad429/code/data/train_label_masks/066f41ab89acaec2ceb40a01b66cd48b_mask.tiff')
                continue
            Image.fromarray(img).save(save_path0)
        elif (tumor_mask_np == 2).sum() >= area:
            save_name = '{}_{}_{}_{}.png'.format('4', i, j, cut_size)
            save_path3 = os.path.join(split_path, '3', save_name)
            try:
                img = np.array(slide.read_region((j, i), 0, (cut_size, cut_size)))
            except:
                slide.close()
                mask = openslide.OpenSlide(
                    '/home/cad429/code/data/train_label_masks/066f41ab89acaec2ceb40a01b66cd48b_mask.tiff')
                continue
            Image.fromarray(img).save(save_path3)