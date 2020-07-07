import matplotlib.pyplot as plt
import openslide
import os
import matplotlib
import numpy as np
mask = openslide.OpenSlide('/home/cad429/code/data/train_label_masks/066f41ab89acaec2ceb40a01b66cd48b_mask.tiff')
mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

plt.figure()
plt.title("Mask with default cmap")
plt.imshow(np.asarray(mask_data)[:,:,0], interpolation='nearest')
plt.show()

plt.figure()
plt.title("Mask with custom cmap")
# Optional: create a custom color map
cmap = matplotlib.colors.ListedColormap(['black', 'pink', 'green','yellow','red','orange'])
plt.imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
plt.show()

mask.close()