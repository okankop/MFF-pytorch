"""
Created the optical flow directories that you want to
write your flow images.

USAGE: 'python create_dir.py'
"""

import os
import glob
import numpy as np
from PIL import Image
import subprocess
import shutil

for ind in range(1, 148093):
    print(ind)
    file_dir = os.path.dirname("/data/jester/20bn-dataset/flow_abs_max/u/" + str(ind) + "/")
    base_dir = os.path.dirname("/data/jester/20bn-dataset/flow_abs_max/")
    dir_list = os.listdir(file_dir)
    prefix = '{:05d}.jpg'

    for i in range(1, len(dir_list) + 1 ):
        img = np.asarray(Image.open(os.path.join(file_dir, prefix.format(i))).convert('L'))
        if (np.max(img) == 0):
            if (i == 1):
                shutil.copyfile(os.path.join(base_dir, "u", str(ind), prefix.format(i+1)), os.path.join(base_dir, "u", str(ind), prefix.format(i)))
                shutil.copyfile(os.path.join(base_dir, "v", str(ind), prefix.format(i+1)), os.path.join(base_dir, "v", str(ind), prefix.format(i)))

            else:
                shutil.copyfile(os.path.join(base_dir, "u", str(ind), prefix.format(i-1)), os.path.join(base_dir, "u", str(ind), prefix.format(i)))
                shutil.copyfile(os.path.join(base_dir, "v", str(ind), prefix.format(i-1)), os.path.join(base_dir, "v", str(ind), prefix.format(i)))



