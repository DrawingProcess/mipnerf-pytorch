import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import glob

pathname = './nerf_synthetic/lego'
images = glob.glob(pathname + '/train/*.png', recursive=True)

os.makedirs('nerf_synthetic_gau/lego_gau_25_2/train/', exist_ok=True)
for image in images:
    print(image)
    img = cv2.imread(image) # 이미지 파일 읽기

    img_gau = cv2.GaussianBlur(img, (25, 25), 2)

    image = image.replace("lego", "lego_gau_25_2")
    print(image)
    cv2.imwrite(image, img_gau)