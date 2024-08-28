'''
This script calculates the difference between two faces to determine whether both images are of the same person or of two different people
From: https://krasserm.github.io/2018/02/07/deep-face-recognition/#:~:text=Keras%20is%20used%20for%20implementing,further%20experiment%20with%20the%20notebook.
Using OpenFace and Dlib
'''

import numpy as np
import cv2

from model import create_model
from align import AlignDlib


def distance(emb1, emb2):
	return np.sum(np.square(emb1 - emb2))

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def load_image(path):
	img = cv2.imread(path, 1)
	# OpenCV loads images with color channels in BGR order. So we need to reverse them
	return img[...,::-1]


#Load models
alignment = AlignDlib('landmarks.dat')
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('nn4.small2.v1.h5')


img_s = ["2.jpg", "1.jpg"]


embedded = np.zeros((2, 128))
for i in enumerate(img_s):
	img = load_image(i[1])
	img = align_image(img)
	# scale RGB values to interval [0,1]
	img = (img / 255.).astype(np.float32)
	
	# obtain embedding vector for image
	embedded[i[0]] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


print(distance(embedded[0], embedded[1]))
