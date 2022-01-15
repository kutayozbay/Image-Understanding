# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:29:53 2022

@author: user
"""
"""
Kutay Özbay 270201017
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#Image goldengate-00.png
img0 = cv.imread('goldengate-00.png')

gray= cv.cvtColor(img0,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_0, descriptors_0 = sift.detectAndCompute(img0,None)

img0 = cv.drawKeypoints(gray,keypoints_0,img0)

cv.imwrite('sift_keypoints_00.png',img0)

f = open("sift_keypoints_00.txt", "w")

for point in keypoints_0:
      p = str(point.pt[0]) + "," + str(point.pt[1]) + "\n"
      f.write(p)

f.close()

f = open("sift_descriptors_00.txt", "w")

for point in descriptors_0:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)

f.close()


#Image goldengate-01.png
img1 = cv.imread('goldengate-01.png')

gray= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

img1 = cv.drawKeypoints(gray,keypoints_1,img1)

cv.imwrite('sift_keypoints_01.png',img1)

f = open("sift_keypoints_01.txt", "w")

for point in keypoints_1:
      p = str(point.pt[0]) + "," + str(point.pt[1]) + "\n"
      f.write(p)

f.close()

f = open("sift_descriptors_01.txt", "w")

for point in descriptors_1:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)

f.close()

#Image goldengate-02.png
img2 = cv.imread('goldengate-02.png')

gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

img2 = cv.drawKeypoints(gray,keypoints_2,img2)

cv.imwrite('sift_keypoints_02.png',img2)

f = open("sift_keypoints_02.txt", "w")

for point in keypoints_2:
      p = str(point.pt[0]) + "," + str(point.pt[1]) + "\n"
      f.write(p)

f.close()

f = open("sift_descriptors_02.txt", "w")

for point in descriptors_2:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)
f.close()

#Image goldengate-03.png
img3 = cv.imread('goldengate-03.png')

gray= cv.cvtColor(img3,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_3, descriptors_3 = sift.detectAndCompute(img3,None)

img3 = cv.drawKeypoints(gray,keypoints_3,img3)

cv.imwrite('sift_keypoints_03.png',img3)

f = open("sift_keypoints_03.txt", "w")

for point in keypoints_3:
      p = str(point.pt[0]) + "," + str(point.pt[1]) + "\n"
      f.write(p)

f.close()

f = open("sift_descriptors_03.txt", "w")

for point in descriptors_3:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)

f.close()
#Image goldengate-04.png
img4 = cv.imread('goldengate-04.png')

gray= cv.cvtColor(img4,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_4, descriptors_4 = sift.detectAndCompute(img4,None)

img4 = cv.drawKeypoints(gray,keypoints_4,img4)

cv.imwrite('sift_keypoints_04.png',img4)

f = open("sift_keypoints_04.txt", "w")

for point in keypoints_4:
      p = str(point.pt[0]) + "," + str(point.pt[1])+ "\n"
      f.write(p)

f.close()

f = open("sift_descriptors_04.txt", "w")

for point in descriptors_4:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)

f.close()

#Image goldengate-05.png
img5 = cv.imread('goldengate-05.png')

gray= cv.cvtColor(img5,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_5, descriptors_5 = sift.detectAndCompute(img5,None)

img5 = cv.drawKeypoints(gray,keypoints_5,img5)

cv.imwrite('sift_keypoints_05.png',img5)

f = open("sift_keypoints_05.txt", "w")

for point in keypoints_5:
      p = str(point.pt[0]) + "," + str(point.pt[1]) + "\n"
      f.write(p)

f.close()

f = open("sift_descriptors_05.txt", "w")

for point in descriptors_5:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)

f.close()


#feature matching
bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)



matches = bf.match(descriptors_0,descriptors_1)
matches = sorted(matches, key = lambda x:x.distance)


"""
for point in matches:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write("Tentative Correspondences: " + p)

f.close()
"""


img0_1 = cv.drawMatches(img0, keypoints_0, img1, keypoints_1, matches[:50], img1, flags=2)
cv.imwrite('tentative_correspondences_00_01.png',img0_1)



matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

"""
for point in matches:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write("Tentative Correspondences: " + p)

f.close()
"""

img1_2 = cv.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
cv.imwrite('tentative_correspondences_01_02.png',img1_2)



matches = bf.match(descriptors_2,descriptors_3)
matches = sorted(matches, key = lambda x:x.distance)

"""
for point in matches:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write("Tentative Correspondences: " + p)

f.close()
"""

img2_3 = cv.drawMatches(img2, keypoints_2, img3, keypoints_3, matches[:50], img3, flags=2)
cv.imwrite('tentative_correspondences_02_03.png',img2_3)



matches = bf.match(descriptors_3,descriptors_4)
matches = sorted(matches, key = lambda x:x.distance)

"""
for point in matches:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write("Tentative Correspondences: " + p)

f.close()
"""

img3_4 = cv.drawMatches(img3, keypoints_3, img4, keypoints_4, matches[:50], img4, flags=2)
cv.imwrite('tentative_correspondences_03_04.png',img3_4)



matches = bf.match(descriptors_4,descriptors_5)
matches = sorted(matches, key = lambda x:x.distance)

"""
for point in matches:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write("Tentative Correspondences: " + p)

f.close()
"""

img4_5 = cv.drawMatches(img4, keypoints_4, img5, keypoints_5, matches[:50], img5, flags=2)
cv.imwrite('tentative_correspondences_04_05.png',img4_5)













