# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 18:26:10 2022

@author: user
"""
"""
Kutay Özbay 270201017
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img0 = cv.imread('goldengate-00.png')

gray= cv.cvtColor(img0,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_0, descriptors_0 = sift.detectAndCompute(img0,None)





img1 = cv.imread('goldengate-01.png')

gray= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)




img2 = cv.imread('goldengate-02.png')

gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)




img3 = cv.imread('goldengate-03.png')

gray= cv.cvtColor(img3,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_3, descriptors_3 = sift.detectAndCompute(img3,None)



img4 = cv.imread('goldengate-04.png')

gray= cv.cvtColor(img4,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_4, descriptors_4 = sift.detectAndCompute(img4,None)




img5 = cv.imread('goldengate-05.png')

gray= cv.cvtColor(img5,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_5, descriptors_5 = sift.detectAndCompute(img5,None)

img5 = cv.drawKeypoints(gray,keypoints_5,img5)




index_params = dict(algorithm = 0, trees = 5)
search_params = dict()
 

flann = cv.FlannBasedMatcher(index_params, search_params)

matches= flann.knnMatch(descriptors_0, descriptors_1, k=2)

good_points=[]
 
for m, n in matches:
    if(m.distance < 0.6*n.distance):
        good_points.append(m)
        

query_pts = np.float32([keypoints_0[m.queryIdx]
                 .pt for m in good_points]).reshape(-1, 1, 2)
 

train_pts = np.float32([keypoints_1[m.trainIdx]
                 .pt for m in good_points]).reshape(-1, 1, 2)
 

matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
 

matches_mask = mask.ravel().tolist()

h, w = img0.shape[:2]
 

pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
 

dst = cv.perspectiveTransform(pts, matrix)
homography = cv.polylines(img1, [np.int32(dst)], True, (255, 0, 0), 3)
 


f = open("h_00_01.txt", "w")

for point in homography:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)

f.close()





matches= flann.knnMatch(descriptors_1, descriptors_2, k=2)

good_points=[]
 
for m, n in matches:
    if(m.distance < 0.6*n.distance):
        good_points.append(m)
        

query_pts = np.float32([keypoints_1[m.queryIdx]
                 .pt for m in good_points]).reshape(-1, 1, 2)
 

train_pts = np.float32([keypoints_2[m.trainIdx]
                 .pt for m in good_points]).reshape(-1, 1, 2)
 

matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
 

matches_mask = mask.ravel().tolist()

h, w = img1.shape[:2]
 

pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
 

dst = cv.perspectiveTransform(pts, matrix)
homography = cv.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3)
 


f = open("h_01_02.txt", "w")

for point in homography:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)

f.close()



        

matches= flann.knnMatch(descriptors_2, descriptors_3, k=2)

good_points=[]
 
for m, n in matches:
    if(m.distance < 0.6*n.distance):
        good_points.append(m)
        

query_pts = np.float32([keypoints_2[m.queryIdx]
                 .pt for m in good_points]).reshape(-1, 1, 2)
 

train_pts = np.float32([keypoints_3[m.trainIdx]
                 .pt for m in good_points]).reshape(-1, 1, 2)
 

matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
 

matches_mask = mask.ravel().tolist()

h, w = img2.shape[:2]
 

pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
 

dst = cv.perspectiveTransform(pts, matrix)
homography = cv.polylines(img3, [np.int32(dst)], True, (255, 0, 0), 3)
 


f = open("h_02_03.txt", "w")

for point in homography:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)

f.close()

matches= flann.knnMatch(descriptors_3, descriptors_4, k=2)

good_points=[]
 
for m, n in matches:
    if(m.distance < 0.6*n.distance):
        good_points.append(m)
        

query_pts = np.float32([keypoints_3[m.queryIdx]
                 .pt for m in good_points]).reshape(-1, 1, 2)
 

train_pts = np.float32([keypoints_4[m.trainIdx]
                 .pt for m in good_points]).reshape(-1, 1, 2)
 

matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
 

matches_mask = mask.ravel().tolist()

h, w = img3.shape[:2]
 

pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
 

dst = cv.perspectiveTransform(pts, matrix)
homography = cv.polylines(img4, [np.int32(dst)], True, (255, 0, 0), 3)
 


f = open("h_03_04.txt", "w")

for point in homography:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)

f.close()



matches= flann.knnMatch(descriptors_4, descriptors_5, k=2)

good_points=[]
 
for m, n in matches:
    if(m.distance < 0.6*n.distance):
        good_points.append(m)
        

query_pts = np.float32([keypoints_4[m.queryIdx]
                 .pt for m in good_points]).reshape(-1, 1, 2)
 

train_pts = np.float32([keypoints_5[m.trainIdx]
                 .pt for m in good_points]).reshape(-1, 1, 2)
 

matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
 

matches_mask = mask.ravel().tolist()

h, w = img4.shape[:2]
 

pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
 

dst = cv.perspectiveTransform(pts, matrix)
homography = cv.polylines(img5, [np.int32(dst)], True, (255, 0, 0), 3)
 


f = open("h_04_05.txt", "w")

for point in homography:
      p = str(point[0]) + "," + str(point[1]) + "\n"
      f.write(p)

f.close()






