# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:35:49 2018

@author: Bikram Pandit
"""
import cv2
import copy
import os
import sys, getopt

""" EDIT THIS """
#This dir should only contain images
imageDir = 'images/'

#This will be empty
croppedImageDir = 'images/cropped/'



""" DON'T EDIT THIS """

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        
def detect(img, cascade):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
    return rects

def drawRect(rects,image):
    init = False
    for (x, y, w, h) in rects:
        #draw rectangle in faces
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if not init:
            left = x
            top = y
            right = x+w
            bottom = y+w
        else:
            left = min(left,x)
            top = min(top,y)
            right = max(right,x+w)
            bottom = max(bottom,y+h)
        init = True
    
    
    height, width, _ = image.shape
    side = min(height,width)
    
    if len(rects) == 0:
        left = 0
        top = 0
        right = bottom = side

    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
    
    startX = (abs(left-right) - side) // 2 + left
    startX = max(0,min(startX,width - side))
    
    startY = (abs(top-bottom) - side) // 2 + top
    startY = max(0,min(startY,height - side))
    
    cv2.rectangle(image, (startX, startY), (startX+side, startY+side), (255, 0, 0), 5)
    
    return startX, startY, side
    
for imageFile in os.listdir(imageDir):
    image = cv2.imread(imageDir+imageFile)
    if image is None:
        continue
    image_org = copy.deepcopy(image)

    rects = detect(image,faceCascade)
    
    startX, startY, side = drawRect(rects,image)
    
    cropped_image = image_org[startY:startY+side, startX:startX+side]
   
    cv2.imwrite(croppedImageDir + imageFile,cropped_image)
    
    print('Saved:'+imageFile)
    #uncommed to show cropped image
    #cv2.imshow(imageFile,cropped_image)