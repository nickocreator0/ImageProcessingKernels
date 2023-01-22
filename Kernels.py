#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Image Processing (Kernels)
# Author: Nicko.creator0
# January 2023


# In[1]:


import pandas as pd
import numpy as np
from decimal import Decimal


# In[2]:


def convolve(img, kernel):
    img_np = np.array(img)
    kernel_np = np.array(kernel)
    res = np.sum(np.dot(img_np, kernel_np))
    return res


# In[3]:


def median_smoothing(img):
    #img = [12,22,15,12,17,22,13,33,24]
    img.sort()
    mid_index = int(len(img)/2)
    one_pixel = img[mid_index]
    three_pixel = (img[mid_index] + img[mid_index+1] + img[mid_index-1])/3
    five_pixel = (img[mid_index] + img[mid_index+1] + img[mid_index-1] + img[mid_index-2] + img[mid_index+2])/5 
    print(f"one_pixel = {one_pixel}")
    print(f"three_pixel = {three_pixel}")
    print(f"five_pixel = {five_pixel}")


# In[4]:


def mean_smoothing(img, weight):
    sum_ = 0
    for item in range(len(img)):
        sum_ += img[item]
    noise = int(sum_/(weight+8))+1
    other_pixels = int((sum_-noise)/8)+1
    print(f" noise = {noise}")
    print(f"others = {other_pixels}")


# In[5]:


def gamma(pix, g):
    pix_new = round(255*((pix/255)**(1/g)))
    if pix_new > 255:
        pix_new = 255
    elif pix_new < 0:
        pix_new = 0
    print(f"gammad!= {pix_new}")


# In[6]:


def sobel(img, x_kernel, y_kernel):
    res_x = convolve(img, x_kernel)
    res_y = convolve(img, y_kernel)
    
    pix = (res_x**2 + res_y**2)**(1/2)
    
    #Truncation
    if pix > 255:
        pix = 255
    elif pix < 0:
        pix = 0
    print(f"sobeled= {pix}")


# In[7]:


def laplacian(img, kernel):
    pix = convolve(img, kernel)
    #Truncation
    if pix > 255:
        pix = 255
    elif pix < 0:
        pix = 0
    print(f"Laplacian= {pix}")
    


# In[8]:


def contrast(pixel, c):
    f = (259*(c+255))/(255*(255-c))
    pix = round(f*(pixel - 128)+128)
    #Truncation
    if pix > 255:
        pix = 255
    elif pix < 0:
        pix = 0
    print(f"contrasted: {pix}")


# In[9]:


def sharpen(img, kernel, given):
    id_k = [0,0,0,0,1,0,0,0,0]
    identity_kernel = np.array(id_k)
    k_np = np.array(kernel)
    if given == False:
        ker = identity_kernel - k_np 
    else:
        ker = k_np
    
    pix = convolve(img, ker)
    #Truncation
    if pix > 255:
        pix = 255
    elif pix < 0:
        pix = 0
    print(f"sharpened: {pix}")


# In[10]:


def cov_mat(x):
    x_ = np.array(x)
    x_bar = []
    for i in range(x_.shape[0]):
        row_sum = 0
        for j in range(x_.shape[1]):
            row_sum += x_[i][j]
        mean = round(row_sum/x_.shape[1])
        x_bar.append(mean)
    #print(x_bar)



    #Convert x_bar to a numpy array
    temp = np.array(x_bar)
    x_bar_np = np.transpose([temp] * 3)

    A_mat = np.array(x_ - x_bar_np)
    print("x bar:")
    print(x_bar_np)
    #print(x_bar_np)
    print("A:")
    print(A_mat)

    A_trans = np.transpose(A_mat)
    print("A T:")
    print(A_trans)
    cov_matrix = np.matmul(A_trans, A_mat)
    print(f"cov_matrix:")
    print(cov_matrix)
    #return A_mat


# In[11]:


def projection_space(face_space, A_mat):
    U = np.array(face_space)
    projection = np.matmul(np.transpose(U), A_mat)
    print(f"projection: {projection}")


# In[14]:


from scipy.signal import convolve2d

def bilinear(img, k, scale_factor):
    # Convert the input lists to numpy arrays
    image = np.array(img)
    kernel = np.array(k)
    # First get the Nearest Neighbor Interpolation
    upscaled = np.repeat(np.repeat(image, scale_factor, axis=0), scale_factor, axis=1)
    # Apply Bilinear
    new_image = convolve2d(upscaled, kernel)/4

    print(new_image[1:, 1:])


# In[21]:


#img = [22,1,24,12,27,24,30,35,23]


#median_smoothing(img)
######################################
#gamma(30,.5)
######################################
#x_k = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
#y_k = [1, 2, 1, 0, 0, 0, -1, -2, -1]
#sobel(img, x_k, y_k)
######################################
#kernel_lap = [0,-1,0,-1,4,-1,0,-1,0]
#laplacian(img, kernel_lap)
######################################
#contrast(120, 50) #(pixel, C)
######################################
#kernel_sharp = [0,-1,0,-1,5,-1,0,-1,0]
#sharpen(img, kernel_sharp, True) #given
######################################
"""
x = [[17,12,12],
     [12,14,6],
     [25,12,23],
     [30,11,67],
     [11,1,3],
     [12,6,68],
     [17,2,24],
     [19,-2,23],
     [15,9,45]]

cov_mat(x)
"""
######################################
"""
U = [[17,12,12],
     [12,14,6],
     [25,12,23]]
projection_space(U, cov_mat(x)) #uncomment return in cov_mat
"""

######################################
#img1 = [[1,3],[4,5]]
#k = [[1,1],[1,1]]
#bilinear(img1, k, 2)   #img, kernel, scale_factor

