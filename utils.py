import cv2
import numpy as np

def downsample(img,scale = 2):
    if len(img.shape) == 3:
        shp = (int(img.shape[0]/scale),int(img.shape[1]/scale),img.shape[2])
    elif len(img.shape) == 2:
        shp = (int(img.shape[0]/scale),int(img.shape[1]/scale))
    else:
        raise TypeError
    ds_img = np.zeros(shp,dtype = img.dtype)
    for i in range(ds_img.shape[0]):
        for j in range(ds_img.shape[1]):
            end_i = min((int((i+1)*scale),img.shape[0]))
            end_j = min((int((j+1)*scale),img.shape[1]))
            ds_img[i][j] = img[int(scale*i)][int(scale*j)]
            if len(img.shape) == 3:
                ds_img[i][j] = np.average(img[int(i*scale):end_i,int(j*scale):end_j],axis = (0,1))[np.newaxis,np.newaxis,:]
            elif len(img.shape) == 2:
                ds_img[i][j] = np.average(img[int(i*scale):end_i,int(j*scale):end_j])
    return ds_img

def downsample_direct(img,scale = 2):
    if len(img.shape) == 3:
        shp = (int(img.shape[0]/scale),int(img.shape[1]/scale),img.shape[2])
    elif len(img.shape) == 2:
        shp = (int(img.shape[0]/scale),int(img.shape[1]/scale))
    else:
        raise TypeError
    ds_img = np.zeros(shp,dtype = img.dtype)
    for i in range(ds_img.shape[0]):
        for j in range(ds_img.shape[1]):
            ds_img[i][j] = img[int(i*scale),int(j*scale)]
    return ds_img

def upsample(img,scale = 2):
    shp = list(img.shape)
    shp[0] = int(shp[0]*scale)
    shp[1] = int(shp[1]*scale)
    us_img = np.zeros(shp,dtype = img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            end_i = min((int((i+1)*scale),us_img.shape[0]))
            end_j = min((int((j+1)*scale),us_img.shape[1]))
            if len(img.shape) == 3:
                ds_img[int(i*scale):end_i,int(j*scale):end_j] = np.average(img[int(i*scale):end_i,int(j*scale):end_j],axis = (0,1))[np.newaxis,np.newaxis,:]
            elif len(img.shape) == 2:
                ds_img[int(i*scale):end_i,int(j*scale):end_j] = np.average(img[int(i*scale):end_i,int(j*scale):end_j])

def downsample_keepdim(img,scale = 2):
    ds_img = np.zeros(img.shape,dtype = img.dtype)
    # for i in range(ds_img.shape[0]):
    #     for j in range(ds_img.shape[1]):
    #         ds_img[i][j] = img[scale*i][scale*j]
    for i in range(int(ds_img.shape[0]/scale)):
        for j in range(int(ds_img.shape[1]/scale)):
            end_i = min((int((i+1)*scale),ds_img.shape[0]))
            end_j = min((int((j+1)*scale),ds_img.shape[1]))
            if len(img.shape) == 3:
                ds_img[int(i*scale):end_i,int(j*scale):end_j] = np.average(img[int(i*scale):end_i,int(j*scale):end_j],axis = (0,1))[np.newaxis,np.newaxis,:]
            elif len(img.shape) == 2:
                ds_img[int(i*scale):end_i,int(j*scale):end_j] = np.average(img[int(i*scale):end_i,int(j*scale):end_j])
            
    return ds_img

# Bartlett Window
# https://en.wikipedia.org/wiki/Two_dimensional_window_design#Bartlett_Window
# The two dimensional mathematical representation of a Bartlett window is as shown below[9]
# The window is cone-shaped with its height equal to 1 and the base is a circle with its radius 2a. The vertical cross-section of this window is a 1-D triangle window.

def bartlett_window(a = 2):
    window_len = 2 * a - 1
    window = np.zeros([1,window_len])
    for i in range(window_len):
        window[0,i] = 1 - abs(i - a + 1)/a
    return window/a

def bartlett_blurring(img,a = 2):
    padding = a - 1
    shp = list(img.shape)
    shp[0] += padding * 2
    shp[1] += padding * 2
    img_padding = np.zeros(shp,img.dtype)
    img_blurring = np.zeros(img.shape,img.dtype)
    shp2 = list(img.shape)
    shp2[0] += padding * 2
    img_blurring_tmp = np.zeros(shp2)
    img_padding[padding:img.shape[0] + padding,padding:img.shape[1] + padding] = img
    bw = bartlett_window(a)
    # print(bw)
    w_len = bw.shape[1]
    for i in range(img.shape[1]):
        img_blurring_tmp[:,i,:] = np.sum(img_padding[:,i:i+w_len,:] * bw[:,:,np.newaxis],axis = 1)
    bw = bw.transpose()
    for j in range(img.shape[0]):
        img_blurring[j,:,:] = np.sum(img_blurring_tmp[j:j+w_len,:,:] * bw[:,:,np.newaxis],axis = 0)

    return img_blurring

if __name__ == "__main__":
    img = cv2.imread('apple.jpeg')
    print(img)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(img.shape)
    ds_img = downsample(img)
    print(ds_img.shape)
    crop = img[200:232,150:182,:]
    test = bartlett_blurring(crop,8)
    print(test.shape)
    ds_test = downsample(test,32/5)
    print(ds_test.shape)
    cv2.imshow("ch",ds_test)
    cv2.imshow("original",crop)
    cv2.waitKey()
    # a = bartlett_window(4)
    # print(a)
    # print(np.sum(a))
    # a = np.ones([5,5,3])
    # print(np.average(a,axis = (0,1)))