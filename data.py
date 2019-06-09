import scipy.io as sio
import cv2
import numpy as np
import os
from utils import downsample_direct

# mat_contents = sio.loadmat('/home/dazhen/data/BSR/BSDS500/data/groundTruth/train/2092.mat')
# print(mat_contents['groundTruth'][0][0][0][0][0].shape)
# print(mat_contents['groundTruth'][0][0][0][0][1].shape)
# two = mat_contents['groundTruth'][0][1][0][0][1] * 255.0
# # print(mat_contents['groundTruth'])

# img = cv2.imread('/home/dazhen/data/BSR/BSDS500/data/images/train/2092.jpg')
# print(img.shape)
# # cv2.imshow("show",two)
# # cv2.waitKey()

# for i in mat_contents['groundTruth'][0]:
#     print(np.max(i[0][0][0]))
#     pic = i[0][0][1] * 255.0
#     cv2.imshow("show",pic)
#     cv2.waitKey()
def matload(filename):
    mat_contents = sio.loadmat(filename)
    content_list = []
    for i in mat_contents['groundTruth'][0]:
        content_list.append(i[0][0])
    return content_list

def random_crop(img,mat,size = [32,32],num = 5):
    img_size = img.shape[:2]
    crop_masks_x = np.random.randint(0,img_size[0] - size[0],[num],int)
    crop_masks_y = np.random.randint(0,img_size[1] - size[1],[num],int)
    # img_crops = np.zeros([num,size[0],size[1],3],dtype = img.dtype)
    # mat_seg_crops = np.zeros([num,size[0],size[1]],dtype = mat[0].dtype)
    # mat_edg_crops = np.zeros([num,size[0],size[1]],dtype = mat[1].dtype)
    img_crops = []
    mat_seg_crops = []
    mat_edg_crops = []
    for i in range(num):
        img_crops.append(img[crop_masks_x[i]:(crop_masks_x[i]+size[0]),crop_masks_y[i]:crop_masks_y[i] + size[1],:])
        mat_seg_crops.append(downsample_direct(mat[0][crop_masks_x[i]:(crop_masks_x[i]+size[0]),crop_masks_y[i]:crop_masks_y[i] + size[1]]))
        mat_edg_crops.append(mat[1][crop_masks_x[i]:(crop_masks_x[i]+size[0]),crop_masks_y[i]:crop_masks_y[i] + size[1]])
    return img_crops,mat_seg_crops,mat_edg_crops

def data_read(img_path):
    print(img_path)
    mat_dir = img_path.replace('images','groundTruth')
    # print(mat_path)
    # print(len(os.listdir(mat_path)))
    filenames = [os.path.splitext(x)[0] for x in os.listdir(img_path)]
    # print('matdir',mat_dir)
    if 'Thumbs' in filenames:
        filenames.remove('Thumbs')
    imgs = []
    mats = []
    for i in filenames:
        img = os.path.join(img_path,i+'.jpg')
        imgs.append(cv2.imread(img))
        mats.append(matload(os.path.join(mat_dir,i+'.mat')))
    return imgs,mats

class BSDS500(object):
    def __init__(self,path = '/home/dazhen/data/BSR/BSDS500/data/'):
        self.path = os.path.dirname(path)
        self.train_img_path = os.path.join(self.path,'images','train')
        self.test_img_path = os.path.join(self.path,'images','test')
        self.val_img_path = os.path.join(self.path,'images','val')
        self.train_img,self.train_mat = data_read(self.train_img_path)
        self.test_img,self.test_mat = data_read(self.test_img_path)
        self.val_img,self.val_mat = data_read(self.val_img_path)

    def crop(self,num = 5,set_num = 0):
        if set_num == 0:
            img_set = self.train_img
            mat_set = self.train_mat
        elif set_num == 1:
            img_set = self.test_img
            mat_set = self.test_mat
        elif set_num == 2:
            img_set = self.val_img
            mat_set = self.val_mat
        crop_img = []
        crop_seg = []
        crop_edg = []
        for idx in range(len(img_set)):
            img,seg,edg = random_crop(img_set[idx],mat_set[idx][0],num = num)
            crop_img += img
            crop_seg += seg
            crop_edg += edg
        return crop_img,crop_seg,crop_edg


if __name__ == "__main__":
    # img_dir = os.path.dirname('/home/dazhen/data/BSR/BSDS500/data/images/train/')
    # data_read('/home/dazhen/data/BSR/BSDS500/data/images/train/')
    # filenames = os.listdir(img_dir)
    # print(len(filenames))
    # print(len(matload('/home/dazhen/data/BSR/BSDS500/data/groundTruth/train/2092.mat')))
    # a = BSDS500()
    a = BSDS500('/Users/dazhen/Data/BSR/BSDS500/data/')
    # input1 = a.train_img[0]
    # input2 = a.train_mat[0][0]
    # imgs,mats = random_crop(input1,input2)
    # print(imgs.shape)
    # print(mats[0].shape)
    # print(mats[1].shape)
    # for idx,i in enumerate(a.crop_img):
    #     cv2.imshow("check",i)
    #     cv2.imshow('seg',a.crop_edg[idx]*1.0)
    #     cv2.waitKey()