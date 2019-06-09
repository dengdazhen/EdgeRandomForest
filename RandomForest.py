import numpy as np
import cv2
from utils import downsample,downsample_keepdim,bartlett_blurring
from data import BSDS500
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

def BGR2CIELUV(bgr_img):
    var_bgr = bgr_img / 255.0
    transform_mat = np.array([[0.180423, 0.072169, 0.950227], 
                              [0.357580, 0.715160, 0.119193], 
                              [0.412453, 0.212671, 0.019334]])
    xyz = np.dot(var_bgr,transform_mat)
    
    L = np.where(xyz[:,:,1] > 0.008856,116*(xyz[:,:,1])**(1/3)-16,903.3*(xyz[:,:,1]))
    # L = 116*np.power(xyz[:,:,1],1/3)-16
    # print("L",L)
    den = xyz[:,:,0] + 15.0 * xyz[:,:,1] + 3.0 * xyz[:,:,2]
    den = np.where(den < 1e-7,1e-7,den)
    var_u = 4.0* xyz[:,:,0]/den
    # print('var_u',var_u)
    var_v = 9.0* xyz[:,:,1]/den
    # print('var_v',var_v)

    var_u = 13* L*(var_u - 0.19793943)
    var_v = 13* L*(var_v - 0.46831096)
    # Convert to uint8
    L = 255*L/100.0
    var_u = 255/354.0*(var_u+134)
    var_v = 255/262.0*(var_v+140)
    LUV = np.concatenate((L[:,:,np.newaxis],var_u[:,:,np.newaxis],var_v[:,:,np.newaxis]),axis = 2)
    # print(LUV[0][0][0])
    return LUV.astype(np.uint8)

def gradient_magnitude(img_gray):
    x_right_shift = np.roll(img_gray,1,axis = 1)
    x_left_shift = np.roll(img_gray,-1,axis = 1)
    y_up_shift = np.roll(img_gray, -1,axis = 0)
    y_down_shift = np.roll(img_gray,1,axis = 0)
    dx = x_right_shift - x_left_shift
    dy = y_down_shift - y_up_shift
    # derivative = np.where(np.abs(x_right_shift - x_left_shift) > 0.00001,(y_down_shift - y_up_shift)/(x_right_shift - x_left_shift),0)
    # zero_divide = 100000 * np.ones_like(dy)
    # derivative = np.divide(dy,dx, out=zero_divide, where=((dx) != 0))
    # return np.arctan(derivative)
    c_1 = (dy >  0) & (dx >  0)
    c_2 = (dy >  0) & (dx <= 0)
    c_3 = (dy <= 0) & (dx >  0)
    c_4 = (dy <= 0) & (dx <= 0)

    gra_mag = np.sqrt(dx**2 + dy**2)

    return gra_mag,c_1*1.0,c_2*1.0,c_3*1.0,c_4*1.0

def channels(img):
    # 32 * 32 * c
    img_luv = BGR2CIELUV(img)
    l,u,v = cv2.split(img_luv/255)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)/255.0
    ds_img_gray = downsample_keepdim(img_gray)
    mag,c1,c2,c3,c4 = gradient_magnitude(img_gray)
    ds_mag,ds_c1,ds_c2,ds_c3,ds_c4 = gradient_magnitude(ds_img_gray)
    merged = cv2.merge((l,u,v,mag,c1,c2,c3,c4,ds_mag,ds_c1,ds_c2,ds_c3,ds_c4))
    ds_merged = downsample(merged,2)
    # print(ds_merged.shape)
    blur_ds = bartlett_blurring(ds_merged,8)
    blur_ds = downsample(blur_ds,16/5).reshape((25,-1))
    # print(blur_ds.shape)
    additional_length = blur_ds.shape[0] * (blur_ds.shape[0] - 1) / 2
    channel_length = ds_merged.shape[0]*ds_merged.shape[1]*ds_merged.shape[2]
    newfeature = np.zeros(int(additional_length * blur_ds.shape[1] + channel_length))
    newfeature[:int( ds_merged.shape[0]*ds_merged.shape[1]*ds_merged.shape[2])] = ds_merged.flatten()
    for i in range(blur_ds.shape[1]):
        newfeature[int(i*additional_length + channel_length):int((i + 1)*additional_length + channel_length)] = pdist(blur_ds[:,i][:,np.newaxis])
    return newfeature



def split_continous(x,y,y_r,idx,value):
    L = []
    Ly = []
    Ll = []
    R = []
    Ry = []
    Rl = []
    for i in range(x.shape[0]):
        if x[i][idx] < value:
            L.append(x[i])
            Ly.append(y[i])
            Ll.append(y_r[i])
        else:
            R.append(x[i])
            Ry.append(y[i])
            Rl.append(y_r[i])
    return len(L)/x.shape[1],len(R)/x.shape[1],L,R,Ly,Ry,Ll,Rl

def split_binary(x,y,labels,idx):
    # 0 or 1
    L = []
    Ly = []
    Ll = []
    R = []
    Ry = []
    Rl = []
    for i in range(x.shape[0]):
        if x[i][idx] == 0:
            L.append(x[i])
            Ly.append(y[i])
            Ll.append()
        else:
            R.append(x[i])
            Ry.append(y[i])
    return len(L)/x.shape[1],len(R)/x.shape[1],L,R,Ly,Ry,Ll,Rl

def Shannon_Entropy(y):
    label_set = {}
    count = 0
    for i in y:
        count += 1
        if i in label_set.keys():
            label_set[i] += 1
        else:
            label_set[i] = 0
    ent = 0
    for i in label_set.keys():
        prob = label_set[i]/count
        if prob > 1e-7:
            ent -= prob * np.log2(prob)
    return ent


def entropy(x,idx,y,y_r,continous = True):
    attrs = list(set(x[:,idx]))
    sorted_attrs = sorted(attrs)
    # print(sorted_attrs)
    # if len(attrs) > 2 or (len(attrs) == 2 and (0 not in sorted_attrs or 1 not in sorted_attrs)):
    #     continous = True
    minE = np.inf
    for i in range(len(sorted_attrs) - 1):
        partition_v = 0.5*(sorted_attrs[i] + sorted_attrs[i + 1])
        pl,pr,L,R,Ly,Ry,Ll,Rl = split_continous(x,y,y_r,idx,partition_v)
        E = pl*Shannon_Entropy(Ly) + pr*Shannon_Entropy(Ry)
        if E < minE:
            minE = E
            best_attrval = partition_v
    return minE,partition_v
    # else:
    #     continous = False
    #     minE = np.inf
    #     label_set = {}
    #     E = 0
    #     for i in attrs:
    #         ps , _ , ys = split_discrete(x,y,idx,i)
    #         E -= -ps * np.log2(ps)
    #     return E,None ,continous      

class Node(object):
    def __init__(self,min_sample = 8,max_depth = 64,m = 256, k = 2, pca_reduced_dim = 5):
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.m = m
        self.k = k
        self.pca_reduced_dim = pca_reduced_dim

    def build_data(self,x,labels,labels_edg,depth):
        '''
        x: n_count * num_features
        labels: n_count * m 
        m = 256 
        the k must be 2, 4, 8, ...
        '''
        print("depth:",depth)
        self.x = x
        self.index_1 = np.random.permutation(self.m)
        self.index_2 = np.random.permutation(self.m)
        self.selected_features = (labels[:,self.index_1] == labels[:,self.index_2])*1.0
        pca = PCA(n_components=self.pca_reduced_dim)
        self.pca_num = int(np.log2(self.k))
        pca_features = pca.fit_transform(self.selected_features)
        distMatrix = pdist(pca_features)
        self.mediod = labels_edg[np.argmin(distMatrix.sum(axis=0))]
        if depth == self.max_depth:
            self.siblings = None
            print("reached max depth")
        else:
            self.reduced_features = pca_features[:,:self.pca_num]
            self.final_labels = np.zeros([labels.shape[0]])
            binary_features = np.where(self.reduced_features > 0, 1, 0)
            # print(binary_features)
            # print(2**np.arange(0,self.k -1))
            self.final_labels = np.sum(binary_features * (2**np.arange(0,self.k - 1)),axis = 1)
            if np.all(self.final_labels == 1) or np.all(self.final_labels == 0):
                self.siblings = None
                print("all samples are with same final label")
            # self.split_continous(0,0.5)
            else:
                best_gain = 0
                best_feature = -1
                best_val = None
                base_ent = Shannon_Entropy(self.final_labels)
                for i in range(x.shape[1]):
                    # print('feature:',i)
                    newEntropy,best_val_tmp = self.entropy(i)
                    info_gain = base_ent - newEntropy
                    if info_gain > best_gain:
                        best_gain = base_ent - newEntropy
                        best_feature = i
                        best_val = best_val_tmp
                        print('best_feature update!')
                if best_feature == -1:
                    self.siblings = None
                else:
                    L_i,R_i = self.split_continous(best_feature,best_val)
                    l_num = np.sum(L_i*1.0)
                    r_num = np.sum(R_i*1.0)
                    if l_num >= self.min_sample and r_num >= self.min_sample:
                        self.best_feature = best_feature
                        self.best_val = best_val
                        print("split:{} and {}".format(np.sum(L_i*1.0),np.sum(R_i*1.0)))
                        Node_Left = Node(self.min_sample,self.max_depth,self.m, self.k, self.pca_reduced_dim)
                        Node_Left.build_data(self.x[L_i],labels[L_i],labels_edg[L_i],depth + 1)
                        Node_Right = Node(self.min_sample,self.max_depth,self.m, self.k, self.pca_reduced_dim)
                        Node_Right.build_data(self.x[R_i],labels[R_i],labels_edg[R_i],depth + 1)
                        self.siblings = [Node_Left,Node_Right]
                    else:
                        self.siblings = None
                        print("the sample number is too small")


    def predict(self,x):
        if type(self.siblings) == type(None):
            return self.mediod
        elif x[self.best_feature] < self.best_val:
            return self.siblings[0].predict(x)
        else:
            return self.siblings[1].predict(x)

    def split_continous(self,idx,value):
        L_i = (self.x[:,idx] < value)
        R_i = ~L_i
        # print(L_i)
        # print(R_i)
        return L_i,R_i

    def entropy(self,idx,sample_num = 10):
        attrs = list(set(self.x[:,idx]))
        sorted_attrs = sorted(attrs)
        partition_v = None
        if len(sorted_attrs) > sample_num:
            scale_rate = len(sorted_attrs)/10
            new_attrs = []
            for i in range(sample_num):
                new_attrs.append(sorted_attrs[int(scale_rate * i)])
            sorted_attrs = new_attrs
        # print(sorted_attrs)
        # if len(attrs) > 2 or (len(attrs) == 2 and (0 not in sorted_attrs or 1 not in sorted_attrs)):
        #     continous = True
        minE = np.inf
        for i in range(len(sorted_attrs) - 1):
            partition_v = 0.5*(sorted_attrs[i] + sorted_attrs[i + 1])
            L_i,R_i = self.split_continous(idx,partition_v)
            pl = np.sum(L_i*1.0)/L_i.shape[0]
            E = pl*Shannon_Entropy(self.final_labels[L_i]) + (1 - pl)*Shannon_Entropy(self.final_labels[R_i])
            if E < minE:
                minE = E
                best_attrval = partition_v
        return minE,partition_v

def recursive_write(curNode,w):
    variable = 'mediod'
    w.write(variable)
    for number_array in curNode.mediod:
        for number in number_array:
            num_str = ' {}'.format(number)
            w.write(num_str)
    w.write('\n')
    if type(curNode.siblings) != type(None):
        variable = 'feature_index {}\n'.format(curNode.best_feature)
        w.write(variable)
        variable = 'feature_value {}\n'.format(curNode.best_val)
        w.write(variable)
        recursive_write(curNode.siblings[0],w)
        recursive_write(curNode.siblings[1],w)
    else:
        variable = 'End\n'
        w.write(variable)

def recursive_read(curNode,w):
    line = w.readline().split()
    if line[0] == 'mediod':
        # print(len(line))
        # print(line[1:-1])
        curNode.mediod = np.array(line[1:1025],dtype = int).reshape([32,32])
    line = w.readline().split()
    if line[0] == 'feature_index':
        curNode.best_feature = int(line[1])
    elif line[0] == 'End':
        curNode.siblings = None
        return
    line = w.readline().split()
    if line[0] == 'feature_value':
        curNode.best_val = float(line[1])
    Node_Left = Node(curNode.min_sample,curNode.max_depth,curNode.m,curNode.k,curNode.pca_reduced_dim)
    recursive_read(Node_Left,w)
    Node_Right = Node(curNode.min_sample,curNode.max_depth,curNode.m,curNode.k,curNode.pca_reduced_dim)
    recursive_read(Node_Right,w)
    curNode.siblings = [Node_Left,Node_Right]

    


class DecisionTree(object):
    def __init__(self,min_sample = 8,max_depth = 64,m = 256, k = 2, pca_reduced_dim = 5):
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.m = m
        self.k = k
        self.pca_reduced_dim = pca_reduced_dim
        self.Root = Node(self.min_sample,self.max_depth,self.m,self.k,self.pca_reduced_dim)

    def build_data(self,x,labels,labels_edg):
        self.Root.build_data(x,labels,labels_edg,1)

    def predict(self,img,bench_mark = None,visualization = False):
        # print("predicting!")
        channel_img = channels(img)
        predict_edg = self.Root.predict(channel_img)
        if visualization:
            cv2.imshow("original img",img)
            cv2.imshow('predicted',predict_edg*1.0)
            if type(bench_mark) != type(None):
                cv2.imshow('benchmark',bench_mark.astype(np.uint))
            cv2.waitKey()
        return predict_edg

class RandomForest(object):
    def __init__(self,T = 8,stride = 2,max_depth = 64,min_sample = 8,m = 256, k = 2, pca_reduced_dim = 5):
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.m = m
        self.k = k
        self.pca_reduced_dim = pca_reduced_dim
        self.stride = stride
        self.T = T
        self.Tree_group1 = []
        self.Tree_group2 = []
        

    def build_data(self,data,num = 5):
        for i in range(self.T):
            if i < 0.5*self.T:
                img,seg,edg = data.crop(num=num,set_num = 0)
                img_1,seg_1,edg_1 = data_preprocess(img,seg,edg)
                dt = DecisionTree(self.min_sample,self.max_depth,self.m,self.k,self.pca_reduced_dim)
                dt.build_data(img_1,seg_1,edg_1)
                self.Tree_group1.append(dt)
            else:
                img,seg,edg = data.crop(num=num,set_num = 0)
                img_1,seg_1,edg_1 = data_preprocess(img,seg,edg)
                dt = DecisionTree(self.min_sample,self.max_depth,self.m,self.k,self.pca_reduced_dim)
                dt.build_data(img_1,seg_1,edg_1)
                self.Tree_group2.append(dt)
        self.model_save()

    def predict(self,img,visualization = False,filename = None):
        predicted_img = np.zeros(img.shape[:2])
        weight_img = np.zeros(img.shape[:2])
        turn = True
        for i in range(0,img.shape[0] - 32,self.stride):
            for j in range(0,img.shape[1] - 32,self.stride):
                print(i,j)
                if turn:
                    for tree in self.Tree_group1:
                        img_patch = img[i:i+32,j:j+32,:]
                        predicted_img[i:i+32,j:j+32] += tree.predict(img_patch)
                        weight_img[i:i+32,j:j+32] += 1
                    turn = not turn
                else:
                    for tree in self.Tree_group2:
                        img_patch = img[i:i+32,j:j+32,:]
                        predicted_img[i:i+32,j:j+32] += tree.predict(img_patch)
                        weight_img[i:i+32,j:j+32] += 1
                    turn = not turn
        predicted_img /= np.max(predicted_img)
        if visualization:
            cv2.imshow("result",predicted_img)
            cv2.imshow("benchmark",img)
            cv2.waitKey()
        if type(filename) != type(None):
            cv2.imwrite(filename,predicted_img*255)
        return predicted_img
    
    def model_save(self,path = 'model.txt'):
        with open(path, 'w+') as w:
            t_variable = 'T {}\n'.format(self.T)
            w.write(t_variable)
            s_variable = 'Stride {}\n'.format(self.stride)
            w.write(s_variable)
            for Tree in self.Tree_group1:
                recursive_write(Tree.Root,w)
            for Tree in self.Tree_group2:
                recursive_write(Tree.Root,w)
                # variable = 'mediod'
                # w.write(variable)
                # for number_array in Tree.Root.mediod:
                #     for number in number_array:
                #         num_str = ' {}'.format(number)
                #         w.write(num_str)
                # w.write('\n')
                # variable = 'feature_index {}\n'.format(Tree.Root.best_feature)
                # w.write(variable)
                # variable = 'feature_value {}\n'.format(Tree.Root.best_val)
                # w.write(variable)

    def build_from_file(self,path):
        with open(path,'r') as w:
            line = w.readline().split()
            if line[0] == 'T':
                self.T = int(line[1])
            line = w.readline().split()
            if line[0] == 'Stride':
                self.stride = int(line[1])
            for i in range(int(0.5*self.T)):
                dt = DecisionTree(self.min_sample,self.max_depth,self.m,self.k,self.pca_reduced_dim)
                recursive_read(dt.Root,w)
                self.Tree_group1.append(dt)
            for i in range(int(0.5*self.T)):
                dt = DecisionTree(self.min_sample,self.max_depth,self.m,self.k,self.pca_reduced_dim)
                recursive_read(dt.Root,w)
                self.Tree_group2.append(dt)

def data_preprocess(img,img_seg,img_edg):
    print('preprocessing')
    total_len = len(img)
    img_p = np.zeros([total_len,7228])
    img_seg_p = np.zeros([total_len,256],dtype = img_seg[0].dtype)
    img_edg_p = np.zeros([total_len,32,32],dtype = img_edg[0].dtype)
    for i in range(total_len):
        img_p[i] = channels(img[i])
        img_seg_p[i] = img_seg[i].flatten()
        img_edg_p[i] = img_edg[i]
    print('finished')
    return img_p,img_seg_p,img_edg_p



if __name__ == "__main__":
    data = BSDS500('/Users/dazhen/Data/BSR/BSDS500/data/')
    rf = RandomForest()
    # rf.build_data(data,50)
    rf.build_from_file('model_10000.txt')

    rf.stride = 4
    for i in range(50):
        print(i)
        img = data.test_img[i]
        rf.predict(img,filename='results/{:05d}.png'.format(i))
    


