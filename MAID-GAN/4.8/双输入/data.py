
import os
import glob
import h5py
import numpy as np
import tensorflow as tf
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'





def read_data(path):
    with h5py.File(path, 'r') as hf:
        print("正在读取数据...")
        data = np.array(hf.get('data'))
    return data


#图片拼接
def merge(images, size):
    '''
    images is the picture maxtrix, which is reconstructed.\
    size is the [nx,ny] that is the weight and length of dispart time.
    '''

    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1],1))
    #print(size[0], size[1])
    for idx, image in enumerate(images):    #enumerate函数可以同时获得索引和值
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w,:] = image
    return img


# if __name__ == "__main__":
#     data_test()
#     data_train()
#     data_validation()

