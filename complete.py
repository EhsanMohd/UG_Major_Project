import os
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
seed = 1234
random.seed = seed
np.random.seed = seed
tf.seed = seed

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        
    def __load__(self, id_name):
        image_path = os.path.join(self.path, id_name, "images", id_name) + ".png"
        mask_path = os.path.join(self.path, id_name, "masks/")
        all_masks = os.listdir(mask_path)
        
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        mask = np.zeros((self.image_size, self.image_size, 1))
        for name in all_masks:
            _mask_path = mask_path + name
            _mask_image = cv2.imread(_mask_path, -1)
            _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size)) 
            _mask_image = np.expand_dims(_mask_image, axis=-1)
            mask = np.maximum(mask, _mask_image)
            
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

image_size = 128
train_path = "dataset/stage1_train/"
epochs = 5
batch_size = 8

train_ids = next(os.walk(train_path))[1]

val_data_size = 10
test_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]
gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
#x, y = gen.__getitem__(0)
#print(x.shape, y.shape)

def down_sampling(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_sampling(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def end_layer(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_sampling(p0, f[0]) #128 -> 64
    c2, p2 = down_sampling(p1, f[1]) #64 -> 32
    c3, p3 = down_sampling(p2, f[2]) #32 -> 16
    c4, p4 = down_sampling(p3, f[3]) #16->8
    
    bn = end_layer(p4, f[4])
    
    up_1 = up_sampling(bn, c4, f[3]) #8 -> 16
    up_2 = up_sampling(up_1, c3, f[2]) #16 -> 32
    up_3 = up_sampling(up_2, c2, f[1]) #32 -> 64
    up_4 = up_sampling(up_3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(up_4)
    #predicted segmentation image 
    model = keras.models.Model(inputs, outputs)
    return model

'''
def acloss(y_true,y_pred):
    del_r = y_pred[:,1:,:]-y_pred[:,:-1,:]
    del_c = y_pred[:,:,1:]-y_pred[:,:,:-1]
    del_r = del_r**2
    del_c = del_c**2
    del_pred = abs(del_r+del_c)
    eps=1e-4
    length = np.mean(sqrt(del_pred+eps))
    c_in = np.ones_like(y_pred)
    c_out = np.zeros_like(y_pred)
    region_in = np.mean(y_pred*(y_true-c_in)**2)
    region_out = np.mean((1-y_pred)*(y_true-c_out)**2)
    region = region_in + region_out
    loss = 10*length + region 
    return loss
'''
model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
#model.summary()

train_dataset = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
test_gen = DataGen(test_ids, train_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
test_steps = len(test_ids)//batch_size

model.fit_generator(train_dataset, validation_data=test_gen, steps_per_epoch=train_steps, validation_steps=test_steps, epochs=epochs)

## Save the Weights
model.save_weights("UNetW.h5")

x, y = test_gen.__getitem__(1)
result = model.predict(x)

