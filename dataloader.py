import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict
from tensorflow.keras.layers.experimental import preprocessing

from utiliz import normalization

class Cifar10DataLoader:
    def __init__(self, dataloader_args):
        self.dataloader_args = dataloader_args
        self.info = edict({'train_size':50000,'test_size':10000,'image_size':[32,32,3],
                           'epochs': dataloader_args['epochs']})

    def _parse_info(self):
        source_info = edict()
        source_info.train_size = int(self.info['train_size']/2)
        source_info.test_size = int(self.info['test_size']/2)
        source_info.batch_size = self.dataloader_args['batch_size']
        source_info.train_step = int(source_info.train_size/source_info.batch_size)
        source_info.test_step = int(source_info.test_size/source_info.batch_size)
        source_info.epochs = self.dataloader_args['epochs']
        
        target_info = edict()
        target_info.train_size = self.dataloader_args['batch_size']*100
        target_info.test_size = int(self.info['test_size']/2)
        target_info.batch_size = self.dataloader_args['batch_size']
        target_info.train_step = 1
        target_info.epochs = -1
        return source_info, target_info
    
    def seperate_task(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        
        x_train = (x_train / 255.0).astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        x_test = (x_test / 255.0).astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        x_a_train = []
        y_a_train = []
        x_a_test = []
        y_a_test = []

        x_b_train = []
        y_b_train = []
        x_b_test  = []
        y_b_test  = []

        for img,lab in zip(x_train,y_train):
            if lab[0]<5:
                x_a_train.append(img)
                y_a_train.append(lab)
            else:
                x_b_train.append(img)
                y_b_train.append(lab)
            
        for img,lab in zip(x_test, y_test):
            if lab[0]<5:
                x_a_test.append(img)
                y_a_test.append(lab)
            else:
                x_b_test.append(img)
                y_b_test.append(lab)

        x_a_train = np.asarray(x_a_train)
        y_a_train = np.asarray(y_a_train)
        x_a_test  = np.asarray(x_a_test)
        y_a_test  = np.asarray(y_a_test)

        x_b_train = np.asarray(x_b_train)
        y_b_train = np.asarray(y_b_train) - 5
        x_b_test  = np.asarray(x_b_test)
        y_b_test  = np.asarray(y_b_test) - 5
        
        source_task = {"x_train":x_a_train,"y_train":y_a_train,"x_test":x_a_test,"y_test":y_a_test}
        target_task = {"x_train":x_b_train,"y_train":y_b_train,"x_test":x_b_test,"y_test":y_b_test}
        return source_task, target_task
        
    
    def make_dataset(self, task, info, shuffle=True):
        
        x_train,x_test = task["x_train"],task["x_test"]
        y_train,y_test = task["y_train"],task["y_test"]
        
        # one-hot
        y_train = tf.keras.utils.to_categorical(y_train, 5)
        y_test = tf.keras.utils.to_categorical(y_test, 5)
        
        
        train_size = info.train_size
        test_size = info.test_size

        full_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_train, 'labels': y_train})
        if shuffle:
            full_dataset = full_dataset.shuffle(train_size)

        train_dataset = full_dataset.take(train_size)
        train_dataset = train_dataset.batch(info.batch_size)
        train_dataset = train_dataset.prefetch(1)
        train_dataset = train_dataset.repeat(info.epochs)

        test_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_test, 'labels': y_test})
        test_dataset = test_dataset.take(test_size)
        test_dataset = test_dataset.batch(info.batch_size, drop_remainder=True).repeat(-1)
        
        return train_dataset, test_dataset
        
    def load_dataset(self):

        source_task, target_task = self.seperate_task()
        self.source_info, self.target_info = self._parse_info()
        
        source_train, source_test = self.make_dataset(source_task, self.source_info)
        target_train, target_test = self.make_dataset(target_task, self.target_info, shuffle=False)

        return source_train, source_test, target_train, target_test