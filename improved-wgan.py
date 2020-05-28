from glob import glob
import os
from random import choice
import numpy as np
import cv2
import keras
from keras import layers
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge,concatenate, Conv2D
from keras.layers import Reshape
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.core import Flatten
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, LeakyReLU
from keras.optimizers import SGD, Adagrad,RMSprop
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import *
from keras.optimizers import Adam,Adagrad
import keras.backend as K
from functools import partial
from keras.layers.merge import _Merge
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

def read_tactile(file):
    data_list = []
    label_list = []
    classes_name = os.listdir(file)
    classes_name = np.array(classes_name)
    classes_name = np.sort(classes_name)
    class_dict = {label:index for index,label in enumerate(classes_name)} #生成字典,名字：类别
    for class_name in classes_name:
        imgs_path = os.path.join(file,class_name)
        imgs_name = glob(os.path.join(imgs_path, "*/*/*.jpg"))
        data_list += imgs_name
        label_list += [class_dict[class_name] for i in range (len(imgs_name))]

    np.random.seed(12)
    np.random.shuffle(data_list)
    np.random.seed(12)
    np.random.shuffle(label_list)
    return data_list, label_list

def read_visual(file):
    data_list = []
    label_list = []
    classes_name = os.listdir(file)
    classes_name = np.array(classes_name)
    classes_name = np.sort(classes_name)
    class_dict = {label:index for index,label in enumerate(classes_name)} #生成字典,名字：类别
    for class_name in classes_name:
        imgs_path = os.path.join(file,class_name)
        imgs_name = glob(os.path.join(imgs_path, "*.JPG"))
        data_list += imgs_name
        label_list += [class_dict[class_name] for i in range (len(imgs_name))]

    np.random.seed(12)
    np.random.shuffle(data_list)
    np.random.seed(12)
    np.random.shuffle(label_list)
    return data_list, label_list


def get_Tdict(file):
    dict = {} #生成字典,名字：类别
    data, label = read_tactile(file)
    for i in range(len(data)):
        dict[data[i]] = label[i]
    return dict 

def get_Vdict(file):
    dict = {} #生成字典,名字：类别
    data, label = read_visual(file)
    for i in range(len(data)):
        dict[data[i]] = label[i]
    return dict 

def get_paired (visual, tactile):
    V_dict = get_Vdict(visual)
    T_dict = get_Tdict(tactile)
    data = []
    label = []
    for key, value in T_dict.items():
        V_key = [k for k,v in V_dict.items() if v == value]
        data.append(key)
        np.random.seed(12)
        label.append(choice(V_key))
    return data, label

def read_image(reading_path):
    imgs = []
    
    for single_path in reading_path:
        img = cv2.imread(single_path)
        img = cv2.resize(img,(256,256))
        img = np.array(img)
        # img = img/255
        img = (img.astype(np.float32) - 127.5) / 127.5
        imgs.append(img)
        
    imgs = np.array(imgs)
    imgs = imgs.reshape(-1,256,256,3)
    return imgs

def upsample(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def build_discriminator_model():
    input_real = keras.Input(shape=(Theight, Twidth, Tchannels))
    input_condition = keras.Input(shape=(Vheight, Vwidth, Vchannels))


    x = concatenate([input_real, input_condition],axis=-1)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters=256, kernel_size=5, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    # x = Activation('sigmoid')(x)

    # model_input is conditioned by labels
    discriminator = keras.models.Model([input_real, input_condition], x, name='discriminator')
    discriminator.summary()

    return discriminator






def build_generator():
    
    # imgs: input: 256x256xch
    # U-Net structure, must change to relu
    
    input_noise = Input(shape=(latent_dim,))
    input_condition = Input(shape=(Vheight, Vwidth, Vchannels))
    noise = Dense(256*256)(input_noise)
    noise = Reshape((256,256,1))(noise)
    inputs = concatenate([input_condition,noise],axis=-1)

    c1 = layers.Conv2D(16, (3, 3), padding='same') (inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.Conv2D(16, (3, 3), padding='same') (c1)

    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)

    # p1 = layers.MaxPooling2D((2, 2)) (c1)
    p1 = layers.Conv2D(16, (3, 3), strides = 2, padding='same') (c1)
    p1 = layers.BatchNormalization()(p1)
    p1 = layers.ReLU()(p1)



    c2 = layers.Conv2D(32, (3, 3), padding='same') (p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)

    c2 = layers.Conv2D(32, (3, 3), padding='same') (c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)

    # p2 = layers.MaxPooling2D((2, 2)) (c2)
    p2 = layers.Conv2D(32, (3, 3), strides = 2, padding='same') (c2)
    p2 = layers.BatchNormalization()(p2)
    p2 = layers.ReLU()(p2)

    c3 = layers.Conv2D(64, (3, 3), padding='same') (p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)

    c3 = layers.Conv2D(64, (3, 3), padding='same') (c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)

    # p3 = layers.MaxPooling2D((2, 2)) (c3)
    p3 = layers.Conv2D(64, (3, 3), strides = 2, padding='same') (c3)
    p3 = layers.BatchNormalization()(p3)
    p3 = layers.ReLU()(p3)


    c4 = layers.Conv2D(128, (3, 3), padding='same') (p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)
    c4 = layers.Conv2D(128, (3, 3), padding='same') (c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)
    # p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = layers.Conv2D(128, (3, 3), strides = 2, padding='same') (c4)
    p4 = layers.BatchNormalization()(p4)
    p4 = layers.ReLU()(p4)



    c5 = layers.Conv2D(256, (3, 3), padding='same') (p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.ReLU()(c5)
    c5 = layers.Conv2D(256, (3, 3), padding='same') (c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.ReLU()(c5)


    u6 = upsample(128, (2, 2), strides=(2, 2), padding='same') (c5)

    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), padding='same') (u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.ReLU()(c6)
    c6 = layers.Conv2D(128, (3, 3), padding='same') (c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.ReLU()(c6)

    u7 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), padding='same') (u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.ReLU()(c7)
    c7 = layers.Conv2D(64, (3, 3), padding='same') (c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.ReLU()(c7)


    u8 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), padding='same') (u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.ReLU()(c8)
    c8 = layers.Conv2D(32, (3, 3), padding='same') (c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.ReLU()(c8)



    u9 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), padding='same') (u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.ReLU()(c9)
    c9 = layers.Conv2D(16, (3, 3), padding='same') (c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.ReLU()(c9)

    d = layers.Conv2D(3, (1, 1), activation='tanh') (c9)

    model = keras.models.Model([input_noise, input_condition], d,name='generator')
    model.summary()

    return model

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((16, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])





def train(visual_dir, tactile_dir, epochs = 10, batch_size=20):
    #build conditional GAN
    optimizer_s = RMSprop(lr=0.00005)


    generator = build_generator()
    critic = build_discriminator_model()

    #-------------------------------
    # Construct Computational Graph
    #       for the Critic
    #-------------------------------

    # Freeze generator's layers while training critic
    generator.trainable = False
    # Image input (real sample)
    real_img = layers.Input(shape=(Vheight, Vwidth, Vchannels))
    noise = layers.Input(shape=(latent_dim,))
    condition = layers.Input(shape=(Vheight, Vwidth, Vchannels))
    # Generate image based of noise (fake sample)
    fake_img = generator([noise,condition])

    # Discriminator determines validity of the real and fake images
    fake = critic([fake_img,condition])
    valid =critic([real_img,condition])

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage()([real_img,fake_img])
    # Determine validity of weighted sample
    validity_interpolated = critic([interpolated_img,condition])

    # Use Python partial to provide loss function with additional
    # 'averaged_samples' argument
    partial_gp_loss = partial(gradient_penalty_loss,
                      averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

    critic_model = Model(inputs=[real_img, noise, condition],outputs=[valid, fake, validity_interpolated])
    critic_model.compile(loss=[wasserstein_loss,
                                        wasserstein_loss,
                                        partial_gp_loss],
                        optimizer=optimizer_s,
                        loss_weights=[1, 1, 10])
    critic_model.summary()

       #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

    # For the generator we freeze the critic's layers
    critic.trainable = False
    generator.trainable = True

    # Sampled noise for input to generator
    noise = layers.Input(shape=(latent_dim,))
    condition = layers.Input(shape=(Vheight, Vwidth, Vchannels))
    # Generate images based of noise
    img = generator([noise,condition])
    # Discriminator determines validity
    valid = critic([img,condition])
    # Defines generator model
    
    generator_model = Model(input=[noise, condition], output=valid)
    generator_model.compile(loss = wasserstein_loss, optimizer=optimizer_s)
    generator_model.summary()




    real_images_dir, condition_dir = get_paired(visual_dir,tactile_dir)

    step_epoch = len(real_images_dir)//batch_size

    epoch_num = 0
    iter_num = 0
    while epoch_num < epochs:
        # d_loss = 0
        # d_acc = 0
        
        
        loss_history_D  = 0
        acc_history_D = 0
        loss_history_G = 0
        acc_history_G = 0
        
        while iter_num < step_epoch:
            train_real = np.array(read_image(real_images_dir[batch_size*iter_num:batch_size*(iter_num+1)]))
            train_condition = np.array(read_image(condition_dir[batch_size*iter_num:batch_size*(iter_num+1)]))

            random_latent_vectors_D = np.array(np.random.normal(0,1,size=(batch_size, latent_dim)))
            # print(random_latent_vectors.shape)
            #数据加载
            
            
            valid = -np.ones((batch_size, 1))
            fake =  np.ones((batch_size, 1))
            dummy = np.zeros((batch_size, 1)) 
            # real += 0.05 * np.random.random(real.shape)
            # fake += 0.05 * np.random.random(fake.shape)

            d_loss = critic_model.train_on_batch([train_real,random_latent_vectors_D,train_condition],
                                                        [valid, fake, dummy])
            loss_history_D+=d_loss[0]

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = generator_model.train_on_batch([random_latent_vectors_D,train_condition], valid)
            
            loss_history_G+= g_loss
    
            
            print ("%d [D loss: %f] [G loss: %f]" % (iter_num, d_loss[0], g_loss))

            iter_num += 1
            if iter_num%10==0:
                # print("step:%d [D loss: %f, acc.: %f] [G loss: %f, fool_D: %f]" % (iter_num, loss_history_D, acc_history_D, loss_history_G, acc_history_G))

                        # 保存生成的图像
                generated_images = generator.predict([random_latent_vectors_D, train_condition])
                img = generated_images[0] * 127.5+127.5
                # print(img.shape)
                cv2.imwrite(os.path.join(save_dir, 'generated' + 'epoch'+str(epoch_num)+'iter'+ str(iter_num) + '.png'),img)
         
                # 保存真实图像，以便进行比较
                img = train_condition[0] * 127.5+127.5
                cv2.imwrite(os.path.join(save_dir, 'condition' + 'epoch'+ str(epoch_num)+ 'iter'+ str(iter_num) + '.png'),img)

        loss_history_D = loss_history_D/(step_epoch)
        # acc_history_D = acc_history_D/step_epoch
        loss_history_G = loss_history_G/step_epoch
        # acc_history_G = acc_history_G/step_epoch    # print(acc_real,acc_fake)
        print('------------------------------------epoch--------------------------------------')
        print("epoch: %d [D loss: %f] [G loss: %f]" % (epoch_num, loss_history_D,  loss_history_G))
        print('------------------------------------epoch--------------------------------------')
        epoch_num += 1
        iter_num = 0 # reset counter
    # 开始训练迭代

 
Vheight =256
Vwidth = 256
Vchannels = 3
Theight =256
Twidth = 256
Tchannels = 3
latent_dim = 256
visual_path = '/home/guan/Desktop/cgan/data/visual'
tactile_path = '/home/guan/Desktop/cgan/data/tactile'
save_dir = '/home/guan/Desktop/cgan/generated1'
noise_dir = '/home/guan/Desktop/cgan/data/noise/1293.jpg'
# a = build_discriminator_model()

train(visual_path,tactile_path,epochs = 25, batch_size= 16)