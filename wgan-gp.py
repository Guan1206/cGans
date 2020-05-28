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
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD, Adagrad,RMSprop
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import *
from keras.optimizers import Adam,Adagrad
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import regularizers
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
        imgs_name = glob(os.path.join(imgs_path, "*.png"))
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
        imgs_name = glob(os.path.join(imgs_path, "*.png"))
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
        img = cv2.resize(img,(224,224))
        img = img/255
        imgs.append(img)
        
    imgs = np.array(imgs)
    imgs = imgs.reshape(-1,224,224,3)
    return imgs

def read_tactile_image(reading_path):
    imgs = []
    
    for single_path in reading_path:
        img = cv2.imread(single_path)
        img = np.array(img)
        img = img/255
        img = img[128:128+224, 208:208+224]
        imgs.append(img)
        
    imgs = np.array(imgs)
    imgs = imgs.reshape(-1,224,224,3)
    return imgs


def upsample(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

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
        alpha = K.random_uniform((8, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def build_discriminator_model():
    input_real = keras.Input(shape=(Theight, Twidth, Tchannels))
    input_condition = keras.Input(shape=(Vheight, Vwidth, Vchannels))


    x = concatenate([input_real, input_condition],axis=-1)
    x = Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(1, kernel_size=1, strides=1, padding="same", use_bias=False)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)   # activation = None

    # model_input is conditioned by labels
    discriminator = keras.models.Model([input_real, input_condition], x, name='discriminator')
    discriminator.summary()

    return discriminator



def build_generator():
    
    # inputs = concatenate([input_condition,noise],axis=-1)
    
    input_noise = Input(shape=(latent_dim,))
    input_condition = Input(shape=(Vheight, Vwidth, Vchannels))

    encoder_base = ResNet50(weights='imagenet', include_top = False)
    encoder = encoder_base(input_condition)
    encoder = GlobalAveragePooling2D()(encoder)
    encoder = Dense(512, activation='relu')(encoder)
    encoder = concatenate([input_noise,encoder],axis=-1)

    encoder = Dense(14*14*512)(encoder)
    encoder = layers.Reshape((14, 14, 512))(encoder)

    x = UpSampling2D((2, 2))(encoder)
    x = Conv2D(64*4, kernel_size=3, strides=1, padding="same",
                   use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
    x = Conv2D(64*4, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, )(x, training=1)
    x = Activation("relu")(x)
    x = Conv2D(64*4, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
    x = Activation("relu")(x)


    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64*2, kernel_size=3, strides=1, padding="same",
               use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
    x = Conv2D(64*2, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, )(x, training=1)
    x = Activation("relu")(x)
    x = Conv2D(64*2, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
    x = Activation("relu")(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64*2, kernel_size=3, strides=1, padding="same",
               use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
    x = Conv2D(64*2, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, )(x, training=1)
    x = Activation("relu")(x)
    x = Conv2D(64*2, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
    x = Activation("relu")(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64*1, kernel_size=3, strides=1, padding="same",
                   use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x,training=1)
    x = Conv2D(64*1, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, )(x, training=1)
    x = Activation("relu")(x)
    x = Conv2D(64*1, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
    x = Activation("relu")(x)
    x = Conv2D(3, kernel_size=3, strides=1, padding="same", activation="sigmoid",
               use_bias=False,)(x)

    model = keras.models.Model([input_noise, input_condition], x,name='generator')
    model.summary()

    return model

# visual_path = '/home/guan/Desktop/data/visual'
# tactile_path = '/home/guan/Desktop/data/tactile'
# real_images_dir, condition_dir = get_paired(visual_path,tactile_path)
# print(condition_dir[0:30])


def train(visual_dir, tactile_dir, epochs = 10, batch_size=20):
    #build conditional GAN
    discriminator_optimizer = Adam(lr=1e-5, beta_1=0.5, beta_2=0.9)
    gan_optimizer = Adam(lr=1e-5, beta_1=0.5, beta_2=0.9)
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
                        optimizer=discriminator_optimizer,
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
    generator_model.compile(loss = wasserstein_loss, optimizer=gan_optimizer)
    generator_model.summary()


    real_images_dir, condition_dir = get_paired(visual_dir,tactile_dir)

    step_epoch = len(real_images_dir)//batch_size

    epoch_num = 0
    iter_num = 0
    while epoch_num < epochs:
        d_loss = 0
        d_acc = 0
        loss_history_D  = 0
        acc_history_D = 0
        loss_history_G = 0
        acc_history_G = 0
        
        while iter_num < step_epoch:
            train_real = np.array(read_tactile_image(real_images_dir[batch_size*iter_num:batch_size*(iter_num+1)]))
            train_condition = np.array(read_image(condition_dir[batch_size*iter_num:batch_size*(iter_num+1)]))

            random_latent_vectors_D = np.array(np.random.normal(0,1,size=(batch_size, latent_dim)))
            # print(random_latent_vectors.shape)
            #数据加载
            generated_images = generator.predict([random_latent_vectors_D, train_condition])
            
            valid = -np.ones((batch_size, 1))
            fake =  np.ones((batch_size, 1))
            dummy = np.zeros((batch_size, 1))

            d_loss = critic_model.train_on_batch([train_real,random_latent_vectors_D,train_condition],
                                                        [valid, fake, dummy])
            loss_history_D+=d_loss[0]

            shuffle_condition = train_condition.copy()
            np.random.shuffle(shuffle_condition)

            g_loss = generator_model.train_on_batch([random_latent_vectors_D,shuffle_condition], valid)
            
            loss_history_G+= g_loss

            print ("%d [D loss: %f] [G loss: %f]" % (iter_num, d_loss[0], g_loss))

            iter_num += 1

            if iter_num%40==0:
                generated_images = generator.predict([random_latent_vectors_D, train_condition])

                for i in range (batch_size):

                        # 保存生成的图像
                    img = generated_images[i] * 255
                    # print(img.shape)
                    cv2.imwrite(os.path.join(save_dir, 'generated' + 'epoch'+str(epoch_num)+'iter'+ str(iter_num) + 'cate'+str(i)+'.png'),img)
             
                    # 保存真实图像，以便进行比较
                    img = train_condition[i] * 255
                    cv2.imwrite(os.path.join(save_dir, 'condition' + 'epoch'+ str(epoch_num)+ 'iter'+ str(iter_num) + 'cate'+str(i)+'.png'),img)

        # loss_history_D = loss_history_D/(step_epoch)
        # acc_history_D = acc_history_D/step_epoch
        # loss_history_G = loss_history_G/step_epoch
        # acc_history_G = acc_history_G/step_epoch    # print(acc_real,acc_fake)
        print('------------------------------------epoch--------------------------------------')
        print("epoch: %d [D loss: %f] [G loss: %f]" % (epoch_num, loss_history_D/step_epoch, loss_history_G/step_epoch))
        print('------------------------------------epoch--------------------------------------')
        epoch_num += 1
        iter_num = 0 # reset counter

 
Vheight =224
Vwidth = 224
Vchannels = 3
Theight =224
Twidth = 224
Tchannels = 3
latent_dim = 512
visual_path = '/home/guan/Desktop/data/visual'
tactile_path = '/home/guan/Desktop/data/tactile'
save_dir = '/home/guan/Desktop/cgan/generated4'



train(visual_path,tactile_path,epochs = 150, batch_size= 8)
