import numpy as np
import os
import cv2
from os.path import isfile,isdir

from numpy.random import randn,randint

from keras.optimizers import Adam
from keras.models import Model,Sequential

from keras.layers import Input,multiply,Dense,Reshape,Flatten,Conv2D,Conv2DTranspose,LeakyReLU,BatchNormalization,Dropout,Embedding,Activation,Concatenate
from keras.initializers import RandomNormal
from keras.layers.convolutional import UpSampling2D, Conv2D

from matplotlib import pyplot


def load_data(path, label, image_size=(256, 256)):
    images = list()
    images_name = os.listdir(path)
    for index, image_name in enumerate(images_name):
        print(f'\rreading {label} data. image : {index + 1} / {len(images_name)}', end='')
        image = cv2.imread(os.path.join(path, image_name),0)

        if image is not None:
            image = cv2.resize(image, image_size)
            images.append(image)

    labels = np.full(shape=len(images), fill_value=label)
    print()
    return images, labels


def discriminator_model(in_shape = (224, 224, 1), n_classes=2):
    model = Sequential()

    model.add(Conv2D(32,( 3, 3), strides=(2,2),padding='same',input_shape=in_shape))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(64,( 3, 3), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())

    image = Input(shape=in_shape)

    features = model(image)

    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(2, activation='softmax', name='auxiliary')(features)

    model = Model(image, [fake, aux])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


def generator_model(latent_dim, n_classes=2):
    init = RandomNormal(stddev=0.02)
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = 56 * 56 * 3
    li = Dense(n_nodes, kernel_initializer=init)(li)
    li = Reshape((56, 56, 3))(li)
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 384 * 56 * 56
    gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
    gen = Activation('relu')(gen)
    gen = Reshape((56, 56, 384))(gen)
    merge = Concatenate()([gen, li])
    gen = Conv2DTranspose(192, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(merge)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    gen = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    out_layer = Activation('tanh')(gen)
    model = Model([in_lat, in_label], out_layer)
    return model


def gan_model(generator_model, descriminator_model,learning_rate = 0.0001):
    descriminator_model.trainable = False
    gan_output = descriminator_model(generator_model.output)
    model = Model(generator_model.input, gan_output)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],optimizer=Adam(learning_rate=learning_rate))
    return model
    

def create_real_samples(x_train, y_train, samples_count):
    random_indexs = randint(0, x_train.shape[0], samples_count)
    x, y = x_train[random_indexs], y_train[random_indexs]
    valid_label  = np.ones((samples_count, 1))
    return x, y, valid_label 


def create_fake_samples(generator_model, latent_size, samples_count,classes_count):
    z = np.random.uniform(-1.0, 1.0, size=[samples_count, latent_size])
    y = randint(0, classes_count, samples_count)
    
    x = generator_model.predict([z, y])
    fake_label = np.zeros((samples_count, 1))
    return x, y, fake_label


def train(gan_model, generator_model, descriminator_model, x_train,y_train, latent_size, samples,path,epochs=50, batch=32):
    total_batch_count = len(x_train) // batch
    half_batch = batch // 2
    iterations = total_batch_count * epochs
    for i in range(iterations):
        print(f'\riter {i} / {iterations}',end = '')
        x_real, x_real_labels, y_real = create_real_samples(x_train,y_train, half_batch)
        x_fake, x_fake_labels, y_fake = create_fake_samples(generator_model, latent_size, half_batch,2)
        
        descriminator_model.train_on_batch(x_real, [y_real, x_real_labels])
        descriminator_model.train_on_batch(x_fake, [y_fake, x_fake_labels])

        
        z = np.random.uniform(-1.0, 1.0, size=[batch, latent_size])
        z_labels = randint(0, 2, batch)
        y_gan = np.ones((batch, 1))
        _,g_1,g_2 = gan_model.train_on_batch([z, z_labels], [y_gan, z_labels])


        if (i+1)%total_batch_count == 0:
            show_result(i, generator_model, samples,path)
    print()
                
            
def show_result(index, generator_model, samples,path):
    z,y = samples
    images = generator_model.predict([z,y])
    images = (images * 127.5) + 127.5
    images = images.astype('uint8')

    n = len(images)
    f, axs = pyplot.subplots(int(np.ceil(n/4)), 4,figsize=(15,15))
    axs = axs.flat
    for i in range(n):
        axs[i].axis('off')
        axs[i].imshow(images[i, :, :, 0],cmap='gray')


    image_name = f'{path}/image{index}.png'
    pyplot.savefig(image_name)
    pyplot.show()

    model_name = f'{path}/model{index}.h5'
    generator_model.save(model_name)
    print(f'saves data {index}')  
            
            
            
def main():
    image_size=(224, 224)
    
    path_covid_train = '/content/drive/MyDrive/Copy of COVID,Non COVID-CT Images.rar (Unzipped Files)/COVID,Non COVID-CT Images/train/COVID'
    path_non_covid_train = '/content/drive/MyDrive/Copy of COVID,Non COVID-CT Images.rar (Unzipped Files)/COVID,Non COVID-CT Images/train/Non-COVID'
    path_covid_test = '/content/drive/MyDrive/Copy of COVID,Non COVID-CT Images.rar (Unzipped Files)/COVID,Non COVID-CT Images/test/COVID'
    path_non_covid_test = '/content/drive/MyDrive/Copy of COVID,Non COVID-CT Images.rar (Unzipped Files)/COVID,Non COVID-CT Images/test/Non-COVID'

    path_train_data = '/content/drive/MyDrive/nndl3/train_data.npy'
    path_train_data_label = '/content/drive/MyDrive/nndl3/train_data_label.npy'
    path_test_data = '/content/drive/MyDrive/nndl3/test_data.npy'
    path_test_data_label = '/content/drive/MyDrive/nndl3/test_data_label.npy'


    if isfile(path_train_data) and isfile(path_train_data_label) and isfile(path_test_data) and isfile(path_test_data_label): 
    
        f1 = open(path_train_data,'rb')
        f2 = open(path_train_data_label,'rb')
        f3 = open(path_test_data,'rb')
        f4 = open(path_test_data_label,'rb')

        train_data = np.load(f1)
        train_data_label = np.load(f2)
        test_data = np.load(f3)
        test_data_label = np.load(f4)
        print('load data complete.')
    
    else:
        print('reading train data...')
        images_covid_train, images_covid_train_labels = load_data(path_covid_train, 'covid',image_size)
        images_non_covid_train, images_non_covid_train_labels = load_data(path_non_covid_train, 'non_covid',image_size)
        print('reading test data...')
        images_covid_test, images_covid_test_labels = load_data(path_covid_test, 'covid',image_size)
        images_non_covid_test, images_non_covid_test_labels = load_data(path_non_covid_test, 'non_covid',image_size)
        
        train_data = np.vstack([images_covid_train, images_non_covid_train])
        train_data_label = np.hstack([images_covid_train_labels, images_non_covid_train_labels])
        
        test_data = np.vstack([images_covid_test, images_non_covid_test])
        test_data_label = np.hstack([images_covid_test_labels, images_non_covid_test_labels])

        train_data_label[train_data_label=='covid'] = 1
        train_data_label[train_data_label=='non_covid'] = 0
        train_data_label = train_data_label.astype('int32')
        
        test_data_label[test_data_label=='covid'] = 1
        test_data_label[test_data_label=='non_covid'] = 0
        test_data_label = test_data_label.astype('int32')
        
        print('saving data.')
        f1 = open(path_train_data,'wb')
        f2 = open(path_train_data_label,'wb')
        f3 = open(path_test_data,'wb')
        f4 = open(path_test_data_label,'wb')

        np.save(f1,train_data)
        np.save(f2,train_data_label)
        np.save(f3,test_data)
        np.save(f4,test_data_label)

        print('load data complete.')
        
    train_data = np.expand_dims(train_data, axis=3)    
    train_data = train_data.astype('float32')
    train_data = (train_data - 127.5) / 127.5
        
    path = '/content/drive/MyDrive/gan'
    
    latent_size = 100
    sample_n = 12
    

    z = np.random.uniform(-1.0, 1.0, size=[sample_n, latent_size])
    y = randint(0, 2, sample_n)
    samples = (z,y)
    generator = generator_model(latent_size)
    discriminator = discriminator_model()
    gan = gan_model(generator, discriminator)
        
    train(gan, generator, discriminator, train_data,train_data_label, latent_size,samples, path,epochs=50, batch=32)
    return generator,discriminator,gan
  
generator,discriminator,gan = main()