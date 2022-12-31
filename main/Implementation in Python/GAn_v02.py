from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from PIL import Image
import sys

from os import listdir
from numpy import asarray
import numpy as np
X_covid=np.zeros((1000,28,28))

np.set_printoptions(threshold=sys.maxsize)  # to see all elements of of array
i=0
for filename in listdir('COVID_mod_Train/'):

    image = Image.open('COVID_mod_Train/' + filename)  # open colour image

    X_covid[i,:,:]=image
    i=i+1


############################
X_NonCovid=np.zeros((999,28,28))

np.set_printoptions(threshold=sys.maxsize)  # to see all elements of of array
i=0
for filename in listdir('NonCOVID_mod_Train/'):

    image = Image.open('NonCOVID_mod_Train/' + filename)  # open colour image

    X_NonCovid[i,:,:]=image
    i=i+1
X_train=np.concatenate((X_covid, X_NonCovid), axis=0)
Y_covid=np.zeros((1000,))
Y_NonCovid=np.ones((999,))
Y_train=np.concatenate((Y_covid, Y_NonCovid), axis=0)
print(Y_train.shape)


# example of fitting an auxiliary classifier gan (ac-gan) on fashion mnsit
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot



# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=2):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=in_shape)
	# downsample to 14x14
	fe = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# normal
	fe = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# downsample to 7x7
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# normal
	fe = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# real/fake output
	out1 = Dense(1, activation='sigmoid')(fe)
	# class label output
	out2 = Dense(n_classes, activation='softmax')(fe)
	# define model
	model = Model(in_image, [out1, out2])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=2):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 7 * 7
	li = Dense(n_nodes, kernel_initializer=init)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 384 * 7 * 7
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = Reshape((7, 7, 384))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(192, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(merge)
	gen = BatchNormalization()(gen)
	gen = Activation('relu')(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	out_layer = Activation('tanh')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect the outputs of the generator to the inputs of the discriminator
	gan_output = d_model(g_model.output)
	# define gan model as taking noise and label and outputting real/fake and label outputs
	model = Model(g_model.input, gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
	return model

# load images
def load_real_samples(X_train,Y_train):
    ###mahdi start
    # load dataset

    # (trainX, trainy), (_, _) = load_data()

    trainX = X_train
    trainy = Y_train

    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    print(X.shape, trainy.shape)
    return [X, trainy]

# select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=2):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
	# prepare fake examples
	[X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(100):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename1 = 'generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%04d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_steps):
		# get randomly selected 'real' samples
		[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
		# update discriminator model weights
		_,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
		# generate 'fake' examples
		[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model weights
		_,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
		# prepare points in latent space as input for the generator
		[z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		_,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
		# summarize loss on this batch
		print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f,d_f2, g_1,g_2))
		# evaluate the model performance every 'epoch'
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, latent_dim)

# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples(X_train,Y_train)
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)



# example of loading the generator model and generating images
from math import sqrt
from numpy import asarray
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_class):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = asarray([n_class for _ in range(n_samples)])
	return [z_input, labels]

# create and save a plot of generated images
def save_plot(examples, n_examples):
	# plot images
	for i in range(n_examples):
		# define subplot
		pyplot.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()

# load model
model = load_model('model_3100.h5')
latent_dim = 100
n_examples = 10000 # must be a square
n_class = 0 # sneaker
# generate images
latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
# generate images
X  = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
save_plot(X, n_examples)


# baseline model with dropout on the cifar10 dataset
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD

from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from sklearn import metrics
from PIL import Image
import sys
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from os import listdir
from numpy import asarray
import numpy as np

from numpy import argmax

np.set_printoptions(threshold=sys.maxsize)  # to see all elements of of array

######################
###      reading train data    #########
X_covid_train = np.zeros((1000, 28, 28))
i = 0
for filename in listdir('COVID_mod_Train/'):
    image = Image.open('COVID_mod_Train/' + filename)  # open colour image

    X_covid_train[i, :, :] = image
    i = i + 1

############################
X_NonCovid_train = np.zeros((999, 28, 28))

np.set_printoptions(threshold=sys.maxsize)  # to see all elements of of array
i = 0
for filename in listdir('NonCOVID_mod_Train/'):
    image = Image.open('NonCOVID_mod_Train/' + filename)  # open colour image

    X_NonCovid_train[i, :, :] = image
    i = i + 1

X_train = np.concatenate((X_covid_train, X_NonCovid_train), axis=0)
Y_covid_train = np.zeros((1000,))
Y_NonCovid_train = np.ones((999,))
Y_train = np.concatenate((Y_covid_train, Y_NonCovid_train), axis=0)
###############################
###        reading test data      ########
X_covid_test = np.zeros((252, 28, 28))

i = 0
for filename in listdir('COVID_mod_Test/'):
    image = Image.open('COVID_mod_Test/' + filename)  # open colour image

    X_covid_test[i, :, :] = image
    i = i + 1

############################
X_NonCovid_test = np.zeros((230, 28, 28))

np.set_printoptions(threshold=sys.maxsize)  # to see all elements of of array
i = 0
for filename in listdir('NonCOVID_mod_Test/'):
    image = Image.open('NonCOVID_mod_Test/' + filename)  # open colour image

    X_NonCovid_test[i, :, :] = image
    i = i + 1

X_test = np.concatenate((X_covid_test, X_NonCovid_test), axis=0)
Y_covid_test = np.zeros((252,))
Y_NonCovid_test = np.ones((230,))
Y_test = np.concatenate((Y_covid_test, Y_NonCovid_test), axis=0)

##################3

width, height, channels = X.shape[1], X.shape[2], 1
test_GAN = X.reshape((X.shape[0], width, height, channels))
Y_test_GAN = np.zeros((100,))
Y_test_GAN[99] = 1
Y_test_GAN = to_categorical(Y_test_GAN)


width, height, channels = X_test.shape[1], X_test.shape[2], 1
X_test2=X_test.reshape((X_test.shape[0], width, height, channels))
# _, acc = model.evaluate(test_GAN, Y_test_GAN, verbose=0)
# print('> %.3f' % (acc * 100.0))



# load train and test dataset
def load_dataset(X_train, Y_train, X_test, Y_test):
    # load dataset
    # (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainX = X_train
    trainY = Y_train
    testX = X_test
    testY = Y_test
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(trainX, testX):
    # reshape grayscale images to have a single channel
    width, height, channels = trainX.shape[1], trainX.shape[2], 1
    train = trainX.reshape((trainX.shape[0], width, height, channels))
    test = testX.reshape((testX.shape[0], width, height, channels))
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


# define cnn model
def define_model():
    model = Sequential()
    model.add(
        Conv2D(28, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(28, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset(X_train, Y_train, X_test, Y_test)
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    print(trainX.shape)
    # define model
    model = define_model()
    # fit model
    history = model.fit(trainX, trainY, epochs=100, batch_size=64, verbose=0)
    y_predict = model.predict(test_GAN)
    y_predict2 = np.zeros(len(Y_test_GAN))
    testY_main=np.zeros(len(Y_test_GAN))
    for j in range(len(Y_test_GAN)):
        y_predict2[j] = argmax(y_predict[j])
        testY_main[j] = 1
    print(y_predict2)
    print(testY_main)
    precision, recall, fscore, _ = precision_recall_fscore_support(testY_main, y_predict2, average='macro')
    print('precision is: ', ' %.3f' % (precision * 100))
    print('recall is: ', ' %.3f' % (recall * 100))
    print('fscore is: ', ' %.3f' % (fscore * 100))
    ###
    accuracy = accuracy_score(testY_main, y_predict2)
    print('accuracy is: ', ' %.3f' % (accuracy * 100))
    ###
    fpr, tpr, thresholds = metrics.roc_curve(testY_main, y_predict2)
    plt.plot(fpr, tpr)
    #####
    auc = metrics.auc(fpr, tpr)
    print('auc is: ', ' %.3f' % (auc))
    ###########
    X_test3=np.concatenate((X_test2[0:round(0.2*len(Y_test))], test_GAN), axis=0)
    Y_test3=np.concatenate((Y_test[0:round(0.2*len(Y_test))], testY_main), axis=0)

    y_predict = model.predict(X_test3)
    y_predict2 = np.zeros(len(Y_test3))
    for j in range(len(Y_test_GAN)):
        y_predict2[j] = argmax(y_predict[j])

    precision, recall, fscore, _ = precision_recall_fscore_support(Y_test3, y_predict2, average='macro')
    print('precision is: ', ' %.3f' % (precision * 100))
    print('recall is: ', ' %.3f' % (recall * 100))
    print('fscore is: ', ' %.3f' % (fscore * 100))
    ###
    accuracy = accuracy_score(Y_test3, y_predict2)
    print('accuracy is: ', ' %.3f' % (accuracy * 100))
    ###
    fpr, tpr, thresholds = metrics.roc_curve(Y_test3, y_predict2)
    plt.plot(fpr, tpr)
    #####
    auc = metrics.auc(fpr, tpr)
    print('auc is: ', ' %.3f' % (auc))

    ########3
    ######
    #######3
    X_test3=np.concatenate((X_test2, test_GAN), axis=0)
    Y_test3=np.concatenate((Y_test, testY_main), axis=0)

    y_predict = model.predict(X_test3)
    y_predict2 = np.zeros(len(Y_test3))
    for j in range(len(Y_test_GAN)):
        y_predict2[j] = argmax(y_predict[j])

    precision, recall, fscore, _ = precision_recall_fscore_support(Y_test3, y_predict2, average='macro')
    print('precision is: ', ' %.3f' % (precision * 100))
    print('recall is: ', ' %.3f' % (recall * 100))
    print('fscore is: ', ' %.3f' % (fscore * 100))
    ###
    accuracy = accuracy_score(Y_test3, y_predict2)
    print('accuracy is: ', ' %.3f' % (accuracy * 100))
    ###
    fpr, tpr, thresholds = metrics.roc_curve(Y_test3, y_predict2)
    plt.plot(fpr, tpr)
    #####
    auc = metrics.auc(fpr, tpr)
    print('auc is: ', ' %.3f' % (auc))
run_test_harness()






