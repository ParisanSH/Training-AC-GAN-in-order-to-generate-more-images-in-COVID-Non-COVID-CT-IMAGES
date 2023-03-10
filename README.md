# Training-AC-GAN-in-order-to-generate-more-images-in-COVID-Non-COVID-CT-IMAGES
The purpose of this project is to generate more images than the given dataset. The AC-GAN has been trained to generate images while predicting their classes.

In order to do that, I have used the COVID, Non-COVID CT IMAGES dataset which is available in the below link: https://drive.google.com/drive/folders/1kVIe0HIYz_k9Jcjn27ViHPe51AG9y_fr?usp=sharings

# Problem Definition
GAN consists of two primary networks: Generator, πΊ(π§), and Discriminator, π·(π₯). The generator module is used to create artificial samples of data by incorporating feedback from the discriminator. Its objective is to deceive the discriminator into classifying the generated data as belonging to the original dataset and ultimately minimize the cost function: min(πΊ)max(π·) πΈ(π₯βΌπdata(π₯)) logπ·(π₯) + πΈ(π§βΌπ(π§)) log(1 β π·(πΊ(π§))

The Auxiliary Classifier GAN( AC-GAN: https://arxiv.org/pdf/1610.09585.pdf) is simply an extension of GAN that requires the discriminator to not only predict if the image is βrealβ or βfakeβ, but also provide the βsourceβ or the βclass labelβ of the given image.
In AC-GAN, each generated sample has a corresponding class label, π ~ π(π), in addition to the noise π§. Generator πΊ tries to generate π₯fake = πΊ(π, π§). The discriminator π· gives probability distributions over both sources and class labels, π(π |π₯), π(π|π₯) = π·(π₯). The objective function has two parts: the loglikelihood of the correct source, πΏs, and the log-likelihood of the correct class, πΏc.

πΏπ  = πΈ[log π(π  = real | π₯real )] + πΈ[log π(π  = fake | π₯fake )]
πΏπ = πΈ[log π(π = real | π₯real )] + πΈ[log π(π = fake | π₯fake )]
π· is trained to maximize πΏπ + πΏπ  while πΊ is trained to maximize πΏπ β πΏπ . For more information, refer to AC-GAN paper.

# Steps:
1) First, the data has been divided into 80% for training and 20% for testing the model. Note that there is more than one image for a patient, so the images of a particular patient should not be in both training and test data.
2) Some necessary pre-processing on the images has been done, and since the images in the database have different sizes, all images should resize suitably for entering the network.
3) The AC-GAN has been trained in order to can generate COVID and Non-COVID images with labels. 
4) The ConvNet, which was trained in https://github.com/ParisanSH/A-Deep-ConvNet-to-classify-COVID-and-Non-COVID-images., has been used and it has been tested with the generated (labeled) images in this project.
5) Finally, the ROC curve has been plotted.
