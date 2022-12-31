# Training-AC-GAN-in-order-to-generate-more-images-in-COVID-Non-COVID-CT-IMAGES
The purpose of this project is to generate more images than the given dataset. The AC-GAN has been trained to generate images while predicting their classes.

In order to do that, I have used the COVID, Non-COVID CT IMAGES dataset which is available in the below link: https://drive.google.com/drive/folders/1kVIe0HIYz_k9Jcjn27ViHPe51AG9y_fr?usp=sharings

# Problem Definition
GAN consists of two primary networks: Generator, 𝐺(𝑧), and Discriminator, 𝐷(𝑥). The generator module is used to create artificial samples of data by incorporating feedback from the discriminator. Its objective is to deceive the discriminator into classifying the generated data as belonging to the original dataset and ultimately minimize the cost function: min(𝐺)max(𝐷) 𝐸(𝑥∼𝑃data(𝑥)) log𝐷(𝑥) + 𝐸(𝑧∼𝑃(𝑧)) log(1 − 𝐷(𝐺(𝑧))

The Auxiliary Classifier GAN( AC-GAN: https://arxiv.org/pdf/1610.09585.pdf) is simply an extension of GAN that requires the discriminator to not only predict if the image is ‘real’ or ‘fake’, but also provide the ‘source’ or the ‘class label’ of the given image.
In AC-GAN, each generated sample has a corresponding class label, 𝑐 ~ 𝑃(𝑐), in addition to the noise 𝑧. Generator 𝐺 tries to generate 𝑥fake = 𝐺(𝑐, 𝑧). The discriminator 𝐷 gives probability distributions over both sources and class labels, 𝑃(𝑠|𝑥), 𝑃(𝑐|𝑥) = 𝐷(𝑥). The objective function has two parts: the loglikelihood of the correct source, 𝐿s, and the log-likelihood of the correct class, 𝐿c.

𝐿𝑠 = 𝐸[log 𝑃(𝑠 = real | 𝑥real )] + 𝐸[log 𝑃(𝑠 = fake | 𝑥fake )]
𝐿𝑐 = 𝐸[log 𝑃(𝑐 = real | 𝑥real )] + 𝐸[log 𝑃(𝑐 = fake | 𝑥fake )]
𝐷 is trained to maximize 𝐿𝑐 + 𝐿𝑠 while 𝐺 is trained to maximize 𝐿𝑐 − 𝐿𝑠. For more information, refer to AC-GAN paper.

# Steps:
1) First, the data has been divided into 80% for training and 20% for testing the model. Note that there is more than one image for a patient, so the images of a particular patient should not be in both training and test data.
2) Some necessary pre-processing on the images has been done, and since the images in the database have different sizes, all images should resize suitably for entering the network.
3) The AC-GAN has been trained in order to can generate COVID and Non-COVID images with labels. 
4) The ConvNet, which was trained in https://github.com/ParisanSH/A-Deep-ConvNet-to-classify-COVID-and-Non-COVID-images., has been used and it has been tested with the generated (labeled) images in this project.
5) Finally, the ROC curve has been plotted.
