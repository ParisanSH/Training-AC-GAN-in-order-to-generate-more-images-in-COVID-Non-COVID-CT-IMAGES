clear
clc

alex=alexnet;
layers = alex.Layers 

layers(23)=fullyConnectedLayer(2);
layers(25)=classificationLayer;

Itrain=imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
Itest=imageDatastore('TestGEN','IncludeSubfolders',true,'LabelSource','foldernames');

opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',30,'Minibatchsize',64,'Plots','training-progress');
NET=trainNetwork(Itrain,layers,opts);


TEST=classify(NET,Itest);
accuracy=mean(TEST==Itest.Labels)

A=Itest.Labels;

Lcov=A(1:50);
Lnon=A(51:end);
Tcov=TEST(1:50);
Tnon=TEST(51:end);

TP=sum(Tcov==Lcov)
FN=50-TP
TN=sum(Tnon==Lnon)
FP=50-TN

F1=TP/(TP+0.5*(FP+FN))

precision=TP/(TP+FP)
recall=TP/(TP+FN)

sensitivity=TP/(TP+FN);
specificity=TN/(TN+FP);
%ROC=plot(sensitivity,specificity)
%AUC=trapz(sensitivity,specificity)
