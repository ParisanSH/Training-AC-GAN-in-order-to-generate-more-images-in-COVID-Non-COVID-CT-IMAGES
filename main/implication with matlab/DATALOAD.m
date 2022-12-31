clear
clc

train1=imageDatastore('train\COVID');
train2=imageDatastore('train\Non-COVID');
test1=imageDatastore('test\COVID');
test2=imageDatastore('test\Non-COVID');

for ii=1:252
    Test1=readimage(test1,ii);
    Test1=imresize(Test1,[28,28]);
    Test1=rgb2gray(Test1);
    TEST1{ii,:}=Test1;
end
for ii=1:230
    Test2=readimage(test2,ii);
    Test2=imresize(Test2,[28,28]);
    Test2=rgb2gray(Test2);
    TEST2{ii,:}=Test2;
end
for ii=1:1000
    Train1=readimage(train1,ii);
    Train1=imresize(Train1,[28,28]);
    Train1=rgb2gray(Train1);
    TRAIN1{ii,:}=Train1;
end
for ii=1:999
    Train2=readimage(train2,ii);
    Train2=imresize(Train2,[28,28]);
    Train2=rgb2gray(Train2);
    TRAIN2{ii,:}=Train2;
end

TRAIN(1:1000)=TRAIN1;
TRAIN(1001:1999)=TRAIN2;
TRAIN=cat(3, TRAIN{:});

TEST(1:252)=TEST1;
TEST(253:482)=TEST2;
TEST=cat(3, TEST{:});

DATA.train_images=TRAIN;
DATA.test_images=TEST;


label1=ones(1,1000);
label2=zeros(1,999);
Ltrain(1:1000)=label1;
Ltrain(1001:1999)=label2;

label1=ones(1,252);
label2=zeros(1,230);
Ltest(1:252)=label1;
Ltest(253:482)=label2;

DATA.label_train=Ltrain';
DATA.label_test=Ltest';
