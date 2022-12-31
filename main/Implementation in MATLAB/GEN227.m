clear
clc

GEN=imageDatastore('\gannn\generated');


for ii=1:100
    GG=readimage(GEN,ii);
    %AA=rgb2gray(AA);
    GG=imresize(GG,[227,227]);
    
    imwrite(GG,sprintf('%d.jpg',ii)) 
end






































