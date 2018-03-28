% Beril Sirmacek
% University of Twente
% Medical Image Processing
% Exercize-1: Visualization of MRI volume data
%%
clear all
close all
clc
%% Read volumetric data: 
filename = 'Hamid/volume-10.nii';
V = double(niftiread(filename));
V = (255.*((V-min(min(min(V))))./(max(max(max(V)))-min(min(min(V))))));

%% Get slices and visualize as images:

image_list = [100,150, 200, 250, 300, 350, 400, 450] ;

figure,
for(i = 1:size(image_list,2))
    slice = V(:,:,image_list(i));
    slice_list(i) = {slice};
    xImage = [size(slice,1) size(slice,1); 1 1];   %# The x data for the image corners
    yImage = [1 size(slice,2); 0 size(slice,2) ];             %# The y data for the image corners
    zImage = [image_list(i) image_list(i); image_list(i) image_list(i)];   %# The z data for the image corners
    surf(xImage,yImage,zImage,...    %# Plot the surface
         'CData', slice, 'edgecolor', 'none', 'FaceColor','texturemap'); hold on,
end

%% Take one image:

img = slice_list{4};
figure, imshow(img, []);

% show grayscale histogram:
figure, imhist(uint8(img));

% show Matlab thresholding function turns nothing:


% show Gaussian mixture model can help to determine threshold value:


%% 

img_filt = medfilt2(img, [9,9]);

% find the threshold value:
[counts,x] = imhist(uint8(img_filt),256);
figure, plot(counts);
T = otsuthresh(counts);
BW = imbinarize(img,T);
figure
imshow(BW)

% mask:
mask = imbinarize(img,level);
figure, imshow(mask);

 
%% Region of Interest (body):


%% Segmentation: 

%% Gradients: 

%% Region Growing: 


%% Differences of Gaussians:


%% Adding Salt and Pepper Noise:

%% Filtering the noise with median filter:

%% Linear and Non-linear filters (bilateral filter):

%% 