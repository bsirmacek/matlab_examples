% Beril Sirmacek
% University of Twente
% Medical Image Processing
% Exercise-1: Visualization and processing of MRI volume data
% April 2018
%%
clear all
close all
clc
%% Read volumetric data: 
filename = 'C:\Users\sirmacekb\OneDrive for Business\development\MRI\Hamid\volume-10.nii';
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

%% Matlab thresholding function: 

clear V;

img = slice_list{4};
figure, imshow(img, []);

% normalize the image:
img_norm = (img-min(img(:)))./(max(img(:))-min(img(:)));

% find the threshold value:
[counts,x] = imhist(uint8(img),256);
figure, plot(counts);
T = otsuthresh(counts);

BW = imbinarize(img_norm,T);
figure
imshow(BW)

%% Image histogram analysis:


% show grayscale histogram:
img2 = 255.*(img-min(img(:)))./(max(img(:))-min(img(:)));
figure, histogram(uint8(img2)); hold on,

X = 0:255;
m1 = 110;
s1 = 15;
mag1 = 300000;
f1 = gauss_distribution(X, m1, s1);
plot(X,mag1*f1,'r')

% Gaussian mixture model for two classes:
m2 = 6;
s2 = 10;
mag2 = 600000;
f2 = gauss_distribution(X, m2, s2);
plot(X,mag2*f2,'r')

threshold = 50; % intersection of the Gaussians
mask1 = img2 > threshold;
figure, imshow(mask1);


% show grayscale histogram again:
figure, histogram(uint8(img2)); hold on,

% Gaussian mixture model for three classes:
X = 0:255;

% Gaussian 1:
m1 = 114;
s1 = 5;
mag1 = 300000;
f1 = gauss_distribution(X, m1, s1);
plot(X,mag1*f1,'r')

% Gaussian 2:
m2 = 99;
s2 = 5;
mag2 = 500000;
f2 = gauss_distribution(X, m2, s2);
plot(X,mag2*f2,'r')

% Gaussian 3:
m3 = 6;
s3 = 10;
mag2 = 600000;
f3 = gauss_distribution(X, m3, s3);
plot(X,mag2*f3,'r')

threshold2 = 108; % intersection of the Gaussians
mask2 = img2 > threshold2;
figure, imshow(mask2);
 
%% Morphological operations:

% Use the first mask to detect the human body segment:
figure, imshow(mask1);
mask1_filled = imfill(mask1, 'holes');
figure, imshow(mask1_filled);

% Connected component analysis:
[L, n] = bwlabel(mask1_filled);
figure, imshow(L,[]); 
colormap(jet);

% Find some morphological properties:
cc = bwconncomp(L); 
stats = regionprops(cc, 'Area','Eccentricity'); 
% Eccentricity of the ellipse that has the same second-moments as the 
% region, returned as a scalar. The eccentricity is the ratio of the 
% distance between the foci of the ellipse and its major axis length. 
% The value is between 0 and 1. (0 and 1 are degenerate cases. An ellipse 
% whose eccentricity is 0 is actually a circle, while an ellipse whose 
% eccentricity is 1 is a line segment.)
idx = find([stats.Area] > 80 & [stats.Eccentricity] < 0.8); 
body_mask= ismember(labelmatrix(cc), idx); 
figure, imshow(body_mask);

% plot boundaries:
B = bwboundaries(body_mask,'noholes'); % the function returns a cell
boundary = B{1};
figure, imshow(img, []);
hold on
plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)

% Get the small segments, which are only inside of the body_mask
segments = body_mask & mask2;
segments = imclose(segments, strel('disk',2));
figure, imshow(segments);

small_segments = segments - bwareaopen(segments, 200); %trick!
figure, imshow(small_segments);

% there is noise close to the border
% if we remove the small areas, we might lose small tumors
% remove the areas which are close to the border instead
boundary_mask = zeros(size(body_mask));
for (i=1:size(boundary,1))
    boundary_mask(boundary(i,1), boundary(i,2)) = 1;
end
figure, imshow(boundary_mask);

dilated_mask = imdilate(boundary_mask, strel('disk',50));
figure, imshow(dilated_mask);    

small_segments_cleaned = small_segments - (small_segments&dilated_mask); 
small_segments_cleaned = bwareaopen(small_segments_cleaned,5);
figure, imshow(small_segments_cleaned);

% show boundaries and mass centers of the segments:
B = bwboundaries(small_segments_cleaned,'noholes'); % the function returns a cell
s = regionprops(bwlabel(small_segments_cleaned), 'centroid');
centroids = cat(1,s.Centroid);
figure, imshow(img, []);
hold on
plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
colors=['b' 'g' 'r' 'c' 'm' 'y'];
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'c', 'LineWidth', 2)
   
   col = boundary(length(boundary),2); 
   row = boundary(length(boundary),1);
   h = text(col+1, row-1, num2str(k));
   cidx = mod(k,length(colors))+1;
   set(h,'Color',colors(cidx),'FontSize',14,'FontWeight','bold');
end
hold on,
plot(centroids(:,1), centroids(:,2), 'b+')
hold off

%% Show gradient vectors:

img_mat = mat2gray(img);
[imx,imy] = gaussgradient(img_mat, 2);

figure, 
imshow(abs(imx)+abs(imy)); title('sigma=2.0');
mag = (abs(imx)+abs(imy));
figure, 
imshow(mag,[]);

figure, 
imshow(img, []);
hold on;
quiver(imx(1:end,1:end),imy(1:end,1:end), 'MarkerSize',6);
title('sigma=2.0');

%% Gradient Vector Flow (GVF):
% Find info: http://www.iacl.ece.jhu.edu/static/gvf/ 
% Examples: http://www.iacl.ece.jhu.edu/~chenyang/research/levset/movie/index.html


% See the regions again to choose a seed point:
figure, imshow(img, []);
hold on
plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
colors=['b' 'g' 'r' 'c' 'm' 'y'];
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'c', 'LineWidth', 2)
   
   col = boundary(length(boundary),2); 
   row = boundary(length(boundary),1);
   h = text(col+1, row-1, num2str(k));
   cidx = mod(k,length(colors))+1;
   set(h,'Color',colors(cidx),'FontSize',14,'FontWeight','bold');
end
hold on,
plot(centroids(:,1), centroids(:,2), 'b+')
hold off

seed_point_no = 13;
seed_x = round(centroids(seed_point_no,2));
seed_y = round(centroids(seed_point_no,1));

% growing
figure, 
J = double(histeq(uint8(img)));
imshow(J, []), 
hold on
poly = regiongrowing(J, [seed_x, seed_y], 12); % click somewhere inside the lungs
plot(poly(:,1), poly(:,2), 'LineWidth', 2)

%% bilateral filtering:

% Set bilateral filter parameters.
w     = 5;       % bilateral filter half-width
sigma = [1 0.01]; % bilateral filter standard deviations

% Apply bilateral filter to each image.
bflt_img1 = bfilter2(img_norm,w,sigma);
figure, imshow(bflt_img1, []);


% just to give a color image bilateral filtering example:
test_im = imread('stones.jpg');
figure, imshow(test_im);

% Set bilateral filter parameters.
w     = 5;       % bilateral filter half-width
sigma = [3 0.1]; % bilateral filter standard deviations

test_im = double(test_im);
test_im(:,:,1) = test_im(:,:,1)./(max(max(test_im(:,:,1)))); 
test_im(:,:,2) = test_im(:,:,2)./(max(max(test_im(:,:,2)))); 
test_im(:,:,3) = test_im(:,:,3)./(max(max(test_im(:,:,3)))); 
% Apply bilateral filter to each image.
bflt_img2 = bfilter2(test_im,w,sigma);
figure, imshow(bflt_img2, []);

% difference between the original and the bilateral filtered (texture):
dif_im(:,:,1) = double(bflt_img2(:,:,1)) - test_im(:,:,1);
dif_im(:,:,2) = double(bflt_img2(:,:,2)) - test_im(:,:,2);
dif_im(:,:,3) = double(bflt_img2(:,:,3)) - test_im(:,:,3);
dif_im(:,:,1) = dif_im(:,:,1)./(max(max(dif_im(:,:,1)))); 
dif_im(:,:,2) = dif_im(:,:,2)./(max(max(dif_im(:,:,2)))); 
dif_im(:,:,3) = dif_im(:,:,3)./(max(max(dif_im(:,:,3)))); 
figure, imshow(dif_im,[]);

%%


