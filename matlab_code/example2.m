% Beril Sirmacek
% University of Twente
% Medical Image Processing
% Exercise-1: Visualization and processing of MRI volume data
% April 2018
%%
clear all
close all
clc
%% Step 1: Read training images
% Read the reference image containing the object of interest.
tumorImage = imresize(rgb2gray(imread('tumor_training.jpg')),4);
figure;
imshow(tumorImage);
title('Image of a tumor');

%% Read the test image containing a tumor
brainImage = imresize(rgb2gray(imread('tumor_test1.jpg')),4);
figure; 
imshow(brainImage);
title('Image of a Cluttered Scene');

%% Step 2: Detect Feature Points
% Detect feature points in both images.
tumorPoints = detectSURFFeatures(tumorImage);
brainPoints = detectSURFFeatures(brainImage);

%% Visualize the feature points found in the training image.
figure; 
imshow(tumorImage);
title(' Feature Points');
hold on;
plot(tumorPoints);

%% Visualize the strongest feature points found in the target image.
figure; 
imshow(brainImage);
title('Feature Points from Test Image');
hold on;
plot(brainPoints);

%% Step 3: Extract Feature Descriptors
% Extract feature descriptors at the interest points in both images.
[tumorFeatures, tumorPoints] = extractFeatures(tumorImage, tumorPoints);
[brainFeatures, brainPoints] = extractFeatures(brainImage, brainPoints);

%% Step 4: Find Putative Point Matches
% Match the features using their descriptors. 
tumorPairs = matchFeatures(tumorFeatures, brainFeatures, 'MaxRatio', 0.9);

%% 
% Display putatively matched features. 
matchedtumorPoints = tumorPoints(tumorPairs(:, 1), :);
matchedbrainPoints = brainPoints(tumorPairs(:, 2), :);
figure;
showMatchedFeatures(tumorImage, brainImage, matchedtumorPoints, ...
    matchedbrainPoints, 'montage');
title('Putatively Matched Points (Including Outliers)');

%% Step 5: Locate the Object in the Scene Using Putative Matches
% |estimateGeometricTransform| calculates the transformation relating the
% matched points, while eliminating outliers. This transformation allows us
% to localize the object in the scene.
[tform, inliertumorPoints, inlierbrainPoints] = ...
    estimateGeometricTransform(matchedtumorPoints, matchedbrainPoints, 'affine');

%%
% Display the matching point pairs with the outliers removed
figure;
showMatchedFeatures(tumorImage, brainImage, inliertumorPoints, ...
    inlierbrainPoints, 'montage');
title('Matched Points (Inliers Only)');

%% 
% Get the bounding polygon of the reference image.
boxPolygon = [1, 1;...                           % top-left
        size(tumorImage, 2), 1;...                 % top-right
        size(tumorImage, 2), size(tumorImage, 1);... % bottom-right
        1, size(tumorImage, 1);...                 % bottom-left
        1, 1];                   % top-left again to close the polygon

%%
% Transform the polygon into the coordinate system of the target image.
% The transformed polygon indicates the location of the object in the
% scene.
newBoxPolygon = transformPointsForward(tform, boxPolygon);    

%%
% Display the detected object.
figure;
imshow(brainImage);
hold on;
line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
title('Detected Box');

%%