clear;
clc;

addpath C:\FaceRecognition_YiChen_ECCV12\tools
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ksvdbox13
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ompbox10

numSubjects = 15;
K = 3;
offsets = [];
img_gallery = [];
count = 0;
D = {};
PINVD = {};


dictsize = 3;
iternum = 100;
prevsize = 0;
temp_gallery = [];

ImgData = {};

for i=1:numSubjects,
    var = strcat('Subjects1\',int2str(i),'-*****.mat');
    d = dir(var);
    var = strcat('Subjects1\', d.name);
    load (var);
    
    var = strcat('Segments\',int2str(i),'-*****.mat');
    d = dir(var);
    var = strcat('Segments\',d.name);
    load (var);
    ImgData{i} = SubjectData;
   
end



subject_idx = 97;

K = 3;



save('UTD-data1.mat', 'ImgData');
runidx = 1;
