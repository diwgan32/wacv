clear;
clc;

addpath C:\FaceRecognition_YiChen_ECCV12\tools
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ksvdbox13
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ompbox10

numSubjects = 15;
K = 3;
offsets = [];
img_gallery = [];
for i=0:numSubjects-1,
    var = strcat('Subjects\',int2str(i),'.mat');
    load (var);
    var = strcat('Segments\',int2str(i),'.mat');
    load (var);
    
    for j=1:K,
        for k=1:size(Segments, 2),
            if Segments(j, k) ~= -1,
                img_gallery = [img_gallery SubjectData(:, (Segments(j, k))+1)];
            end
        end
    end
    offsets = [offsets size(SubjectData, 2)];
end
subject_idx = 97;

K = 3;

index_gallery = [];

runidx = 1;
for i=1:numSubjects,
index_gallery = [index_gallery repmat(runidx,1,offsets(i))];
runidx = runidx+1;
end

dictsize = 3;
iternum = 100;

fprintf('Start training dictionary..\n');

Dict = mp_train(img_gallery, index_gallery, dictsize, iternum);

fid = fopen('Dictionaries\dict.bin', 'w');
printD = [];
for i=1:numSubjects,
    printD = [printD Dict.D{i}];
end

fwrite(fid, printD, 'double');
fclose(fid);
printPINVD = [];
for i=1:numSubjects,
    printPINVD = [printPINVD Dict.pinvD{i}];
end

fid = fopen('Dictionaries\pinvDict.bin', 'w');
fwrite(fid, printPINVD, 'double');
fclose(fid);