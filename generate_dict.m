clear;
clc;

addpath C:\FaceRecognition_YiChen_ECCV12\tools
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ksvdbox13
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ompbox10

numSubjects = 5;
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
fwrite(fid, [Dict.D{1} Dict.D{2} Dict.D{3} Dict.D{4} Dict.D{5}], 'double');
fclose(fid);
fid = fopen('Dictionaries\pinvDict.bin', 'w');
fwrite(fid, [Dict.pinvD{1} Dict.pinvD{2} Dict.pinvD{3} Dict.pinvD{4} Dict.pinvD{5}], 'double');
fclose(fid);