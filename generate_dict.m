clear;
clc;

addpath C:\FaceRecognition_YiChen_ECCV12\tools
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ksvdbox13
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ompbox10

numSubjects = 10;
K = 3;
offsets = [];
img_gallery = [];
count = 0;
D = {};
PINVD = {};


dictsize = 3;
iternum = 30;
prevsize = 0;
temp_gallery = [];


for i=1:numSubjects,
    var = strcat('Subjects\',int2str(i),'-*****.mat');
    d = dir(var);
    var = strcat('Subjects\', d.name);
    load (var);
    
    var = strcat('Segments\',int2str(i),'-*****.mat');
    d = dir(var);
    var = strcat('Segments\',d.name);
    load (var);
    
    for j=1:K,
        Dict = [];
        temp_gallery = [];
        index_gallery = [];
        for k=1:size(Segments, 2),
            if Segments(j, k) ~= -1,
                temp_gallery = [temp_gallery SubjectData(:, (Segments(j, k))+1)];
                count=count+1;
            end
            
            
            
        end
       
        if count < 32,
            temp_gallery = [temp_gallery gitter(temp_gallery, 20, 20, 32)];
        end
        count = 0;
        index_gallery = [index_gallery repmat(1,1,size(temp_gallery, 2))];
        Dict = mp_train(temp_gallery, index_gallery, dictsize, iternum);
        if j==1,
            D{i} = Dict.D{1};
            
        else
            D{i} = [D{i} Dict.D{1}];
            
        end
    end
end



subject_idx = 97;

K = 3;




runidx = 1;
for i=1:numSubjects,
    
    runidx = runidx+1;
end

printD = [];
printPINVD = [];

for i=1:numSubjects,
    printD = [printD D{i}];
end

for i=1:numSubjects,
    printPINVD = [printPINVD pinv(printD(:, (i-1)*(9)+1:i*(9)))];
end
fid = fopen('Dictionaries\dict.bin', 'w');
fwrite(fid, printD, 'double');
fclose(fid);

fid = fopen('Dictionaries\pinvDict.bin', 'w');
fwrite(fid, printPINVD, 'double');
fclose(fid);