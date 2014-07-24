clear;
clc;

addpath C:\FaceRecognition_YiChen_ECCV12\tools
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ksvdbox13
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ompbox10





subject_idx = 10;

K = 3;


G = cell(subject_idx, K);
for i=1:subject_idx,
    var = strcat('Subjects\',int2str(i),'-*****.mat');
    d = dir(var);
    var = strcat('Subjects\', d.name);
    load (var);
    
    var = strcat('Segments\',int2str(i),'-*****.mat');
    d = dir(var);
    var = strcat('Segments\',d.name);
    load (var);
    
    
    for j=1:K,
        for k=1:size(Segments, 2),
         if Segments(j, k) ~= -1,
                G{i, j} = [G{i, j} SubjectData(:, (Segments(j, k))+1)];
         end
        end
        
        if size(G{i,j},2) < 32
            
            G{i,j} = [G{i,j} gitter(G{i,j},20,20,32)];
          
        end
    end
end

        
        
img_gallery = [];
index_gallery = [];
runidx = 1;
for i = 1:subject_idx
    for j = 1:K
        img_gallery = [img_gallery G{i,j}];
        index_gallery = [index_gallery repmat(runidx,1,size(G{i,j},2))];
        runidx = runidx + 1;
    end
end

clear G;

dictsize = 3;
iternum = 20;

fprintf('Start training dictionary..\n');

Dict = mp_train(img_gallery, index_gallery, dictsize, iternum);

Dict_prime.D = cell(subject_idx,1); % sequence level dictionary
Dict_prime.pinvD = cell(subject_idx,1); % sequence level dictionary
for i = 1:subject_idx
    for j = 1:K
        Dict_prime.D{i} = [Dict_prime.D{i} Dict.D{K*(i-1)+j}];
    end
    %Dict_prime.D{i} = [Dict.D{3*(i-1)+1} Dict.D{3*(i-1)+2} Dict.D{3*(i-1)+3}];
    Dict_prime.pinvD{i} = pinv(Dict_prime.D{i});
end


printD = [];
printPINVD = [];

for i=1:subject_idx,
    printD = [printD Dict_prime.D{i}];
end

for i=1:subject_idx,
    printPINVD = [printPINVD Dict_prime.pinvD{i}];
end
fid = fopen('Dictionaries\dict.bin', 'w');
fwrite(fid, printD, 'double');
fclose(fid);

fid = fopen('Dictionaries\pinvDict.bin', 'w');
fwrite(fid, printPINVD, 'double');
fclose(fid);