clear;
clc;

addpath C:\FaceRecognition_YiChen_ECCV12\tools
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ksvdbox13
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ompbox10





subject_idx = 481;

K = 3;


dictsize = 5;
iternum = 20;

G = cell(subject_idx, K);
for i=1:subject_idx,
    G = cell(subject_idx, K);

    var = strcat('Subjects1\',int2str(i-1),'-*********.mat');
    d = dir(var);
    var = strcat('Subjects1\', d.name);
    if strcmp(var, 'Subjects1\') == 1,
        continue;
    end
    
    load (var);
    
    var = strcat('Segments1\',int2str(i-1),'-*********.mat');
    d = dir(var);
    var = strcat('Segments1\',d.name);
    load(var);
    G = cell(subject_idx, K);
    img_gallery = [];
    index_gallery = [];
    runidx = 1;
    
    for j=1:K,
        for k=1:size(Segments, 2),
            if Segments(j, k) ~= -1,
                if Segments(j, k)+1 > size(SubjectData, 2),
                    G{i, j} = [G{i, j} SubjectData(:, size(SubjectData, 2))];
                else
                    G{i, j} = [G{i, j} SubjectData(:, (Segments(j, k))+1)];
                end
           
            end
        end
        
        if size(G{i,j},2) < 32
            
            G{i,j} = [G{i,j} gitter(G{i,j},20,20,32)];
            
        end
    end
    
    for j=1:K,
        img_gallery = [img_gallery G{i,j}];
        index_gallery = [index_gallery repmat(runidx,1,size(G{i,j},2))];
        runidx = runidx + 1;
    end
    
    clear G;
    
    Dict = mp_train(img_gallery, index_gallery, dictsize, iternum);
    printD = [];
    printPINVD = [];
    
    for j = 1:K
        printD = [printD Dict.D{j}];
    end
    printPINVD = pinv(printD);
    
    k=strfind(var, '.');
    
    filename = strcat('Dictionaries\', var(1, strfind(var, '-')+1:k-1), '.bin');
    fid = fopen(filename, 'w');
    fwrite(fid, printD, 'double');
    fclose(fid);
    filename = strcat('InverseDictionaries\', var(1, strfind(var, '-')+1:k-1), '.bin');
    fid = fopen(filename, 'w');
    fwrite(fid, printPINVD, 'double');
    fclose(fid);
end






clear G;

