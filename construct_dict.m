function construct_dict(dictsize, K, dims)


addpath C:\FaceRecognition_YiChen_ECCV12\tools
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ksvdbox13
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ompbox10

iternum = 20;
subject_idx = 481;
var = dir('Subjects\*.mat');

G = cell(subject_idx, K);
for i=1:length(var),
    G = cell(subject_idx, K);
    mname = strcat('Subjects\', var(i).name);
    load (mname);
    mname = strcat('Segments\', var(i).name);
    load(mname);
    G = cell(subject_idx, K);
    img_gallery = [];
    index_gallery = [];
    runidx = 1;
    mname = var(i).name;
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
            
            G{i,j} = [G{i,j} gitter(G{i,j},sqrt(dims), sqrt(dims),32)];
            
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
    
    k=strfind(mname, '.');
    
  
    if isempty(strfind(mname, '-')),
    filename = strcat('Dictionaries\', mname(1, 1:9), '.bin');
    fid = fopen(filename, 'w');
    fwrite(fid, printD, 'double');
    fclose(fid);
    filename = strcat('InverseDictionaries\', mname(1, 1:9), '.bin');
    fid = fopen(filename, 'w');
    fwrite(fid, printPINVD, 'double');
    fclose(fid);
    
    else
         filename = strcat('Dictionaries\', mname(1, strfind(mname, '-')+1:k-1), '.bin');
    fid = fopen(filename, 'w');
    fwrite(fid, printD, 'double');
    fclose(fid);
    filename = strcat('InverseDictionaries\', mname(1, strfind(mname, '-')+1:k-1), '.bin');
    fid = fopen(filename, 'w');
    fwrite(fid, printPINVD, 'double');
    fclose(fid);
    end
        
    
end






clear G;

end