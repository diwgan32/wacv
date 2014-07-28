function [printD, printPINVD] = construct_dict(SubjectData, Segments, params)

clear;
clc;

addpath C:\FaceRecognition_YiChen_ECCV12\tools
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ksvdbox13
addpath C:\FaceRecognition_YiChen_ECCV12\tools\ompbox10

K = params(1);
G = cell(K);
    
    for j=1:K,
        for k=1:size(Segments, 2),
         if Segments(j, k) ~= -1,
                G{i, j} = [G{j} SubjectData(:, (Segments(j, k))+1)];
         end
        end
        
        if size(G{i,j},2) < 32
            
            G{j} = [G{j} gitter(G{j},20,20,32)];
          
        end
    end


        
        
img_gallery = [];
index_gallery = [];
runidx = 1;

    for j = 1:K
        img_gallery = [img_gallery G{j}];
        index_gallery = [index_gallery repmat(runidx,1,size(G{j},2))];
        runidx = runidx + 1;
    end


clear G;

dictsize = params(2)
iternum = params(2)

fprintf('Start training dictionary..\n');

Dict = mp_train(img_gallery, index_gallery, dictsize, iternum);

Dict_prime.D = []; % sequence level dictionary
Dict_prime.pinvD = []; % sequence level dictionary

    for j = 1:K
        Dict_prime.D = [Dict_prime.D Dict.D{j}];
    end
    %Dict_prime.D{i} = [Dict.D{3*(i-1)+1} Dict.D{3*(i-1)+2} Dict.D{3*(i-1)+3}];
    Dict_prime.pinvD = pinv(Dict_prime.D);



printD = [];
printPINVD = [];

    printD = Dict_prime.D;


    printPINVD = Dict_prime.pinvD;



end