clear;
load Subjects\7.mat;
img = [];
bestsgt{1} = [3 4 5 6 7 8 9 10 11 12 13 14 ];
bestsgt{2} = [15 16 17 23 24 25 26 27 28 29 30 31 32 33 34 35 ];
bestsgt{3} = [1 2 18 19 20 21 22 ];


for j=1:numel(bestsgt),
    img = [];
for i=1:size(bestsgt{j}, 2),
    
    img = [img vec2mat( ImgData1(:,bestsgt{j}(i)), 20)];
end
figure, imshow(img);
end