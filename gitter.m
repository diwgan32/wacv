function Gout = gitter(Gin,rsz,csz,AugSz)

S = [];
dx = 3;%2;
dy = 3;%2;
for i = 1:min([4 size(Gin,2)]) %size(Gin,2)
    Iin = reshape(Gin(:,i),rsz,csz);
    
    IinL = zeros(rsz+2*dy,csz+2*dx);
    IinL(dy+1:end-dy,dx+1:end-dx) = Iin;
    IinL(dy+1:end-dy,1:dx) = repmat(Iin(:,1),1,dx); % copy left strip
    IinL(dy+1:end-dy,csz+dx+1:csz+2*dx) = repmat(Iin(:,end),1,dx);  % copy right strip
    IinL(1:dy,dx+1:end-dx) = repmat(Iin(1,:),dy,1); % copy top strip
    IinL(rsz+dy+1:rsz+2*dy,dx+1:end-dx) = repmat(Iin(end,:),dy,1); % copy bottom strip
    IinL(1:dy,1:dx) = repmat(Iin(1,1),dy,dx); % copy top-left corner
    IinL(1:dy,csz+dx+1:csz+2*dx) = repmat(Iin(1,end),dy,dx); % copy top-right corner
    IinL(rsz+dy+1:rsz+2*dy,1:dx) = repmat(Iin(end,1),dy,dx); % copy bottom-left corner
    IinL(rsz+dy+1:rsz+2*dy,csz+dx+1:csz+2*dx) = repmat(Iin(end,end),dy,dx); % copy bottom-right corner
    
    % create additional 24 pixel-shifted images
    for j = -dx:dx 
        for k = -dy:dy
            if j == 0 && k == 0
                continue;
            end
            r = dy+1+k;
            c = dx+1+j;
            S = [S reshape(IinL(r:r+rsz-1,c:c+csz-1),rsz*csz,1)];
        end
    end     
end

R = [];
rot = [-12 -9 -6 -3 3 6 9 12]; % create additional 8 rotated images: -12 ~ 12 degrees
for i = 1:min([3 size(Gin,2)]) %size(Gin,2)
    Iin = reshape(Gin(:,i),rsz,csz);
    for r = 1:size(rot,2)
        Ir = imrotate(im2uint8(Iin),rot(1,r),'crop');
        R = [R reshape(im2double(Ir),rsz*csz,1)];
    end
end

SR = [S R];
idx = ceil(rand(1,AugSz)*size(SR,2));
Gout = SR(:,idx);