function [] = visVoxels(Volume)
%VISVOXELS Summary of this function goes here
%   Detailed explanation goes here

%% visualization 1

[X,Y,Z]=ind2sub(size(Volume),find(Volume(:)));
plot3(X,Y,Z,'.','markers',1);
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');
size(Volume)
xlim([0,size(Volume,1)]);
ylim([0,size(Volume,2)]);
zlim([0,size(Volume,3)]);
%% visualization 2
%{
figure(2);
careMask =  imdilate((Volume),ones(2,2,2));
figure(3);
plot3D(careMask,1,'timed', 0.1)
hold on;%plot3D(Volume,1,'b','*')
viewA =33:5:176;
for i =1:length(viewA)
    view(viewA(i),28);
    pause(0.2)
end
%}

%% visualization 3
%{
figure(4);
for i=1:size(Volume,1)
    imagesc(squeeze(Volume(i,:,:)));
    axis equal;
    axis tight;
    axis off
    title(i);
    pause(0.2);
end

for i=1:size(Volume,2)
    imagesc(squeeze(Volume(:,i,:)));
    axis equal;
    axis tight;
    axis off
    title(i);
    pause(0.2);
end

for i=1:size(Volume,3)
    imagesc(squeeze(Volume(:,:,i)),[-0.5,0.7]);
    axis equal;
    colorbar;
    axis tight;
    axis off
    title(i);
    pause(0.2);
end
%}

end