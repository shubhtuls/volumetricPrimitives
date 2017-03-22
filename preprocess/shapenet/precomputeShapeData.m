function [] = precomputeShapeData()
globals;
shapenetDir = shapenetDir;
cachedir = cachedir;
params = params;
vis = 0;
%loop over sysnets
synsetNames = {'03001627','02691156'}; %chair, aeroplane
gridSize=32;
numSamples = 10000;

for s = 1:length(synsetNames)
    fprintf('synset : %d/%d\n\n',s,length(synsetNames));
    synset = synsetNames{s};
    modelsDir = fullfile(shapenetDir,synset);
    tsdfDir = fullfile(cachedir,'shapenet','chamferData',synset);
    mkdirOptional(tsdfDir);
    modelNames = getFileNamesFromDirectory(modelsDir,'types',{''});
    modelNames = modelNames(3:end); %fist two are '.' and '..'
    
    nModels = length(modelNames);
    %nModels = 5;%for debugging we'll use few instances only
    pBar = TimedProgressBar( nModels, 30, 'Time Remaining : ', ' Percentage Completion ', 'Tsdf Extraction Completed.');
    
    for i = 1:nModels
        
        %% Check if computation is really needed
        modelFile = getFileNamesFromDirectory(fullfile(modelsDir,modelNames{i}),'types',{'.obj'});
        tsdfFile = fullfile(tsdfDir,[modelNames{i} '.mat']);
        if(exist(tsdfFile,'file'))
            continue;
        end
        if(isempty(modelFile))
            continue;
        end
        
        %% Read model
        modelFile = fullfile(modelsDir,modelNames{i},modelFile{1});
        %try
        [Shape] = parseObjMesh(modelFile);
        
        surfaceSamples = uniform_sampling(Shape, numSamples);
        surfaceSamples = surfaceSamples';
        %ignore bad meshes
        if(isempty(Shape.vertexPoss) || isempty(Shape.faceVIds))
            continue;
        end
        
        %% Compute Voxels
        faces = Shape.faceVIds';
        vertices = Shape.vertexPoss';
        stepRange = -0.5+1/(2*gridSize):1/gridSize:0.5-1/(2*gridSize);
        [Xp,Yp,Zp] = ndgrid(stepRange, stepRange, stepRange);
        queryPoints = [Xp(:), Yp(:), Zp(:)];

        
        FV = struct();
        FV.faces = faces;
        FV.vertices = (gridSize)*(vertices+0.5) + 0.5;

        Volume=polygon2voxel(FV,gridSize,'none',false);
        
        [tsdfPoints,~,closestPoints] = point_mesh_squared_distance(queryPoints,vertices,faces);
        tsdfPoints = sqrt(tsdfPoints);
        tsdfGrid = reshape(tsdfPoints,size(Xp));
        tsdfGrid = abs(tsdfGrid).*(1-2*Volume);
        closestPointsGrid = reshape(closestPoints,[size(Xp),3]);
        
        savefunc(tsdfFile, tsdfGrid, Volume, closestPointsGrid, surfaceSamples, vertices, faces);
        pBar.progress();
    end
    pBar.stop();
end

end

function savefunc(tsdfFile, tsdf, Volume, closestPoints, surfaceSamples, vertices, faces)
save(tsdfFile,'tsdf','Volume','closestPoints', 'surfaceSamples','vertices','faces');
end

function [samples] = uniform_sampling(Shape, numSamples)
% Perform uniform sampling of a mesh, with the numSamples samples
% samples: a matrix of dimension 3 x numSamples

t = sort(rand(1, numSamples));
faceAreas = tri_mesh_face_area(Shape);
numFaces = length(faceAreas);
for i = 2:numFaces
    faceAreas(i) = faceAreas(i-1) + faceAreas(i);
end
samples = zeros(3, numSamples);

paras = rand(2, numSamples);

faceId = 1;
for sId = 1:numSamples
    while t(sId) > faceAreas(faceId)
        faceId = faceId + 1;
    end
    faceId = min(faceId, numFaces);
    p1 = Shape.vertexPoss(:, Shape.faceVIds(1, faceId));
    p2 = Shape.vertexPoss(:, Shape.faceVIds(2, faceId));
    p3 = Shape.vertexPoss(:, Shape.faceVIds(3, faceId));
    
    r1 = paras(1, sId);
    r2 = paras(2, sId);
    t1 = 1-sqrt(r1);
    t2 = sqrt(r1)*(1-r2);
    t3 = sqrt(r1)*r2;
    samples(:, sId) = t1*p1 + t2*p2 + t3*p3;
end

end

function [faceAreas] = tri_mesh_face_area(Shape)
%
p1 = Shape.vertexPoss(:, Shape.faceVIds(1,:));
p2 = Shape.vertexPoss(:, Shape.faceVIds(2,:));
p3 = Shape.vertexPoss(:, Shape.faceVIds(3,:));

e12 = p1 - p2;
e23 = p2 - p3;
e31 = p3 - p1;

a2 = sum(e12.*e12);
b2 = sum(e23.*e23);
c2 = sum(e31.*e31);

areas = 2*(a2.*(b2+c2)+b2.*c2)-a2.*a2-b2.*b2-c2.*c2;
areas = sqrt(max(0, areas))/4;
faceAreas = areas/sum(areas);
end
