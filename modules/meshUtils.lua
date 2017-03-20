require 'pl'
local transformer = dofile('../modules/transformer.lua')
local M = {}
local colormap = torch.Tensor({{0.000000, 0.000000, 0.515625},{0.000000, 0.000000, 0.531250},{0.000000, 0.000000, 0.546875},{0.000000, 0.000000, 0.562500},{0.000000, 0.000000, 0.578125},{0.000000, 0.000000, 0.593750},{0.000000, 0.000000, 0.609375},{0.000000, 0.000000, 0.625000},{0.000000, 0.000000, 0.640625},{0.000000, 0.000000, 0.656250},{0.000000, 0.000000, 0.671875},{0.000000, 0.000000, 0.687500},{0.000000, 0.000000, 0.703125},{0.000000, 0.000000, 0.718750},{0.000000, 0.000000, 0.734375},{0.000000, 0.000000, 0.750000},{0.000000, 0.000000, 0.765625},{0.000000, 0.000000, 0.781250},{0.000000, 0.000000, 0.796875},{0.000000, 0.000000, 0.812500},{0.000000, 0.000000, 0.828125},{0.000000, 0.000000, 0.843750},{0.000000, 0.000000, 0.859375},{0.000000, 0.000000, 0.875000},{0.000000, 0.000000, 0.890625},{0.000000, 0.000000, 0.906250},{0.000000, 0.000000, 0.921875},{0.000000, 0.000000, 0.937500},{0.000000, 0.000000, 0.953125},{0.000000, 0.000000, 0.968750},{0.000000, 0.000000, 0.984375},{0.000000, 0.000000, 1.000000},{0.000000, 0.015625, 1.000000},{0.000000, 0.031250, 1.000000},{0.000000, 0.046875, 1.000000},{0.000000, 0.062500, 1.000000},{0.000000, 0.078125, 1.000000},{0.000000, 0.093750, 1.000000},{0.000000, 0.109375, 1.000000},{0.000000, 0.125000, 1.000000},{0.000000, 0.140625, 1.000000},{0.000000, 0.156250, 1.000000},{0.000000, 0.171875, 1.000000},{0.000000, 0.187500, 1.000000},{0.000000, 0.203125, 1.000000},{0.000000, 0.218750, 1.000000},{0.000000, 0.234375, 1.000000},{0.000000, 0.250000, 1.000000},{0.000000, 0.265625, 1.000000},{0.000000, 0.281250, 1.000000},{0.000000, 0.296875, 1.000000},{0.000000, 0.312500, 1.000000},{0.000000, 0.328125, 1.000000},{0.000000, 0.343750, 1.000000},{0.000000, 0.359375, 1.000000},{0.000000, 0.375000, 1.000000},{0.000000, 0.390625, 1.000000},{0.000000, 0.406250, 1.000000},{0.000000, 0.421875, 1.000000},{0.000000, 0.437500, 1.000000},{0.000000, 0.453125, 1.000000},{0.000000, 0.468750, 1.000000},{0.000000, 0.484375, 1.000000},{0.000000, 0.500000, 1.000000},{0.000000, 0.515625, 1.000000},{0.000000, 0.531250, 1.000000},{0.000000, 0.546875, 1.000000},{0.000000, 0.562500, 1.000000},{0.000000, 0.578125, 1.000000},{0.000000, 0.593750, 1.000000},{0.000000, 0.609375, 1.000000},{0.000000, 0.625000, 1.000000},{0.000000, 0.640625, 1.000000},{0.000000, 0.656250, 1.000000},{0.000000, 0.671875, 1.000000},{0.000000, 0.687500, 1.000000},{0.000000, 0.703125, 1.000000},{0.000000, 0.718750, 1.000000},{0.000000, 0.734375, 1.000000},{0.000000, 0.750000, 1.000000},{0.000000, 0.765625, 1.000000},{0.000000, 0.781250, 1.000000},{0.000000, 0.796875, 1.000000},{0.000000, 0.812500, 1.000000},{0.000000, 0.828125, 1.000000},{0.000000, 0.843750, 1.000000},{0.000000, 0.859375, 1.000000},{0.000000, 0.875000, 1.000000},{0.000000, 0.890625, 1.000000},{0.000000, 0.906250, 1.000000},{0.000000, 0.921875, 1.000000},{0.000000, 0.937500, 1.000000},{0.000000, 0.953125, 1.000000},{0.000000, 0.968750, 1.000000},{0.000000, 0.984375, 1.000000},{0.000000, 1.000000, 1.000000},{0.015625, 1.000000, 0.984375},{0.031250, 1.000000, 0.968750},{0.046875, 1.000000, 0.953125},{0.062500, 1.000000, 0.937500},{0.078125, 1.000000, 0.921875},{0.093750, 1.000000, 0.906250},{0.109375, 1.000000, 0.890625},{0.125000, 1.000000, 0.875000},{0.140625, 1.000000, 0.859375},{0.156250, 1.000000, 0.843750},{0.171875, 1.000000, 0.828125},{0.187500, 1.000000, 0.812500},{0.203125, 1.000000, 0.796875},{0.218750, 1.000000, 0.781250},{0.234375, 1.000000, 0.765625},{0.250000, 1.000000, 0.750000},{0.265625, 1.000000, 0.734375},{0.281250, 1.000000, 0.718750},{0.296875, 1.000000, 0.703125},{0.312500, 1.000000, 0.687500},{0.328125, 1.000000, 0.671875},{0.343750, 1.000000, 0.656250},{0.359375, 1.000000, 0.640625},{0.375000, 1.000000, 0.625000},{0.390625, 1.000000, 0.609375},{0.406250, 1.000000, 0.593750},{0.421875, 1.000000, 0.578125},{0.437500, 1.000000, 0.562500},{0.453125, 1.000000, 0.546875},{0.468750, 1.000000, 0.531250},{0.484375, 1.000000, 0.515625},{0.500000, 1.000000, 0.500000},{0.515625, 1.000000, 0.484375},{0.531250, 1.000000, 0.468750},{0.546875, 1.000000, 0.453125},{0.562500, 1.000000, 0.437500},{0.578125, 1.000000, 0.421875},{0.593750, 1.000000, 0.406250},{0.609375, 1.000000, 0.390625},{0.625000, 1.000000, 0.375000},{0.640625, 1.000000, 0.359375},{0.656250, 1.000000, 0.343750},{0.671875, 1.000000, 0.328125},{0.687500, 1.000000, 0.312500},{0.703125, 1.000000, 0.296875},{0.718750, 1.000000, 0.281250},{0.734375, 1.000000, 0.265625},{0.750000, 1.000000, 0.250000},{0.765625, 1.000000, 0.234375},{0.781250, 1.000000, 0.218750},{0.796875, 1.000000, 0.203125},{0.812500, 1.000000, 0.187500},{0.828125, 1.000000, 0.171875},{0.843750, 1.000000, 0.156250},{0.859375, 1.000000, 0.140625},{0.875000, 1.000000, 0.125000},{0.890625, 1.000000, 0.109375},{0.906250, 1.000000, 0.093750},{0.921875, 1.000000, 0.078125},{0.937500, 1.000000, 0.062500},{0.953125, 1.000000, 0.046875},{0.968750, 1.000000, 0.031250},{0.984375, 1.000000, 0.015625},{1.000000, 1.000000, 0.000000},{1.000000, 0.984375, 0.000000},{1.000000, 0.968750, 0.000000},{1.000000, 0.953125, 0.000000},{1.000000, 0.937500, 0.000000},{1.000000, 0.921875, 0.000000},{1.000000, 0.906250, 0.000000},{1.000000, 0.890625, 0.000000},{1.000000, 0.875000, 0.000000},{1.000000, 0.859375, 0.000000},{1.000000, 0.843750, 0.000000},{1.000000, 0.828125, 0.000000},{1.000000, 0.812500, 0.000000},{1.000000, 0.796875, 0.000000},{1.000000, 0.781250, 0.000000},{1.000000, 0.765625, 0.000000},{1.000000, 0.750000, 0.000000},{1.000000, 0.734375, 0.000000},{1.000000, 0.718750, 0.000000},{1.000000, 0.703125, 0.000000},{1.000000, 0.687500, 0.000000},{1.000000, 0.671875, 0.000000},{1.000000, 0.656250, 0.000000},{1.000000, 0.640625, 0.000000},{1.000000, 0.625000, 0.000000},{1.000000, 0.609375, 0.000000},{1.000000, 0.593750, 0.000000},{1.000000, 0.578125, 0.000000},{1.000000, 0.562500, 0.000000},{1.000000, 0.546875, 0.000000},{1.000000, 0.531250, 0.000000},{1.000000, 0.515625, 0.000000},{1.000000, 0.500000, 0.000000},{1.000000, 0.484375, 0.000000},{1.000000, 0.468750, 0.000000},{1.000000, 0.453125, 0.000000},{1.000000, 0.437500, 0.000000},{1.000000, 0.421875, 0.000000},{1.000000, 0.406250, 0.000000},{1.000000, 0.390625, 0.000000},{1.000000, 0.375000, 0.000000},{1.000000, 0.359375, 0.000000},{1.000000, 0.343750, 0.000000},{1.000000, 0.328125, 0.000000},{1.000000, 0.312500, 0.000000},{1.000000, 0.296875, 0.000000},{1.000000, 0.281250, 0.000000},{1.000000, 0.265625, 0.000000},{1.000000, 0.250000, 0.000000},{1.000000, 0.234375, 0.000000},{1.000000, 0.218750, 0.000000},{1.000000, 0.203125, 0.000000},{1.000000, 0.187500, 0.000000},{1.000000, 0.171875, 0.000000},{1.000000, 0.156250, 0.000000},{1.000000, 0.140625, 0.000000},{1.000000, 0.125000, 0.000000},{1.000000, 0.109375, 0.000000},{1.000000, 0.093750, 0.000000},{1.000000, 0.078125, 0.000000},{1.000000, 0.062500, 0.000000},{1.000000, 0.046875, 0.000000},{1.000000, 0.031250, 0.000000},{1.000000, 0.015625, 0.000000},{1.000000, 0.000000, 0.000000},{0.984375, 0.000000, 0.000000},{0.968750, 0.000000, 0.000000},{0.953125, 0.000000, 0.000000},{0.937500, 0.000000, 0.000000},{0.921875, 0.000000, 0.000000},{0.906250, 0.000000, 0.000000},{0.890625, 0.000000, 0.000000},{0.875000, 0.000000, 0.000000},{0.859375, 0.000000, 0.000000},{0.843750, 0.000000, 0.000000},{0.828125, 0.000000, 0.000000},{0.812500, 0.000000, 0.000000},{0.796875, 0.000000, 0.000000},{0.781250, 0.000000, 0.000000},{0.765625, 0.000000, 0.000000},{0.750000, 0.000000, 0.000000},{0.734375, 0.000000, 0.000000},{0.718750, 0.000000, 0.000000},{0.703125, 0.000000, 0.000000},{0.687500, 0.000000, 0.000000},{0.671875, 0.000000, 0.000000},{0.656250, 0.000000, 0.000000},{0.640625, 0.000000, 0.000000},{0.625000, 0.000000, 0.000000},{0.609375, 0.000000, 0.000000},{0.593750, 0.000000, 0.000000},{0.578125, 0.000000, 0.000000},{0.562500, 0.000000, 0.000000},{0.546875, 0.000000, 0.000000},{0.531250, 0.000000, 0.000000},{0.515625, 0.000000, 0.000000},{0.500000, 0.000000, 0.000000}})

local cubeV = torch.Tensor({{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}})
cubeV = 2*cubeV - 1

local cubeF = torch.Tensor({{1,  7,  5 }, {1,  3,  7 }, {1,  4,  3 }, {1,  2,  4 }, {3,  8,  7 }, {3,  4,  8 }, {5,  7,  8 }, {5,  8,  6 }, {1,  5,  6 }, {1,  6,  2 }, {2,  6,  8 }, {2,  8,  4}})

local function triliniearCoeff(p, cubPoints)
    local p000 = cubPoints[1]
    local p001 = cubPoints[2]
    local p010 = cubPoints[3]
    local p100 = cubPoints[5]
    -- coeff = alpha_{000, 001, 010, 011, 100, 101, 110, 111}
    local a_x = torch.dot(p - p000, p100 - p000)/torch.dot(p100 - p000, p100 - p000)
    local a_y = torch.dot(p - p000, p010 - p000)/torch.dot(p010 - p000, p010 - p000)
    local a_z = torch.dot(p - p000, p001 - p000)/torch.dot(p001 - p000, p001 - p000)
    
    local a000 = (1-a_x)*(1-a_y)*(1-a_z)
    local a001 = (1-a_x)*(1-a_y)*(a_z)
    local a010 = (1-a_x)*(a_y)*(1-a_z)
    local a011 = (1-a_x)*(a_y)*(a_z)

    local a100 = (a_x)*(1-a_y)*(1-a_z)
    local a101 = (a_x)*(1-a_y)*(a_z)
    local a110 = (a_x)*(a_y)*(1-a_z)
    local a111 = (a_x)*(a_y)*(a_z)
    local coeff = torch.Tensor({a000, a001, a010, a011, a100, a101, a110, a111})
    
    return coeff
    
end

local function cylMesh(shape)
    local radius = shape[1]
    local height = shape[2]
    local nCircleVerts = 20
    local nVerts = 2*nCircleVerts + 2 --two centre vertices
    local nTri = 4*nCircleVerts

    local verts = torch.Tensor(nVerts,3):fill(0)
    local faces = torch.Tensor(nTri,3):fill(0)

    -- vertices
    verts[1][3] = height
    verts[2][3] = -height
    local theta = 2*math.pi/nCircleVerts

    for v = 1,nCircleVerts do
        verts[2+v][3] = height
        verts[2+v][1] = math.cos(v*theta)*radius
        verts[2+v][2] = math.sin(v*theta)*radius
    end

    for v = 1+nCircleVerts,2*nCircleVerts do
        verts[2+v][3] = -height
        verts[2+v][1] = math.cos(v*theta + theta/2)*radius
        verts[2+v][2] = math.sin(v*theta + theta/2)*radius
    end

    -- faces
    local nf = 1
    for v=1,nCircleVerts do
        faces[nf][1] = 1
        faces[nf][2] = 2 + v
        faces[nf][3] = 2 + ((v==nCircleVerts) and 1 or v+1)
        nf = nf+1
    end

    for v=1,nCircleVerts do
        faces[nf][1] = 2
        faces[nf][3] = 2 + nCircleVerts + v
        faces[nf][2] = 2 + nCircleVerts + ((v==nCircleVerts) and 1 or v+1)
        nf = nf+1
    end

    for v=1,nCircleVerts do
        faces[nf][1] = 2 + v
        faces[nf][2] = 2 + nCircleVerts + v
        faces[nf][3] = 2 + ((v==nCircleVerts) and 1 or v+1)
        nf = nf+1
    end

    for v=1,nCircleVerts do
        faces[nf][1] = 2 + nCircleVerts + ((v==1) and nCircleVerts or (v-1))
        faces[nf][3] = 2 + v
        faces[nf][2] = 2 + nCircleVerts + v
        nf = nf+1
    end
    return verts, faces
end

local function cuboidMesh(shape)
    local verts = cubeV:clone()
    for d=1,3 do
        verts:narrow(2,d,1):mul(shape[d])
    end
    return verts, cubeF:clone()
end

local function shapeMesh(shape, primTypes)
    local shapeType, shapeParams
    if(#primTypes == 1) then
        shapeType = primTypes[1]
        shapeParams = shape
    else
        local maxProb=0
        local nz = 0
        local primStarts = {}
        local primSizes = {}
        for ix = 1,#primTypes do
            local primSize = 0
            if(primTypes[ix] == 'Cu') then primSize = 3 end
            if(primTypes[ix] == 'Cy') then primSize = 2 end
            if(primTypes[ix] == 'Nu') then primSize = 1 end
            primSizes[ix] = primSize
            primStarts[ix] = nz+1
            nz = nz+primSize
        end

        for ix=1,#primTypes do
            if(shape[nz+ix] > maxProb) then
                maxProb = shape[nz+ix]
                shapeParams = shape:narrow(1,primStarts[ix],primSizes[ix])
                shapeType = primTypes[ix]
            end
        end
    end
        
    if(shapeType=='Cu') then
        return cuboidMesh(shapeParams)
    end
    if(shapeType=='Cy') then
        return cylMesh(shapeParams)
    end
    if(shapeType=='Nu') then
        return torch.Tensor(), torch.Tensor()
    end
    
    assert(#primTypes > 1)
    
    
end

local function shapeVolume(shape, primTypes)
    local shapeType, shapeParams
    if(#primTypes == 1) then
        shapeType = primTypes[1]
        shapeParams = shape
    else
        local maxProb=0
        local nz = 0
        local primStarts = {}
        local primSizes = {}
        for ix = 1,#primTypes do
            local primSize = 0
            if(primTypes[ix] == 'Cu') then primSize = 3 end
            if(primTypes[ix] == 'Cy') then primSize = 2 end
            if(primTypes[ix] == 'Nu') then primSize = 1 end
            primSizes[ix] = primSize
            primStarts[ix] = nz+1
            nz = nz+primSize
        end

        for ix=1,#primTypes do
            if(shape[nz+ix] > maxProb) then
                maxProb = shape[nz+ix]
                shapeParams = shape:narrow(1,primStarts[ix],primSizes[ix])
                shapeType = primTypes[ix]
            end
        end
    end
        
    if(shapeType=='Cu') then
        return 8*shapeParams[1]*shapeParams[2]*shapeParams[3]
    end
    if(shapeType=='Cy') then
        return 2*math.pi*shapeParams[1]*shapeParams[1]*shapeParams[2]
    end
    if(shapeType=='Nu') then
        return 0
    end
    
    assert(#primTypes > 1)
    
end

local function partVertices(predParts, primTypes)
    local pv = {}
    local nParts = #predParts
    local primTypes = primTypes
    if(primTypes == nil) then primTypes = {'Cu'} end
    for p = 1,nParts do
        local shape, translation, quat = unpack(predParts[p])
        quat = quat:clone()
        quat[1] = -quat[1]
        local verts, faces = shapeMesh(shape, primTypes)
        if(verts:numel() > 0) then
            local nVerts = verts:size(1)  
            verts = verts:view(1,nVerts,3)
            translation = translation:reshape(1,3):repeatTensor(nVerts,1)
            quat = quat:reshape(1,4)
            local rotator = transformer.rotation(nVerts)
            verts = rotator:forward({verts, quat}) + translation
            verts = verts:squeeze()
        end
        pv[p] = verts
    end
    return pv
end

local function saveParts(predParts, outputFile, primTypes, partIndsSpecific)
    local primTypes = primTypes
    if(primTypes == nil) then primTypes = {'Cu'} end
    local mtlFile = outputFile:split('.obj')[1] .. '.mtl'
    local fout = io.open(outputFile, 'w')
    local foutMtl = io.open(mtlFile, 'w')
    local partIndsSpecific = partIndsSpecific or {}
    mtlFile = mtlFile:split('/')
    mtlFile = mtlFile[#mtlFile]    
    --print(outputFile)
    local partCmaps = {}
    local fCounter = 0
    local nParts = #predParts
    for p=1,nParts do
        partCmaps[p] = colormap[p*torch.floor(256/(nParts))]
        foutMtl:write(string.format('newmtl m%d\nKd %f %f %f\nKa 0 0 0\n',p,partCmaps[p][1], partCmaps[p][2],partCmaps[p][3]))
    end
    if(#partIndsSpecific == 0) then 
        for p=1,nParts do
            partIndsSpecific[p] = p
        end
    end
    --print(partIndsSpecific)
    foutMtl:close()
    fout:write(string.format('mtllib %s\n',mtlFile))
    local vertOffset = 0
    for px = 1,#partIndsSpecific do
        local p = partIndsSpecific[px]
        local shape, translation, quat = unpack(predParts[p])
        quat = quat:clone()
        quat[1] = -quat[1]
        local verts, faces = shapeMesh(shape, primTypes)
        if(verts:numel() > 0) then
            faces = faces+vertOffset
            local nVerts = verts:size(1)        
            verts = verts:view(1,nVerts,3)
            translation = translation:reshape(1,3):repeatTensor(nVerts,1)
            quat = quat:reshape(1,4)
            local rotator = transformer.rotation(nVerts)
            local vertsFinal = rotator:forward({verts, quat}) + translation
            vertsFinal = vertsFinal:squeeze()

            fout:write(string.format('usemtl m%d\n',p))
            for vx = 1,nVerts do
                fout:write(string.format('v %f %f %f\n', vertsFinal[vx][1], vertsFinal[vx][2], vertsFinal[vx][3]))
            end
            for fx = 1,faces:size(1) do
                fout:write(string.format('f %d %d %d\n', faces[fx][1], faces[fx][2], faces[fx][3]))
            end
            vertOffset = vertOffset+nVerts
        end
    end
    fout:close()
end

local function savePoints(points, outputFile)
    local fout = io.open(outputFile, 'w')
    local vertOffset = 0
    --print(points:size())
    for p = 1,points:size(1) do
        local verts, faces = cuboidMesh(torch.Tensor(3):fill(0.01))
        faces = faces+vertOffset
        local vertsFinal = points[p]:repeatTensor(verts:size(1),1) + verts
        for vx = 1,vertsFinal:size(1) do
            fout:write(string.format('v %f %f %f\n', vertsFinal[vx][1], vertsFinal[vx][2], vertsFinal[vx][3]))
        end
        for fx = 1,faces:size(1) do
            fout:write(string.format('f %d %d %d\n', faces[fx][1], faces[fx][2], faces[fx][3]))
        end
        vertOffset = vertOffset+vertsFinal:size(1)
    end
    fout:close()
end

local function renderMesh(blendFile, meshFile, pngFile)
    local command = string.format('/home/eecs/shubhtuls/Downloads/renderer/render.sh %s %s %s',blendFile,meshFile,pngFile)
    --print(command)
    os.execute(command)
end

local function writeObj(meshFile, vertices, faces)
    local fout = io.open(meshFile, 'w')
    for vx = 1,vertices:size(1) do
        fout:write(string.format('v %f %f %f\n', vertices[vx][1], vertices[vx][2], vertices[vx][3]))
    end
    for fx = 1,faces:size(1) do
        fout:write(string.format('f %d %d %d\n', faces[fx][1], faces[fx][2], faces[fx][3]))
    end
    fout:close()
end

local function saveParse(meshFile, vertices, faces, faceInds, nParts)
    local mtlFile = meshFile:split('.obj')[1] .. '.mtl'
    local fout = io.open(meshFile, 'w')
    local foutMtl = io.open(mtlFile, 'w')
    
    local partCmaps = {}
    local fCounter = 0
    for p=1,nParts do
        partCmaps[p] = colormap[p*torch.floor(256/(nParts))]
        foutMtl:write(string.format('newmtl m%d\nKd %f %f %f\nKa 0 0 0\n',p,partCmaps[p][1], partCmaps[p][2],partCmaps[p][3]))
    end
    foutMtl:close()
    fout:write(string.format('mtllib %s\n',mtlFile))
    
    for vx = 1,vertices:size(1) do
        fout:write(string.format('v %f %f %f\n', vertices[vx][1], vertices[vx][2], vertices[vx][3]))
    end
    for fx = 1,faces:size(1) do
        --local v1 = faces[fx][1]
        --local v2 = faces[fx][2]
        --local v3 = faces[fx][3]
        local mtlId = faceInds[fx]
        --if(partInds[v2] == partInds[v3]) then mtlId = partInds[v2] end
        fout:write(string.format('usemtl m%d\n',mtlId))
        fout:write(string.format('f %d %d %d\n', faces[fx][1], faces[fx][2], faces[fx][3]))
    end
    fout:close()
end


local function saveParseSpecificPart(meshFile, vertices, faces, faceInds, nParts, chosenPart)
    local mtlFile = meshFile:split('.obj')[1] .. '.mtl'
    local fout = io.open(meshFile, 'w')
    local foutMtl = io.open(mtlFile, 'w')
    
    local partCmaps = {}
    local fCounter = 0
    for p=1,nParts do
        partCmaps[p] = colormap[p*torch.floor(256/(nParts))]
        if(p == chosenPart) then
            foutMtl:write(string.format('newmtl m%d\nKd %f %f %f\nKa 0 0 0\n',p,partCmaps[p][1], partCmaps[p][2],partCmaps[p][3]))
        else
            foutMtl:write(string.format('newmtl m%d\nKd %f %f %f\nKa 0 0 0\n illum 9\n',p,0.5,0.5,0.5))
        end
    end
    foutMtl:close()
    fout:write(string.format('mtllib %s\n',mtlFile))
    
    for vx = 1,vertices:size(1) do
        fout:write(string.format('v %f %f %f\n', vertices[vx][1], vertices[vx][2], vertices[vx][3]))
    end
    for fx = 1,faces:size(1) do
        --local v1 = faces[fx][1]
        --local v2 = faces[fx][2]
        --local v3 = faces[fx][3]
        local mtlId = faceInds[fx]
        --if(partInds[v2] == partInds[v3]) then mtlId = partInds[v2] end
        fout:write(string.format('usemtl m%d\n',mtlId))
        fout:write(string.format('f %d %d %d\n', faces[fx][1], faces[fx][2], faces[fx][3]))
    end
    fout:close()
end

------------------------
local function dumpPoint(fout, x, y, z, radius , matName, counter)
    --if radius < 0.2 then return end
    fout:write(string.format('usemtl %s\n',matName))
    local v = cubeV * radius
    local p_v = torch.data(v)
    local p_f = torch.data(cubeF)
    local s_v = cubeV:stride(1)
    local s_f = cubeF:stride(1)
    local numvert = cubeV:size()[1]
    for i=0,numvert-1 do
        fout:write(string.format('v %f %f %f\n', x+p_v[i*s_v+0], y+p_v[i*s_v+1], z+p_v[i*s_v+2]))
    end
    local numface = cubeF:size()[1]
    for i = 0,numface-1 do
        fout:write(string.format('f %d %d %d\n', p_f[i*s_f+0]+counter, p_f[i*s_f+1]+counter, p_f[i*s_f+2]+counter))
    end
end

local function saveVoxelsMesh(predVol, meshFile, thresh)
    local fCounter = 0
    local mtlFile = meshFile:split('.obj')[1] .. '.mtl'
    local outputFile = meshFile
    local fout = io.open(outputFile, 'w')
    local foutMtl = io.open(mtlFile, 'w')
    
    foutMtl:write(string.format('newmtl m0\nKd %f %f %f\nKa 0 0 0\n', 0.5,0.5,0.5))
    local matName = string.format('m0',p)

    fout:write(string.format('mtllib %s\n',mtlFile))
    local maxDim = 0.5
    --print(predVol:size())
    for x =1,predVol:size(1) do
        for y=1,predVol:size(2) do
            for z=1,predVol:size(3) do
                --print(x,y,z)
                if predVol[x][y][z] > thresh then
                    dumpPoint(fout, 2*maxDim*(x-1)/31 - maxDim, 2*maxDim*(y-1)/31 - maxDim, 2*maxDim*(z-1)/31 - maxDim, maxDim/32 , matName, fCounter)
                    fCounter = fCounter+8
                end
            end
        end
    end
    
    fout:close()
    foutMtl:close()
    
end
------------------------

M.savePoints = savePoints
M.saveParts = saveParts
M.saveParse = saveParse
M.saveParseSpecificPart = saveParseSpecificPart
M.shapeMesh = shapeMesh
M.saveVoxelsMesh = saveVoxelsMesh
M.partVertices = partVertices
M.renderMesh = renderMesh
M.writeObj = writeObj
M.shapeVolume = shapeVolume
M.triliniearCoeff = triliniearCoeff
return M