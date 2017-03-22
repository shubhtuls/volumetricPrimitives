local transformer = dofile('../modules/transformer.lua')
local interior = dofile('../modules/interior.lua')
local M = {}
-------------------------------
-------------------------------
local function shapeParamsExtract(shape, primTypes)
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
    return shapeParams,shapeType
end
-------------------------------
-------------------------------
local function sampleCuboidPoints(params, nS)
    local samples = torch.Tensor(nS,3):uniform(-1,1)
    samples:narrow(2,1,1):mul(params[1])
    samples:narrow(2,2,1):mul(params[2])
    samples:narrow(2,3,1):mul(params[3])
    return samples
end

local function samplePartPoints(pred, primTypes)
    local shape, translation, quat = unpack(pred)
    shape = shape:squeeze()
    local shapeParams, shapeType = shapeParamsExtract(shape, primTypes)
    local verts = torch.Tensor(1,3):fill(0)
    if(shapeType == 'Cu') then
        verts = sampleCuboidPoints(shapeParams, 1000)
    elseif(shapeType == 'Nu') then
        return nil
    end
    
    local nVerts = verts:size(1)        
    verts = verts:view(1,nVerts,3)
    translation = translation:reshape(1,3):repeatTensor(nVerts,1)
    quat = quat:reshape(1,4):clone()
    quat[1][1] = -quat[1][1]
    local rotator = transformer.rotation(nVerts)
    local vertsFinal = rotator:forward({verts, quat}) + translation
    vertsFinal = vertsFinal:squeeze()
    --print(vertsFinal:mean(1))
    return vertsFinal
    
end

local function isInside(pred, points, primTypes)
    local function pgenFuncTsdfInt(nP)
        return interior.primitiveSelector(primTypes, nP)
    end
    local nP = points:size(1)
    local ptsQuery = points:reshape(1,nP,3)
    local tsdfFunc = transformer.partComposition(pgenFuncTsdfInt, 1, nP)
    local tsdf_p = tsdfFunc:forward({ptsQuery, {pred}}):reshape(nP):clone()
    local var = torch.lt(tsdf_p,0):typeAs(points)
    --print(var:sum(), var:numel(), tsdf_p:mean())
    return var
end

local function intraPrimitiveIntersection(predParts, primTypes)
    local nParts = #predParts
    local intersection = torch.Tensor(nParts):fill(0)
    for pInit=1,nParts do
        local pSamples = samplePartPoints(predParts[pInit], primTypes)
        if(pSamples ~= nil) then
            local exteriorPoint = torch.Tensor(pSamples:size(1)):fill(1)
            for pQuery=1,nParts do
                if(pQuery ~= pInit) then
                    exteriorPoint:cmul(1-isInside(predParts[pQuery], pSamples, primTypes))
                end
            end
            intersection[pInit] = 1 -  exteriorPoint:sum()/exteriorPoint:numel()
        end        
    end
    return intersection    
end

local function setNull(part, primTypes)
    local nz = 0
    local nullInd = 1
    assert(#primTypes > 1)
    for ix = 1,#primTypes do
        local primSize = 0
        if(primTypes[ix] == 'Cu') then primSize = 3 end
        if(primTypes[ix] == 'Nu') then primSize = 1; nullInd = ix end
        nz = nz+primSize
    end
    local sclone = part[1]:clone():squeeze()
    sclone:narrow(1,nz+1,#primTypes):fill(0)
    sclone[nz+nullInd] = 1
    --print(sclone, part[1])
    part[1]:copy(sclone)
    
end

local function p1p2Intersection(predParts, primTypes, pInit, pQuery)
    local nParts = #predParts
    local intersection = torch.Tensor(nParts):fill(0)
    local pSamples = samplePartPoints(predParts[pInit], primTypes)
    if(pSamples ~= nil) then
        local exteriorPoint = torch.Tensor(pSamples:size(1)):fill(1)
        exteriorPoint:cmul(1-isInside(predParts[pQuery], pSamples, primTypes))
        return 1 - exteriorPoint:sum()/exteriorPoint:numel()
    end
    return 0
end
-------------------------------
-------------------------------
local function partDistance(p1, p2, primTypes, lambdaNull, lambdaS, lambdaQ, lambdaT)
    local nz = 3
    local lambdaNull = lambdaNull or 1
    local lambdaS = lambdaS or 1
    local lambdaQ = lambdaQ or 1
    local lambdaT = lambdaT or 1
    
    local s1, t1, q1 = unpack(p1)
    local s2, t2, q2 = unpack(p2)
    s1 = s1:squeeze();s2 = s2:squeeze()
    
    local s1Params, s1Type = shapeParamsExtract(s1, primTypes)
    local s2Params, s2Type = shapeParamsExtract(s2, primTypes)
    
    if(s1Type ~= s2Type) then return lambdaNull end
    if(s1Type == 'Nu') then return 0 end
    
    local e1 = torch.cat(s1Params*lambdaS, t1*lambdaT, q1*lambdaQ)
    local e2 = torch.cat(s2Params*lambdaS, t2*lambdaT, q2*lambdaQ)
    --print(e1,e2)
    
    return (e1-e2):pow(2):sum()
end

local function predDistance(p1, p2, primTypes, lambdaNull, lambdaS, lambdaQ, lambdaT, partIds)
    local lambdaNull = lambdaNull or 1
    local lambdaS = lambdaS or 1
    local lambdaQ = lambdaQ or 1
    local lambdaT = lambdaT or 1
    local dist = 0
    local allIds = {}; for ix = 1,#p1 do allIds[ix] = ix end
    local partIds = partIds or allIds
    for ix = 1,#partIds do
        local pId = partIds[ix]
        local dPart = partDistance(p1[pId], p2[pId], primTypes, lambdaNull, lambdaS, lambdaQ, lambdaT)
        dist = dist + dPart
        --print(dPart)
    end
    return dist
end

-------------------------------
-------------------------------
M.predDistance = predDistance
M.intraPrimitiveIntersection = intraPrimitiveIntersection
M.p1p2Intersection = p1p2Intersection
M.setNull = setNull
return M