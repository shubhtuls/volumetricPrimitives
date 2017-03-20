require 'nn'
require 'nngraph'
local M = {}
if(nn.CuboidSurface == nil) then
    dofile('../modules/surface/cuboid.lua')
end
if(nn.NullSurface == nil) then
    dofile('../modules/surface/null.lua')
end
local quatUtils = dofile('../modules/quatUtils.lua')
local transformer = dofile('../modules/transformer.lua')
-------------------------------
-------------------------------
local function primitiveModule(key_normFactor, nP)
    local keys = string.split(key_normFactor, '_')
    local key = keys[1]
    if(key == 'Cu') then
        return nn.CuboidSurface(nP, keys[2]), 3
    elseif(key=='Nu') then
        return nn.NullSurface(nP), 1
    else
        error("invalid input")
    end
end
-------------------------------
-------------------------------
-- input is BXnZ part params
-- output is BXnPX3 samples, BXnP weights
local function primitiveSelector(primitives, nSamples)
    if(#primitives == 1) then
        local pMod, _ = primitiveModule(primitives[1], nSamples)
        return pMod
    end
    local nChoices = #primitives
    
    local dims = - nn.Identity()
    local start=1
    local ptGenModules = {}
    local wtGenModules = {}
    
    for p=1,#primitives do
        local pMod, nz = primitiveModule(primitives[p], nSamples)
        local shapeParam = dims - nn.Narrow(2,start,nz)
        local sampleModule = shapeParam - pMod
        ptGenModules[p] = sampleModule - nn.SelectTable(1) - nn.Unsqueeze(4) -- size is B X nSamples X 3 X 1
        wtGenModules[p] = sampleModule - nn.SelectTable(2) - nn.Unsqueeze(3) -- size is B X nSamples X 1
        start = start+nz
    end
    
    local samplerWts = dims - nn.Narrow(2,start,#primitives) - nn.Replicate(nSamples,2)
    local samplerPts = samplerWts - nn.Replicate(3,3)
    
    local ptsAll = ptGenModules - nn.JoinTable(4)
    local ptsFinal = {ptsAll, samplerPts} - nn.CMulTable() - nn.Sum(4)

    local wtsAll = wtGenModules - nn.JoinTable(3)
    local wtsFinal = {wtsAll, samplerWts} - nn.CMulTable() - nn.Sum(3)
    
    local gmod = nn.gModule({dims}, {ptsFinal, wtsFinal})
    return gmod
end
-------------------------------
-------------------------------
-- input is BXnPX3 points, BX3 translation vectors, BX4 quaternions
-- output is BXnPX3 points
-- performs p_out = R'*p_in + t
local function rigidPointsTransform(nP)
    local points = - nn.Identity()
    local trans = - nn.Identity()
    local quat = - nn.Identity()
    local quatConj = quat - nn.Unsqueeze(2) - quatUtils.quatConjugateModule() - nn.Squeeze(2)
    
    local p1 = {points,quatConj} - transformer.rotation(nP)
    local p2 = {p1,trans} - transformer.translation(nP)
    
    local gmod = nn.gModule({points, trans, quat}, {p2})
    return gmod
end
-------------------------------
-------------------------------
-- input is {shapeParams, trans, quat}, output is samples, weights
local function primitiveSurfaceSamples(samplerModule, nSamples)
    local part = - nn.Identity()
    local shape = part - nn.SelectTable(1)
    local trans = part - nn.SelectTable(2)
    local quat = part - nn.SelectTable(3)
    
    local samples = {shape} - samplerModule
    local samplePoints = samples - nn.SelectTable(1)
    local sampleImportance = samples - nn.SelectTable(2)
    
    local transformedSamples = {samplePoints, trans, quat} - rigidPointsTransform(nSamples)
    
    local gmod = nn.gModule({part}, {transformedSamples, sampleImportance})
    return gmod
end
-------------------------------
-------------------------------
-- input will be {s_i}, where s_i = {z_i, tr_i, quat_i}
local function partComposition(surfaceSampler, nParts, nSamples)
    if(nParts == 1) then
        local shapeParams = - nn.Identity()
        local s_i = shapeParams - nn.SelectTable(1)
        local samples = s_i - primitiveSurfaceSamples(surfaceSampler(nSamples), nSamples)
        local gmod = nn.gModule({shapeParams}, {samples})
        return gmod
    end
    
    local shapeParams = - nn.Identity()
    local sampledPoints = {}
    local sampledWeights = {}
    
    for i = 1,nParts do
        local s_i = shapeParams - nn.SelectTable(i)
        local samples_i = s_i - primitiveSurfaceSamples(surfaceSampler(nSamples), nSamples)
        sampledPoints[i] = samples_i - nn.SelectTable(1)
        sampledWeights[i] = samples_i - nn.SelectTable(2)
    end
    --local tsdfOut = tsdfParts - nn.CMinTable()
    local pointsOut = sampledPoints - nn.JoinTable(2)
    local weightsOut = sampledWeights - nn.JoinTable(2)
    
    local gmod = nn.gModule({shapeParams}, {pointsOut, weightsOut})
    return gmod
end
-------------------------------
-------------------------------
M.partComposition = partComposition
M.primitiveSelector = primitiveSelector
return M