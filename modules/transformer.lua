require 'nn'
require 'nngraph'
local M = {}
local quatUtils = dofile('../modules/quatUtils.lua')
-------------------------------
-------------------------------
-- input is BXnPX3 points, BX4 quauternion dimensions
-- output is BXnPX3 points
local function rotation(nP)
    local points = - nn.Identity()
    local zero = points - nn.Narrow(3,1,1) - nn.MulConstant(0,false)
    local pointsQuat = {zero,points} - nn.JoinTable(3) --prepernds zero to 'real' part of points
    
    local quat = - nn.Identity()
    local quatRep = quat - nn.Replicate(nP,2)
    local rot = {pointsQuat, quatRep} - quatUtils.quatRotateModule()
    
    nngraph.annotateNodes()
    local gmod = nn.gModule({points, quat}, {rot})
    return gmod
end
-------------------------------
-------------------------------
-- input is BXnPX3 points, BX3 translation vectors
-- output is BXnPX3 points
local function translation(nP)
    local points = - nn.Identity()
    local trans = - nn.Identity()
    local transRep = trans - nn.Replicate(nP,2)
    
    local final = {points,transRep} - nn.CAddTable()
    
    nngraph.annotateNodes()
    local gmod = nn.gModule({points, trans}, {final})
    return gmod
end
-------------------------------
-------------------------------
-- input is BXnPX3 points, BX3 translation vectors, BX4 quaternions
-- output is BXnPX3 points
-- performs p_out = R*(p_in - t)
local function rigidTsdf(nP)
    local points = - nn.Identity()
    local trans = - nn.Identity()
    local quat = - nn.Identity()
    
    local minus_t = trans - nn.MulConstant(-1,false)
    local p1 = {points,minus_t} - translation(nP)
    local p2 = {p1,quat} - rotation(nP)
    
    nngraph.annotateNodes()
    local gmod = nn.gModule({points, trans, quat}, {p2})
    return gmod
end
-------------------------------
-------------------------------
-- input is {gridPoints, {partParams, trans, quat}}
local function tsdfTransform(tsdfFunc, nP)
    local points = - nn.Identity()
    local part = - nn.Identity()
    local shape = part - nn.SelectTable(1)
    local trans = part - nn.SelectTable(2)
    local quat = part - nn.SelectTable(3)
    
    local p1 = {points, trans, quat} - rigidTsdf(nP)
    local tsdf = {p1, shape} - tsdfFunc
    
    nngraph.annotateNodes()
    local gmod = nn.gModule({points, part}, {tsdf})
    return gmod
end
-------------------------------
-------------------------------
-- input will be {points, {s_i}}, where s_i = {z_i, tr_i, quat_i}
local function partComposition(primitiveGenerator, nParts, nPoints)
    if(nParts == 1) then
        local points = - nn.Identity()
        local shapeParams = - nn.Identity()
        local s_i = shapeParams - nn.SelectTable(1)
        local tsdf = tsdfTransform(primitiveGenerator(nPoints), nPoints)
        local tsdfPart = {points, s_i} - tsdf - nn.Unsqueeze(3)
        local gmod = nn.gModule({points, shapeParams}, {tsdfPart})
        return gmod
    end
    
    local points = - nn.Identity()
    local shapeParams = - nn.Identity()
    local tsdfParts = {}
    for i = 1,nParts do
        local tsdf = tsdfTransform(primitiveGenerator(nPoints), nPoints)
        local s_i = shapeParams - nn.SelectTable(i)
        tsdfParts[i] = {points, s_i} - tsdf - nn.Unsqueeze(3)
    end
    --local tsdfOut = tsdfParts - nn.CMinTable()
    local tsdfOut = tsdfParts - nn.JoinTable(3) - nn.Min(3)
    
    nngraph.annotateNodes()
    local gmod = nn.gModule({points, shapeParams}, {tsdfOut})
    return gmod
end


local function imgFrameTransformer(primitiveGenerator, nParts, nPoints)
    local points = - nn.Identity()
    local pred = - nn.Identity()
    
    local globalRot = pred - nn.SelectTable(1)
    local rotatedPts = {points, globalRot} - rotation(nPoints)
    
    local predShapeCanonical = pred - nn.SelectTable(2)
    local predTsdf = {rotatedPts, predShapeCanonical} - partComposition(primitiveGenerator, nParts, nPoints)

    nngraph.annotateNodes()
    local gmod = nn.gModule({points, pred}, {predTsdf})
    return gmod
end
-------------------------------
-------------------------------
M.rotation = rotation
M.translation = translation
M.rigidTsdf = M.rigidTsdf
M.tsdfTransform = tsdfTransform
M.imgFrameTransformer = imgFrameTransformer
M.partComposition = partComposition
return M