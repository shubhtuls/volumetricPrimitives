require 'nn'
require 'nngraph'
if(nn.ReinforceCategorical == nil) then
    dofile('../modules/ReinforceCategorical.lua')
end
local nUtils = dofile('../modules/netUtils.lua')
local M = {}
-------------------------------
-------------------------------
-- input is BXnPX3 points, BX3 cuboid dimensions
-- output is BXnP tsdf^2 values
local function cuboid(nP)
    local points = - nn.Identity()
    local dims = - nn.Identity()
    local pAbs = points - nn.Abs()
    
    local dimsRep = dims - nn.Replicate(nP,2)
    local tsdfSq = {pAbs,dimsRep} - nn.CSubTable() - nn.ReLU() - nn.Power(2) - nn.Sum(3)
    local gmod = nn.gModule({points, dims}, {tsdfSq})
    return gmod
end

local function cuboidInterior(nP)
    local points = - nn.Identity()
    local dims = - nn.Identity()
    local pAbs = points - nn.Abs()
    
    local dimsRep = dims - nn.Replicate(nP,2)
    local tsdfSq = {dimsRep, pAbs} - nn.CSubTable() - nn.ReLU() - nn.Min(3) - nn.Power(2)
    local gmod = nn.gModule({points, dims}, {tsdfSq})
    return gmod
end
-------------------------------
-------------------------------
-- input is BXnPX3 points, BX1 cuboid dimensions
-- output is BXnP tsdf^2 values, all of which are set to 1.
local function null(nP)
    local points = - nn.Identity()
    local dims = - nn.Identity()
    --Add a ReLU as it blocks gradients
    local pAbs = points - nn.MulConstant(0) - nn.Sum(3) - nn.AddConstant(-1) - nn.ReLU()
    local dimsRep = dims - nn.Sum(2) - nn.MulConstant(0) - nn.AddConstant(-1) - nn.ReLU() - nn.Replicate(nP,2)
    local tsdfSq = {pAbs,dimsRep} - nn.CAddTable() - nn.AddConstant(1)
    local gmod = nn.gModule({points, dims}, {tsdfSq})
    return gmod
end

-- input is BXnPX3 points, BX1 cuboid dimensions
-- output is BXnP tsdf^2 values, all of which are set to 0.
local function nullInterior(nP)
    local points = - nn.Identity()
    local dims = - nn.Identity()
    --Add a ReLU as it blocks gradients
    local pAbs = points - nn.MulConstant(0) - nn.Sum(3) - nn.AddConstant(-1) - nn.ReLU()
    local dimsRep = dims - nn.Sum(2) - nn.MulConstant(0) - nn.AddConstant(-1) - nn.ReLU() - nn.Replicate(nP,2)
    local tsdfSq = {pAbs,dimsRep} - nn.CAddTable() - nn.AddConstant(0)
    local gmod = nn.gModule({points, dims}, {tsdfSq})
    return gmod
end
-------------------------------
-------------------------------
local function primitiveModule(key, nP)
    if(key == 'Cu') then
        return cuboid(nP), 3
    elseif(key=='Nu') then
        return null(nP), 1
    else
        error("invalid input")
    end
end
-------------------------------
-------------------------------
-- input is BXnPX3 points, BXnZ part params
-- output is BXnP tsdf^2 values
local function primitiveSelector(primitives, nP)
    if(#primitives == 1) then
        local pMod, _ = primitiveModule(primitives[1], nP)
        return pMod
    end
    local nChoices = #primitives
    
    local points = - nn.Identity()
    local dims = - nn.Identity()
    
    local start=1
    local pGenModules = {}
    for p=1,#primitives do
        local pMod, nz = primitiveModule(primitives[p], nP)
        local shapeParam = dims - nn.Narrow(2,start,nz)
        pGenModules[p] = {points, shapeParam} - pMod - nn.Unsqueeze(3)
        start = start+nz
    end
    
    local sampler = dims - nn.Narrow(2,start,#primitives) - nn.Replicate(nP,2)
    local tsdfAll = pGenModules - nn.JoinTable(3)
    local tsdfSq = {tsdfAll, sampler} - nn.CMulTable() - nn.Sum(3)
    
    local gmod = nn.gModule({points, dims}, {tsdfSq})
    return gmod
end
-------------------------------
-------------------------------
local function meshGrid(minVal, maxVal, gridSize)
    -- Xs, Ys, Zs = MeshGrid
    
    local pointsX = torch.linspace(minVal[1],maxVal[1],gridSize[1])
    local pointsY = torch.linspace(minVal[2],maxVal[2],gridSize[2])
    local pointsZ = torch.linspace(minVal[3],maxVal[3],gridSize[3])

    local xs = torch.repeatTensor(pointsX:view(-1, 1, 1, 1), 1, gridSize[2], gridSize[3], 1)
    local ys = torch.repeatTensor(pointsY:view(1, -1, 1, 1), gridSize[1], 1, gridSize[3], 1)
    local zs = torch.repeatTensor(pointsZ:view(1, 1, -1, 1), gridSize[1], gridSize[2], 1, 1)
    return torch.cat(xs, torch.cat(ys, zs))
end
-------------------------------
-------------------------------
-- input is B X C X 1 X 1 X1, output is B X nz
local function shapePred(params, outChannelsV, biasTerms)
    local shapeLayer = nn.VolumetricConvolution(outChannelsV, params.nz, 1, 1, 1)
    local shapeLrDecay = params.shapeLrDecay or 1
    shapeLayer:apply(nUtils.weightsInit)
    shapeLayer.note = 'shapePred'
    --shapeLayer.weight:fill(0)
    local biasTerms = biasTerms or {}
    if biasTerms['shape'] then
        shapeLayer.bias = biasTerms['shape']:clone()
    --else
    --    shapeLayer.bias = torch.Tensor(params.nz):uniform(-2,0)
    end
    local shapeModule = nn.Sequential():add(shapeLayer):add(nn.MulConstant(shapeLrDecay)):add(nn.Sigmoid()):add(nn.MulConstant(params.gridBound)):add(nn.Squeeze())
    return shapeModule
end

-- input is B X C X 1 X 1 X1, output is B X 3
local function translationPred(params, outChannelsV, biasTerms)
    local transLayer = nn.VolumetricConvolution(outChannelsV, 3, 1, 1, 1)
    transLayer:apply(nUtils.weightsInit)
    local biasTerms = biasTerms or {}
    if biasTerms['trans'] then
        transLayer.bias = biasTerms['trans']:clone()
    --else
    --    transLayer.bias = torch.Tensor(3):uniform(-1,1)
    end
    local transModule = nn.Sequential():add(transLayer):add(nn.Tanh()):add(nn.MulConstant(params.gridBound)):add(nn.Squeeze())
    return transModule
end

-- input is B X C X 1 X 1 X1, output is B X 4
local function quatPred(params, outChannelsV, biasTerms)
    local quatLayer = nn.VolumetricConvolution(outChannelsV, 4, 1, 1, 1)
    quatLayer:apply(nUtils.weightsInit)
    local biasTerms = biasTerms or {}
    if biasTerms['quat'] then
        quatLayer.bias = biasTerms['quat']:clone()
    end
    local transModule = nn.Sequential():add(quatLayer):add(nn.Squeeze()):add(nn.Normalize(2))
    return transModule
end

-- input is B X C X 1 X 1 X1, output is B X nChoices
local function probPred(params, outChannelsV, nChoices, biasTerms)
    local probLayer = nn.VolumetricConvolution(outChannelsV, nChoices, 1, 1, 1)
    probLayer:apply(nUtils.weightsInit)
    probLayer.note = 'probPred'
    local probLrDecay = params.probLrDecay or 1
    local biasTerms = biasTerms or {}
    if biasTerms['prob'] then
        probLayer.bias = biasTerms['prob']:clone()
    end
    local transModule = nn.Sequential():add(probLayer):add(nn.MulConstant(probLrDecay)):add(nn.Squeeze()):add(nn.SoftMax())
    return transModule
end

local function primitivePred(params, outChannelsV, biasTerms)
    local feat = - nn.Identity()
    local shape = feat - shapePred(params, outChannelsV, biasTerms)
    local quat = feat - quatPred(params, outChannelsV, biasTerms)
    local trans = feat - translationPred(params, outChannelsV, biasTerms)
    
    local gmod = nn.gModule({feat}, {shape, trans, quat})
    return gmod
end

local function primitiveSelectorPred(params, outChannelsV, biasTerms)
    local feat = - nn.Identity()
    local nPrims = params.nPrimChoices
    local shape, quat, trans
    if(nPrims == 1) then
        shape = feat - shapePred(params, outChannelsV, biasTerms)
        quat = feat - quatPred(params, outChannelsV, biasTerms)
        trans = feat - translationPred(params, outChannelsV, biasTerms)
    else
        local probs = feat - probPred(params, outChannelsV, nPrims, biasTerms)
        probs = probs:annotate{name = 'probs'}
        local samples = probs - nn.ReinforceCategorical(params.bMomentum, params.entropyWt, params.intrinsicReward)
        samples = samples:annotate{name = 'samples'}
        
        local shapeParams = feat - shapePred(params, outChannelsV, biasTerms)
        trans = feat - translationPred(params, outChannelsV, biasTerms) --a single translation predicted across all parts
        
        local quatTable = {}
        --local transTable = {}
        for p=1,nPrims do
            quatTable[p] = feat - quatPred(params, outChannelsV, biasTerms) - nn.Unsqueeze(3)
            --transTable[p] = feat - translationPred(params, outChannelsV, biasTerms) - nn.Unsqueeze(3)
        end
        
        local quatAll = quatTable - nn.JoinTable(3)
        --local transAll = transTable - nn.JoinTable(3)
        
        local sampleQuatRep = samples - nn.Replicate(4,2)
        --local sampleTransRep = samples - nn.Replicate(3,2)
        
        quat = {quatAll, sampleQuatRep} - nn.CMulTable() - nn.Sum(3)
        quat = quat:annotate{name = 'quat'}
        --trans = {transAll, sampleTransRep} - nn.CMulTable() - nn.Sum(3)
        shape = {shapeParams, samples} - nn.JoinTable(2)
    end
    local gmod = nn.gModule({feat}, {shape, trans, quat})
    return gmod
end

local function primitiveSelectorTable(params, outChannels, biasTerms)
    local primitivesTable = nn.ConcatTable()
    for p=1,params.nParts do
        primitivesTable:add(primitiveSelectorPred(params, outChannels, biasTerms))
    end
    return primitivesTable
end

-------------------------------
-------------------------------
M.cuboid = cuboid
M.null = null
M.cuboidInterior = cuboidInterior
M.nullInterior = nullInterior

M.primitiveSelector = primitiveSelector
M.meshGrid = meshGrid
M.quatPred = quatPred
M.primitivePred = primitivePred
M.primitiveSelectorPred = primitiveSelectorPred
M.primitiveSelectorTable = primitiveSelectorTable
return M