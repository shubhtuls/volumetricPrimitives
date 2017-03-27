-- sample script for chair training (stage 1)
--disp=0 gpu=1 nParts=20 nullReward=0 probLrDecay=0.0001 shapeLrDecay=0.01 synset=3001627 usePretrain=0 numTrainIter=20000 name=chairChamferSurf_null_small_init_prob0pt0001_shape0pt01 th cadAutoEncCuboids/primSelTsdfChamfer.lua

-- sample script for chair training (stage 2)
--pretrainNet=chairChamferSurf_null_small_init_prob0pt0001_shape0pt01 pretrainIter=20000 disp=0 gpu=1 nParts=20 nullReward=8e-5 shapeLrDecay=0.5   synset=3001627 probLrDecay=0.2 usePretrain=1  numTrainIter=30000 name=chairChamferSurf_null_small_ft_prob0pt2_shape0pt5_null8em5 th cadAutoEncCuboids/primSelTsdfChamfer.lua

require 'cunn'
require 'nngraph'
require 'optim'
--nngraph.setDebug(true)

local function addTableRec(t1,t2,w1,w2)
    if(torch.isTensor(t1)) then
        return w1*t1+w2*t2
    end
    local result = {}
    for ix=1,#t1 do
        result[ix] = addTableRec(t1[ix],t2[ix],w1,w2)
    end
    return result
end

local nUtils = dofile('../modules/netUtils.lua')
local vUtils = dofile('../modules/visUtils.lua')
local transformer = dofile('../modules/transformer.lua')
local primitives = dofile('../modules/primitives.lua')
local transformerSurface = dofile('../modules/transformerSurface.lua')
local chamferUtils = dofile('../modules/surface/chamferCriterionSeparable.lua')
local symUtils = dofile('../modules/surface/symmetryCriterion.lua')
local vE = dofile('../modules/volumeEncoder.lua')
local mUtils = dofile('../modules/meshUtils.lua')
local mc = dofile('../modules/marchingCubes.lua')
local cbData = dofile('../data/cadConfigsChamfer.lua')

-- parameters
local params = {}
params.learningRate = 0.001
params.meshSaveIter = 1000
params.numTrainIter = 50000
params.batchSize = 32
params.batchSizeVis = 4
params.visPower = 0.25
params.lossPower = 2
params.chamferLossWt = 1
params.symLossWt = 1
params.gridSize = 32
params.gridBound = 0.5
params.useBn = 1
params.nParts = 6
params.disp = 0
params.imsave = 0
params.shapeLrDecay = 1
params.probLrDecay = 1
params.gpu = 1
params.visIter = 100
params.modelIter = 2 --data loader reloads models after these many iterations
params.synset = 3001627 --chair:3001627, aero:2691156, table:4379243
params.name = 'mainCadAutoEnc'
params.bMomentum = 0.9 --baseline momentum for reinforce
params.entropyWt = 0
params.nullReward = 0.0001
params.nSamplePoints = 1000
params.nSamplesChamfer = 150 --number of points we'll sample per part
params.useCubOnly = 0
params.usePretrain = 0
params.normFactor = 'Surf'
params.pretrainNet = 'chairChamferSurf_null_small_init_prob0pt0001_shape0pt01'
params.pretrainLrShape = 0.01
params.pretrainLrProb = 0.0001
params.pretrainIter = 20000

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if params.useBn == 0 then params.useBn = false end
if params.usePretrain == 0 then params.usePretrain = false end
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.pretrainLrs = {params.pretrainLrShape, params.pretrainLrProb}

params.modelsDataDir = '../cachedir/shapenet/chamferData/' .. params.synset
params.visDir = '../cachedir/visualization/' .. params.name
params.visMeshesDir = '../cachedir/visualization/meshes/' .. params.name
params.snapshotDir = '../cachedir/snapshots/' .. params.name

local dataLoader = cbData.SimpleCad(params)

if(params.useCubOnly == 1) then
    params.nz = 3-- dimension of cuboid gen space
    params.primTypes = {'Cu'}
else
    params.nz = 3 + 1-- dimension of cuboid gen space + cylinder space + null space
    params.primTypes = {'Cu','Nu'}
end
params.nPrimChoices = #params.primTypes
params.intrinsicReward = torch.Tensor(#params.primTypes):fill(0)
for p=1,#params.primTypes do
    if(params.primTypes[p] == 'Nu') then params.intrinsicReward[p] = -params.nullReward end
end

paths.mkdir(params.visDir)
paths.mkdir(params.visMeshesDir)
paths.mkdir(params.snapshotDir)
cutorch.setDevice(params.gpu)
local fout = io.open(paths.concat(params.snapshotDir,'log.txt'), 'w')
for k,v in pairs(params) do
    fout:write(string.format('%s : %s\n',tostring(k),tostring(v)))
end
print(params)
local meshSaveIter = params.meshSaveIter

local function pgenFuncTsdf(nP)
    return primitives.primitiveSelector(params.primTypes, nP)
end

params.primTypesSurface = {}
for p=1,#params.primTypes do
    params.primTypesSurface[p] = params.primTypes[p]
    if(params.primTypes[p] ~= 'Nu') then params.primTypesSurface[p] = params.primTypes[p] .. '_' .. params.normFactor end
end
local function pgenFuncSurface(nP)
    return transformerSurface.primitiveSelector(params.primTypesSurface, nP)
end

--------------------------------------------------------------
-- Net for predicting primitives
local netPred, outChannelsInput = vE.convEncoderSimple3d(5,4,1,params.useBn)
netPred:apply(nUtils.weightsInit)
local outChannels = outChannelsInput
for nLayer=1,2 do -- fc layers for joint reasoning
    netPred:add(nn.VolumetricConvolution(outChannels, outChannels, 1, 1, 1))
    if(params.useBn) then netPred:add(nn.VolumetricBatchNormalization(outChannels)) end
    netPred:add(nn.LeakyReLU(0.2, true))
end
netPred:apply(nUtils.weightsInit)
local biasTerms = {}

biasTerms.quat = torch.Tensor({1,0,0,0})
biasTerms.shape = torch.Tensor(params.nz):fill(-3)/params.shapeLrDecay
biasTerms.prob = torch.Tensor(#params.primTypes):fill(0)

for p=1,#params.primTypes do
    if(params.primTypes[p] == 'Cu') then biasTerms.prob[p] = 2.5/params.probLrDecay; end
end

local primitivesTable = primitives.primitiveSelectorTable(params, outChannels, biasTerms)
netPred:add(primitivesTable)

--------------------------------------------------------------
--------------------------------------------------------------
local nSamplePointsTrain = params.nSamplePoints
local nSamplePointsTest = params.gridSize^3

-- Modules for Loss function via TSDF
local tsdfComputerTrain = nn.Sequential():add(transformer.partComposition(pgenFuncTsdf, params.nParts, nSamplePointsTrain)):add(nn.ReLU()):add(nn.Power(0.5*params.lossPower))

local tsdfComputerTest = nn.Sequential():add(transformer.partComposition(pgenFuncTsdf, params.nParts, nSamplePointsTest)):add(nn.Power(0.5*params.visPower))

tsdfComputerTrain = tsdfComputerTrain:cuda()
tsdfComputerTest = tsdfComputerTest:cuda()

--------------------------------------------------------------
--------------------------------------------------------------
-- Modules for Loss function via Chamfer Distance on Samples
local surfaceSamplerModule = transformerSurface.partComposition(pgenFuncSurface, params.nParts, params.nSamplesChamfer)

local chamferCriterion = chamferUtils.ChamferCriterion(surfaceSamplerModule, params.nParts*params.nSamplesChamfer)

--------------------------------------------------------------
--------------------------------------------------------------
-- Modules for Symmetry Loss function
local surfaceSamplerModuleSym = transformerSurface.partComposition(pgenFuncSurface, params.nParts, params.nSamplesChamfer)
local tsdfComputerSym = nn.Sequential():add(transformer.partComposition(pgenFuncTsdf, params.nParts, params.nParts*params.nSamplesChamfer))
local symCriterion = symUtils.SymmetryCriterion(surfaceSamplerModuleSym, tsdfComputerSym, params.nParts*params.nSamplesChamfer)

--------------------------------------------------------------
--------------------------------------------------------------
-- Optimization parameters
local optimState = {
   learningRate = params.learningRate,
   beta1 = 0.9,
}

local criterion = nn.AbsCriterion()
local tsdfSqModTrain = nn.Sequential():add(nn.ReLU()):add(nn.Power(params.lossPower))
local tsdfSqModTest = nn.Sequential():add(nn.ReLU()):add(nn.Power(params.visPower))
local err = 0
local errTsdf = 0
local errChamfer = 0
local errSym = 0
netPred = netPred:cuda()
criterion = criterion:cuda()
chamferCriterion:cuda()
symCriterion:cuda()
tsdfSqModTrain = tsdfSqModTrain:cuda()
tsdfSqModTest = tsdfSqModTest:cuda()

local netParameters, netGradParameters = netPred:getParameters()

if params.usePretrain then
    local updateShapeWtFunc = nUtils.scaleWeightsFunc(params.pretrainLrs[1]/params.shapeLrDecay, 'shapePred')
    local updateProbWtFunc = nUtils.scaleWeightsFunc(params.pretrainLrs[2]/params.probLrDecay, 'probPred')
    
    local netPretrain = torch.load(paths.concat('../cachedir/snapshots', params.pretrainNet,'iter' .. tostring(params.pretrainIter) .. '.t7'))
    local netPretainParameters, _ = netPretrain:getParameters()
    netParameters:copy(netPretainParameters)
    netPred:apply(updateShapeWtFunc)
    netPred:apply(updateProbWtFunc)
end

-- fX required for training
local fx = function(x)
    netGradParameters:zero()
    local inputVol, tsdfGt, sampledPoints = unpack(dataLoader:forward())
    inputVol = inputVol:clone():cuda()
    sampledPoints = sampledPoints:cuda()
    local predParts = netPred:forward(inputVol)
    --print(predParts[1][1]:mean(1), predParts[1][2]:mean(1), predParts[1][3])

    -- Tsdf Loss
    local predTsdf = tsdfComputerTrain:forward({sampledPoints, predParts})
    local tsdfGtSq = tsdfSqModTrain:forward(tsdfGt:clone():cuda()):reshape(predTsdf:size())
    errTsdf = criterion:forward(predTsdf, tsdfGtSq)
    local gradPredTsdf = tsdfComputerTrain:backward({sampledPoints, predParts},criterion:backward(predTsdf, tsdfGtSq))[2]
    local reinforceErrs = torch.abs(tsdfGtSq:clone() - predTsdf:clone()):mean(2):squeeze()
    local gradPred = gradPredTsdf
    
    -- Chamfer Loss
    if(params.chamferLossWt > 0) then
        errChamfer = chamferCriterion:forward(predParts, dataLoader)
        local gradPredChamfer = chamferCriterion:backward(predParts, dataLoader)
        reinforceErrs = reinforceErrs + params.chamferLossWt*chamferCriterion.weightedLosses:sum(2):squeeze()
        gradPred = addTableRec(gradPred, gradPredChamfer, 1, params.chamferLossWt)
    end
    
    -- Symmetry Loss
    if(params.symLossWt > 0) then
        errSym = symCriterion:forward(predParts)
        local gradPredSym = symCriterion:backward(predParts)
        reinforceErrs = reinforceErrs + params.symLossWt*symCriterion.weightedLosses:sum(2):squeeze()
        gradPred = addTableRec(gradPred, gradPredSym, 1, params.symLossWt)
    end
        
    local rewFunc = nUtils.updateRewardsFunc(reinforceErrs)
    netPred:apply(rewFunc)
    ----------------------
    err = errTsdf + params.chamferLossWt*errChamfer + params.symLossWt*errSym
    netPred:backward(inputVol, gradPred)
    return err, netGradParameters
end

if(params.disp == 1) then disp = require 'display' end

---------------------------
-- Logging part probs
local partProbLoggers = {}
local loggerStyles = {}
for ix=1,#params.primTypes do
    loggerStyles[ix] = '-'
end
for p=1,params.nParts do
    local logger = optim.Logger(paths.concat(params.snapshotDir,'part_' .. tostring(p) .. '.log'))
    logger:setNames(params.primTypes)
    logger.showPlot = false
    logger:style(loggerStyles)
    partProbLoggers[p] = logger
end
---------------------------
-- Logging part probs
local loggerStyles = {}
local errNames = {'errTot','errTsdf','errChamfer','errSym'}
for ix=1,#errNames do
    loggerStyles[ix] = '-'
end
local errLogger = optim.Logger(paths.concat(params.snapshotDir,'error.log'))
errLogger:setNames(errNames)
errLogger.showPlot = false
errLogger:style(loggerStyles)
---------------------------
---------------------------
-- Training
for iter=1,params.numTrainIter do
    print(iter,err,errTsdf,errChamfer,errSym)
    fout:write(string.format('%d %f\n',iter,err))
    if(iter % params.visIter)==0 then
        for p=1,params.nParts do
            partProbLoggers[p]:plot()
        end
        errLogger:plot()
        if(params.disp == 1 or params.imsave == 1 or iter%meshSaveIter == 0) then
            local reshapeSize = torch.LongStorage({params.batchSizeVis,1,params.gridSize,params.gridSize,params.gridSize})

            local sample, tsdfGt, sampledPoints = unpack(dataLoader:forwardTest())
            sampledPoints = sampledPoints:narrow(1,1,params.batchSizeVis):cuda()
            sample = sample:narrow(1,1,params.batchSizeVis):cuda()
            tsdfGt = tsdfGt:narrow(1,1,params.batchSizeVis):reshape(reshapeSize)
            
            local tsdfGtSq = tsdfSqModTest:forward(tsdfGt:clone():cuda())
            netPred:apply(nUtils.setTestMode)
            local shapePredParams = netPred:forward(sample)
            netPred:apply(nUtils.unsetTestMode)
            local pred = tsdfComputerTest:forward({sampledPoints:narrow(1,1,params.batchSizeVis), shapePredParams})
            local tsdfPredSq = pred:reshape(reshapeSize)
            
            if(params.disp == 1) then
                disp.image(sample:max(3):squeeze(), {win=1, title='sampleX'})
                disp.image(sample:max(4):squeeze(), {win=2, title='sampleY'})
                disp.image(sample:max(5):squeeze(), {win=3, title='sampleZ'})
            end
            if(params.imsave == 1) then
                vUtils.imsave(sample:max(3):squeeze(), params.visDir .. '/sampleX'.. iter .. '.png')
                vUtils.imsave(sample:max(4):squeeze(), params.visDir .. '/sampleY'.. iter .. '.png')
                vUtils.imsave(sample:max(5):squeeze(), params.visDir .. '/sampleZ'.. iter .. '.png')
            end
            local dispOut = -1*tsdfPredSq:clone()
            --print(dispOut:max(), dispOut:min())
            local dispOutGt = -1*tsdfGtSq:clone()
            if(params.disp == 1) then
                disp.image(dispOut:max(3):squeeze(), {win=4, title='predX'})
                disp.image(dispOut:max(4):squeeze(), {win=5, title='predY'})
                disp.image(dispOut:max(5):squeeze(), {win=6, title='predZ'})
                
                disp.image(dispOutGt:max(3):squeeze(), {win=7, title='gtX'})
                disp.image(dispOutGt:max(4):squeeze(), {win=8, title='gtY'})
                disp.image(dispOutGt:max(5):squeeze(), {win=9, title='gtZ'})
                
                disp.image(torch.eq(dispOut,0):max(3):squeeze(), {win=10, title='predX_thresh'})
                disp.image(torch.eq(dispOut,0):max(4):squeeze(), {win=11, title='predY_thresh'})
                disp.image(torch.eq(dispOut,0):max(5):squeeze(), {win=12, title='predZ_thresh'})
            end
            if(params.imsave == 1) then
                vUtils.imsave(dispOut:max(3):squeeze(), params.visDir.. '/predX' .. iter .. 'step'.. forwIter .. '.png')
                vUtils.imsave(dispOut:max(4):squeeze(), params.visDir.. '/predY' .. iter .. 'step'.. forwIter .. '.png')
                vUtils.imsave(dispOut:max(5):squeeze(), params.visDir.. '/predZ' .. iter .. 'step'.. forwIter .. '.png')
            end
            
            if(iter%meshSaveIter == 0) then
                local predParams = netPred.output
                local meshGridInit = primitives.meshGrid({-params.gridBound,-params.gridBound,-params.gridBound},{params.gridBound,params.gridBound,params.gridBound},{params.gridSize,params.gridSize,params.gridSize})
                for b = 1,tsdfGt:size(1) do
                    local visTriSurf = mc.march(tsdfGt[b][1], meshGridInit, 0.000001)
                    mc.writeObj(string.format(params.visMeshesDir .. '/iter%d_inst%d_gt.obj',iter,b), visTriSurf)
                    
                    local pred_b = {}
                    for px = 1,params.nParts do
                        pred_b[px] = {predParams[px][1][b]:clone():double(),predParams[px][2][b]:clone():double(),predParams[px][3][b]:clone():double()}
                    end
                    mUtils.saveParts(pred_b,string.format(params.visMeshesDir .. '/iter%d_inst%d_pred.obj',iter,b), params.primTypes)
                end
            end
        end
    end
    if(iter%10000)==0 then
        torch.save(params.snapshotDir .. '/iter'.. iter .. '.t7', netPred)
    end
    optim.adam(fx, netParameters, optimState)
    errLogger:add({err,errTsdf,errChamfer,errSym})
    for p=1,params.nParts do
        partProbLoggers[p]:add(nUtils.nngraphExtractNode(primitivesTable.modules[p], 'probs'):mean(1):squeeze():totable())
    end
end
fout:close()