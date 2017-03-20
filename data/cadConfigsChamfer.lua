require 'nn'
local matio = require 'matio'
local primitives = dofile('../modules/primitives.lua')
local M = {}

-------------------------------
-------------------------------
--print(modelNames:size())
local function BuildArray(...)
  local arr = {}
  for v in ... do
    arr[#arr + 1] = v
  end
  return arr
end

local SimpleCad = {}
SimpleCad.__index = SimpleCad

setmetatable(SimpleCad, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function SimpleCad.new(params)
    local self = setmetatable({}, SimpleCad)
    self.gridSize = params.gridSize
    self.modelsDir = params.modelsDataDir
    self.modelIter = params.modelIter
    self.batchSize = params.batchSize
    self.modelSize = params.gridSize
    self.nSamplePoints = params.nSamplePoints
    self.gridBound = params.gridBound
    
    self.iter = 0 -- we'll load new models everytime mod(iter,modelIter)==0
    self.startModelIndex = 0
    self.modelNames = BuildArray(paths.files(self.modelsDir , '.mat'))
    table.sort(self.modelNames)
    
    self.loadedVoxels = torch.Tensor(self.batchSize, 1, self.modelSize, self.modelSize, self.modelSize):fill(0)
    self.loadedTsdfs = torch.Tensor(self.batchSize, 1, self.modelSize, self.modelSize, self.modelSize):fill(0)
    self.loadedCPs = torch.Tensor(self.batchSize, 1, self.modelSize, self.modelSize, self.modelSize, 3):fill(0)
    self.loadedSurfaceSamples = {}
    
    local gridMin = -params.gridBound + params.gridBound/params.gridSize
    local gridMax = params.gridBound - params.gridBound/params.gridSize
    local meshGridInit = primitives.meshGrid({gridMin,gridMin,gridMin},{gridMax,gridMax,gridMax},{params.gridSize,params.gridSize,params.gridSize})
    
    local meshGrid = meshGridInit:repeatTensor(params.batchSize,1,1,1)
    self.gridPoints = meshGrid:reshape(params.batchSize,params.gridSize^3,3):clone()
    
    return self
end

function SimpleCad:reloadShapes()
    for ix = 1,self.batchSize do
        self.startModelIndex = torch.random(1,#(self.modelNames))
        local shape = matio.load(paths.concat(self.modelsDir,self.modelNames[self.startModelIndex]),{'Volume','tsdf','surfaceSamples','closestPoints'})
        self.loadedVoxels[ix][1]:copy(shape.Volume:typeAs(self.loadedVoxels))
        self.loadedTsdfs[ix][1]:copy(shape.tsdf:typeAs(self.loadedTsdfs))
        self.loadedCPs[ix][1]:copy(shape.closestPoints:typeAs(self.loadedCPs))
        self.loadedSurfaceSamples[ix] = shape.surfaceSamples:clone()
    end
    self.loadedShapes = self.loadedVoxels:clone()
end

function SimpleCad:forward()
    if(self.iter % self.modelIter == 0) then
        self:reloadShapes()
    end
    self.iter = self.iter+1
    
    local output
    local outSampleTsfds = torch.Tensor(self.batchSize, self.nSamplePoints):fill(0)
    local outSamplePoints = torch.Tensor(self.batchSize, self.nSamplePoints, 3)
    
    for b=1,self.batchSize do
        local nPointsTot = self.loadedSurfaceSamples[b]:size(1)
        for ns = 1,self.nSamplePoints do
            local pId = torch.random(nPointsTot)
            outSamplePoints[b][ns] = self.loadedSurfaceSamples[b][pId]
        end
    end
    --print(outSamplePoints:min(), outSamplePoints:max())
    output = {self.loadedShapes:clone(), outSampleTsfds, outSamplePoints}
        
    return output
end

-- Useful for visualization
function SimpleCad:forwardTest()
    if(self.iter % self.modelIter == 0) then
        self:reloadShapes()
    end
    self.iter = self.iter+1
    local outTsfds = self.loadedTsdfs:reshape(self.batchSize, self.gridSize^3)
    local outPoints = self.gridPoints:clone()
    local output = {self.loadedShapes, outTsfds, outPoints}
    return output
end

-- queryPoints is bs X nQ X 3
function SimpleCad:chamferForward(queryPoints)
    local qp = queryPoints:double()
    --print(qp[1])
    local neighborInds = self:pointClosestCellIndex(qp)
    local queryDiffs = torch.Tensor(queryPoints:size()):fill(0)
    
    for b=1,queryPoints:size(1) do
        for np = 1,queryPoints:size(2) do
            local ind = neighborInds[b][np]
            --if(np==1) then print(qp[b][np], ind) end
            if(self.loadedVoxels[b][1][ind[1]][ind[2]][ind[3]] == 0) then
                local cp = self.loadedCPs[b][1][ind[1]][ind[2]][ind[3]]
                queryDiffs[b][np]:copy(qp[b][np] - cp)
            end
        end
    end
        
    local outDists = queryDiffs:clone():pow(2):sum(3)
    self.queryDiffs = queryDiffs
    
    return outDists:clone():typeAs(queryPoints)
end

function SimpleCad:chamferBackward(queryPoints)
    return self.queryDiffs:typeAs(queryPoints)
end


function SimpleCad:pointClosestCellIndex(points)
    local gridMin = -self.gridBound + self.gridBound/self.gridSize
    local gridMax = self.gridBound - self.gridBound/self.gridSize
    
    local inds = (points - gridMin)*self.gridSize/(2*self.gridBound) + 1
    inds = torch.round(torch.cmin(torch.cmax(inds, 1), self.gridSize))
    return inds
end

-------------------------------
-------------------------------

M.SimpleCad = SimpleCad
return M
