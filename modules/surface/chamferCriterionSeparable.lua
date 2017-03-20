require 'optim'
require 'cunn'
local M = {}

-------------------------------
-------------------------------
local ChamferCriterion = {}
ChamferCriterion.__index = ChamferCriterion

setmetatable(ChamferCriterion, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function ChamferCriterion.new(pointSamplerModule, nSamples)
    local self = setmetatable({}, ChamferCriterion)
    self.pointSamplerModule = pointSamplerModule
    self.nSamples = nSamples
    --
    local impWeights = - nn.Identity()
    local totWeights = impWeights - nn.Sum(2) - nn.AddConstant(1e-6) - nn.Replicate(nSamples,2) --constant handles cases where all parts are null parts
    local normWeights = {impWeights, totWeights} - nn.CDivTable()
    self.normWeightsModule = nn.gModule({impWeights}, {normWeights})
    
    return self
end

function ChamferCriterion:cuda()
    self.useGpu = true
    self.pointSamplerModule = self.pointSamplerModule:cuda()
    self.normWeightsModule = self.normWeightsModule:cuda()
end

function ChamferCriterion:forward(input, dataLoader)
    self.samples, self.impWeights = unpack(self.pointSamplerModule:forward(input))
    self.tsdfLosses = dataLoader:chamferForward(self.samples)
    self.normWeights = self.normWeightsModule:forward(self.impWeights)
    --print(self.normWeights:sum(2):mean(1))
    
    self.sampleWeights = self.normWeights:clone():expand(self.samples:size())
    self.weightedLosses = self.tsdfLosses:clone():cmul(self.normWeights)
    
    return self.weightedLosses:sum(2):mean()
    
end

function ChamferCriterion:backward(input, dataLoader)
    local normalizationFactor
    normalizationFactor = (1/self.samples:size(1))
    
    local gradNorm = self.tsdfLosses*normalizationFactor
    local gradImp = self.normWeightsModule:updateGradInput(self.impWeights, gradNorm)
    --gradImp:fill(0)
    
    local gradSamples = dataLoader:chamferBackward(self.samples):clone()
    gradSamples:cmul(self.sampleWeights*normalizationFactor)
    --print(self.samples[1], self.impWeights[1], self.normWeights[1],gradSamples[1], gradImp[1])
    
    self.gradInput = self.pointSamplerModule:updateGradInput(input, {gradSamples, gradImp})
    return self.gradInput
end
-------------------------------
-------------------------------
M.ChamferCriterion = ChamferCriterion
return M