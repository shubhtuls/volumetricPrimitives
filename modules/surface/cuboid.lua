require 'nn'
require 'nngraph'
------------------------------------------------------------------------
local CuboidSurface, parent = torch.class("nn.CuboidSurface", "nn.Module")

local function cuboidAreaModule(nSamples)
    -- Module for surface area
    local dims = - nn.Identity()
    local width = dims - nn.Narrow(2,1,1)
    local height = dims - nn.Narrow(2,2,1)
    local depth = dims - nn.Narrow(2,3,1)
    local wh = {width, height} - nn.CMulTable()
    local hd = {height, depth} - nn.CMulTable()
    local wd = {width, depth} - nn.CMulTable()
    local surfArea = {wh, hd, wd} - nn.CAddTable() - nn.MulConstant(2)
    local areaRep = surfArea - nn.Replicate(nSamples,2)
    local gmod = nn.gModule({dims}, {areaRep})
    return gmod
end

local function cuboidL1Module(nSamples)
    -- Module for surface area
    local dims = - nn.Identity()
    local width = dims - nn.Narrow(2,1,1)
    local height = dims - nn.Narrow(2,2,1)
    local depth = dims - nn.Narrow(2,3,1)
    local volume = {width, height, depth} - nn.CAddTable()
    local volRep = volume - nn.Replicate(nSamples,2)
    local gmod = nn.gModule({dims}, {volRep})
    return gmod
end

local function cuboidVolumeModule(nSamples)
    -- Module for surface area
    local dims = - nn.Identity()
    local width = dims - nn.Narrow(2,1,1)
    local height = dims - nn.Narrow(2,2,1)
    local depth = dims - nn.Narrow(2,3,1)
    local volume = {width, height, depth} - nn.CMulTable()
    local volRep = volume - nn.Replicate(nSamples,2)
    local gmod = nn.gModule({dims}, {volRep})
    return gmod
end

local function sampleWtModule(nSamplesPerFace, normFactor)
    -- Module for surface area
    local dims = - nn.Identity()
    local area
    if(normFactor == 'Vol') then
        area = dims - cuboidVolumeModule(nSamplesPerFace*3)
    elseif(normFactor == 'Surf') then
        area = dims - cuboidAreaModule(nSamplesPerFace*3)
    elseif(normFactor == 'L1') then
        area = dims - cuboidL1Module(nSamplesPerFace*3)
    end
    
    local dimsInv = dims - nn.Power(-1)
    local dimsInvNorm = dimsInv - nn.Sum(2) - nn.Replicate(3,2)
    local normWeights = {dimsInv, dimsInvNorm} - nn.CDivTable() - nn.MulConstant(3)
    
    local widthWt = normWeights - nn.Narrow(2,1,1) - nn.Replicate(nSamplesPerFace,2)
    local heightWt = normWeights - nn.Narrow(2,2,1) - nn.Replicate(nSamplesPerFace,2)
    local depthWt = normWeights - nn.Narrow(2,3,1) - nn.Replicate(nSamplesPerFace,2)

    local sampleWt = {widthWt, heightWt, depthWt} - nn.JoinTable(2)
    local finalWt
    
    if(normFactor == 'None') then
        --print('NormFactor : None')
        finalWt = sampleWt
    else
        finalWt = {sampleWt, area} - nn.CMulTable() - nn.MulConstant(1/nSamplesPerFace)
    end
    
    local gmod = nn.gModule({dims}, {finalWt})
    return gmod
end

function CuboidSurface:__init(nSamples, normFactor)
    parent.__init(self)
    self.nSamples = nSamples
    assert(nSamples%3 == 0)
    self.samplesPerFace = nSamples/3
    self.normFactor = normFactor or 'None'
    self.weightModule = sampleWtModule(self.samplesPerFace, self.normFactor)
    
    local dims = - nn.Identity()
    local coeffs = - nn.Identity()
    local dimsRep = dims - nn.Replicate(nSamples,2)
    local samples = {dimsRep, coeffs} - nn.CMulTable()
    self.sampleModule = nn.gModule({dims, coeffs}, {samples})
end

-- input is B X 3, output is {B X nSamples X 3, B X nSamples}
function CuboidSurface:updateOutput(input)
    local importanceWeight = self.weightModule:forward(input)
    local bs = input:size(1)
    local ns = self.nSamples
    local nsp = self.samplesPerFace
    
    local coeffBernoulli = torch.Tensor(bs, nsp, 3):typeAs(input):bernoulli()
    coeffBernoulli = 2*coeffBernoulli - 1
    
    local coeff_w = torch.Tensor(bs, nsp, 3):typeAs(input):uniform(-1,1)
    coeff_w:narrow(3,1,1):copy(coeffBernoulli:narrow(3,1,1):clone()) -- coeff_w becomes ({-1,1}, unif, unif)
    
    local coeff_h = torch.Tensor(bs, nsp, 3):typeAs(input):uniform(-1,1)
    coeff_h:narrow(3,2,1):copy(coeffBernoulli:narrow(3,2,1):clone()) -- coeff_h becomes (unif, {-1,1}, unif)
    
    local coeff_d = torch.Tensor(bs, nsp, 3):typeAs(input):uniform(-1,1)
    coeff_d:narrow(3,3,1):copy(coeffBernoulli:narrow(3,3,1):clone()) -- coeff_d becomes (unif, unif, {-1,1})
        
    self.coeffs = torch.cat(torch.cat(coeff_w,coeff_h,2),coeff_d,2)
    
    local samples = self.sampleModule:forward({input, self.coeffs})
    self.output = {samples, importanceWeight}
    return self.output
end

-- input is B X 3
function CuboidSurface:updateGradInput(input, gradOutput)
    local g1 = self.sampleModule:updateGradInput({input,self.coeffs}, gradOutput[1])
    local g2 = self.weightModule:updateGradInput(input, gradOutput[2])
    self.gradInput = g1[1] + g2
    return self.gradInput
end
