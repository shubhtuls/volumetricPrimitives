require 'nn'
require 'nngraph'
------------------------------------------------------------------------
local NullSurface, parent = torch.class("nn.NullSurface", "nn.Module")

function NullSurface:__init(nSamples)
    parent.__init(self)
    self.nSamples = nSamples
end

-- input is B X 1, output is {B X nSamples X 3, B X nSamples}
function NullSurface:updateOutput(input)
    local bs = input:size(1)
    local ns = self.nSamples
    local importanceWeight = torch.Tensor(bs, ns):fill(0):typeAs(input)
    local samples = torch.Tensor(bs, ns, 3):fill(0):typeAs(input)
    self.output = {samples, importanceWeight}
    return self.output
end

-- input is B X 2
function NullSurface:updateGradInput(input, gradOutput)
    self.gradInput = torch.Tensor(input:size()):typeAs(input):fill(0)
    return self.gradInput
end