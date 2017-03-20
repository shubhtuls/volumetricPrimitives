-- https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
require 'nn'

local HamiltonProduct, parent = torch.class('nn.HamiltonProduct', 'nn.Module')

function HamiltonProduct:__init()
    parent.__init(self)
    self.gradInput = {}
end

-- inputs are {q1List, q2List} where each is of form B X _ X  .. X 4
-- we reshape these to B' X 4
function HamiltonProduct:updateOutput(input)
    local q1List = input[1]:view(-1,4)
    local q2List = input[2]:view(-1,4)
    local output = torch.Tensor(q1List:size()):typeAs(q1List):fill(0)
    
    for ix = 1,output:size(1) do
        if(ix%10000)==0 then
            print(ix,output:size(1))
        end
        local q1 = q1List[ix]
        local q2 = q2List[ix]
        output[ix][1] = q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3] - q1[4]*q2[4]
        output[ix][2] = q1[1]*q2[2] + q1[2]*q2[1] + q1[3]*q2[4] - q1[4]*q2[3]
        output[ix][3] = q1[1]*q2[3] - q1[2]*q2[4] + q1[3]*q2[1] + q1[4]*q2[2]
        output[ix][4] = q1[1]*q2[4] + q1[2]*q2[3] - q1[3]*q2[2] + q1[4]*q2[1]
    end
    
    self.output = output:view(input[1]:size()):clone()
    return self.output
end

function HamiltonProduct:updateGradInput(input, _gradOutput)
    local q1List = input[1]:view(-1,4)
    local q2List = input[2]:view(-1,4)
    local gradOutput = _gradOutput:view(-1,4)
    gradQ1 = torch.Tensor(q1List:size()):typeAs(q1List):fill(0)
    gradQ2 = torch.Tensor(q2List:size()):typeAs(q2List):fill(0)
    
    for ix = 1,gradQ1:size(1) do
        local q1 = q1List[ix]
        local q2 = q2List[ix]
        local gradQ = gradOutput[ix]
        gradQ1[ix][1] = gradQ[1]*q2[1] + gradQ[2]*q2[2] + gradQ[3]*q2[3] + gradQ[4]*q2[4]
        gradQ1[ix][2] = -gradQ[1]*q2[2] + gradQ[2]*q2[1] - gradQ[3]*q2[4] + gradQ[4]*q2[3]
        gradQ1[ix][3] = -gradQ[1]*q2[3] + gradQ[2]*q2[4] + gradQ[3]*q2[1] - gradQ[4]*q2[2]
        gradQ1[ix][4] = -gradQ[1]*q2[4] - gradQ[2]*q2[3] + gradQ[3]*q2[2] + gradQ[4]*q2[1]
        
        gradQ2[ix][1] = gradQ[1]*q1[1] + gradQ[2]*q1[2] + gradQ[3]*q1[3] + gradQ[4]*q1[4]
        gradQ2[ix][2] = -gradQ[1]*q1[2] + gradQ[2]*q1[1] + gradQ[3]*q1[4] - gradQ[4]*q1[3]
        gradQ2[ix][3] = -gradQ[1]*q1[3] - gradQ[2]*q1[4] + gradQ[3]*q1[1] + gradQ[4]*q1[2]
        gradQ2[ix][4] = -gradQ[1]*q1[4] + gradQ[2]*q1[3] - gradQ[3]*q1[2] + gradQ[4]*q1[1]
    end
    
    self.gradInput = {gradQ1:view(input[1]:size()):clone(), gradQ2:view(input[2]:size()):clone()}
    return self.gradInput
end