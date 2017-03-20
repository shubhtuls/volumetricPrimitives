require 'nn'
------------------------------------------------------------------------
--[[ ReinforceCategorical ]]-- 
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Inputs are a B X K tensors of categorical prob : (p[1], p[2], ..., p[k]) 
-- Ouputs are samples drawn from this distribution.
-- Uses the REINFORCE algorithm (ref. A sec 6. p.230-236) which is 
-- gradOutputs are ignored (REINFORCE algorithm).
------------------------------------------------------------------------
local ReinforceCategorical, parent = torch.class("nn.ReinforceCategorical", "nn.Module")

function ReinforceCategorical:__init(bMomentum, entropyWt, intrinsicReward)
    parent.__init(self)
    -- true makes it stochastic during evaluation and training
    -- false makes it stochastic only during training
    self.baseline = 0
    self.intrinsicReward = (intrinsicReward~=nil) and intrinsicReward:clone() or nil
    self.entropyWt = entropyWt or 0
    self.bMomentum = bMomentum
    self.testMode = false
end

function ReinforceCategorical:reinforce(rewards)
    assert(rewards:size(1) == self.output:size(1))
    self.baseline = self.bMomentum*self.baseline + (1-self.bMomentum)*rewards:mean()
    self.rewards = rewards - self.baseline
end

function ReinforceCategorical:updateOutput(input)
    --print(input)
    self.output = torch.Tensor(input:size()):fill(0)
    self.argInds = {}
    self.argProbs = {}
    local bs = input:size(1)
    local nChoices = input:size(2)
    for b=1,bs do
        local sumProb = torch.uniform(0,1)
        --print(sumProb)
        local weight = 0
        if(self.testMode) then
            for n=1,nChoices do
                if(weight < input[b][n]) then
                    --print(n)
                    weight = input[b][n]
                    self.argProbs[b] = input[b][n] + 1e-8
                    self.argInds[b] = n
                    --self.output[b][n] = 1
                    --print(sumProb,n)
                end
            end
            self.output[b][self.argInds[b]] = 1
        else
            -- default choice is first one
            self.argProbs[b] = input[b][1] + 1e-8
            self.argInds[b] = 1

            for n=1,nChoices do
                weight = weight + input[b][n]
                if(weight >= sumProb) then
                    --print(n)
                    self.argProbs[b] = input[b][n] + 1e-8
                    self.argInds[b] = n
                    self.output[b][n] = 1
                    --print(sumProb,n)
                    break
                end
            end
        end
    end

    --self.output = torch.Tensor(input:size()):fill(0)
    --self.output:narrow(2,1,1):fill(1)
    
    self.output = self.output:typeAs(input)
    return self.output
end

function ReinforceCategorical:updateGradInput(input, gradOutput)
    local bs = input:size(1)
    local nChoices = input:size(2)
    local gradInput = torch.Tensor(bs,nChoices):fill(0):typeAs(gradOutput)
    for b=1,bs do
        local n = self.argInds[b]
        if(self.rewards[b]==nil) then
            print("Incorrect rewards")
        elseif(self.argProbs[b]==nil) then
            print("Incorrect prob")
        end
        --if(n==1) then self.rewards[b] = -10 end --just to debug
        gradInput[b][n] = (self.rewards[b] + ((self.intrinsicReward~=nil) and self.intrinsicReward[n] or 0))/self.argProbs[b]
    end
    gradInput = gradInput:typeAs(input)
    if(self.entropyWt ~= 0) then
        local meanProb = input:mean(1)
        --print(torch.log(meanProb))
        local entropyGrad = 1 + torch.log(meanProb)
        entropyGrad = torch.repeatTensor(entropyGrad,bs,1)
        gradInput = gradInput + entropyGrad*self.entropyWt*(1/bs)
    end
    --print(input:mean(1), gradInput:mean(1))
    --print(gradInput)
    self.gradInput = gradInput
    
    --self.gradInput:fill(0)
    
    return self.gradInput
end
