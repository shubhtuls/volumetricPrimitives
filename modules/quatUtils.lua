require 'nn'
require 'nngraph'
local M = {}
-------------------------------
-------------------------------
-- input is {NXPX4, NXPX4}. output is {NXPX4}
local function HamiltonProductModule()
    local q1 = -nn.Identity()
    local q2 = -nn.Identity()
    
    local inds = torch.Tensor({
            1,-2,-3,-4,
            2,1,4,-3,
            3,-4,1,2,
            4,3,-2,1
        }):reshape(4,4)
    local sign = inds:clone():sign()
    inds = inds:clone():abs()
    
    local q1_q2_prods = {}
    
    for d=1,4 do
        local q2_v1 = q2 - nn.Narrow(3,inds[d][1],1) - nn.MulConstant(sign[d][1], false)
        local q2_v2 = q2 - nn.Narrow(3,inds[d][2],1) - nn.MulConstant(sign[d][2], false)
        local q2_v3 = q2 - nn.Narrow(3,inds[d][3],1) - nn.MulConstant(sign[d][3], false)
        local q2_v4 = q2 - nn.Narrow(3,inds[d][4],1) - nn.MulConstant(sign[d][4], false)
        local q2Sel = {q2_v1, q2_v2, q2_v3, q2_v4} - nn.JoinTable(3)
        q1_q2_prods[d] = {q1, q2Sel} - nn.CMulTable() - nn.Sum(3) - nn.Unsqueeze(3)
    end
    
    local qMult = q1_q2_prods - nn.JoinTable(3)
    local gmod = nn.gModule({q1, q2}, {qMult})
    return gmod
end
-------------------------------
-------------------------------
-- input is BXPX4 quaternions, output is also BXPX4 quaternions
local function quatConjugateModule()
    local split = nn.ConcatTable():add(nn.Narrow(3,1,1)):add(nn.Narrow(3,2,3))
    local mult = nn.ParallelTable():add(nn.Identity()):add(nn.MulConstant(-1,false))
    local qc = nn.Sequential():add(split):add(mult):add(nn.JoinTable(3))
    return qc
end
-------------------------------
-------------------------------
-- input is {BXPX4 vectors, BXPX4 quaternions} output is BXPX3 rotated vectors
-- input vectors have 'real' dimension = 0
local function quatRotateModule()
    local quatIn = - nn.Identity()
    local quat = quatIn - nn.Contiguous()
    local vec = nn.Identity()()
    
    local quatConj = quatConjugateModule()(quat)
    local mult = HamiltonProductModule()({HamiltonProductModule()({quat,vec}),quatConj})
    local truncate = nn.Narrow(3,2,3)(mult)
    local gmod = nn.gModule({vec, quatIn}, {truncate})
    return gmod
end
-------------------------------
-------------------------------
M.quatConjugateModule = quatConjugateModule
M.quatRotateModule = quatRotateModule
M.HamiltonProductModule = HamiltonProductModule
return M
