require 'nn'
require 'nngraph'
local primitives = dofile('../modules/primitives.lua')
local nUtils = dofile('../modules/netUtils.lua')
local M = {}
-------------------------------
-------------------------------
local function primitiveModule(key, nP)
    if(key == 'Cu') then
        return primitives.cuboidInterior(nP), 3
    elseif(key=='Nu') then
        return primitives.nullInterior(nP), 1
    else
        error("invalid input")
    end
end

-------------------------------
-------------------------------
-- input is BXnPX3 points, BXnZ part params
-- output is BXnP tsdf^2 values
-- Values are negative for interior, zero for exterior
local function primitiveSelector(primitives, nP)
    if(#primitives == 1) then
        local pMod, _ = primitiveModule(primitives[1], nP)
        return nn.Sequential():add(pMod):add(nn.MulConstant(-1))
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
    local tsdfSq = {tsdfAll, sampler} - nn.CMulTable() - nn.Sum(3) - nn.MulConstant(-1)
    
    local gmod = nn.gModule({points, dims}, {tsdfSq})
    return gmod
end

-------------------------------
-------------------------------

M.primitiveSelector = primitiveSelector
return M