local intersectionUtil = dofile('../modules/intersection.lua')
local primitives = dofile('../modules/primitives.lua')
local transformer = dofile('../modules/transformer.lua')

local M = {}
-------------------
-------------------
local function removeRedundantParts(predParams, partVols, primTypes)
    local removalIntersectionThresh = 0.6
    local predParams = predParams
    local partVols = partVols
    local partIntersections = intersectionUtil.intraPrimitiveIntersection(predParams, primTypes)
    while(partIntersections:max() > removalIntersectionThresh) do
        local _, maxInd = torch.max(partIntersections,1)
        maxInd = maxInd[1]
        intersectionUtil.setNull(predParams[maxInd], params.primTypes)
        partVols[maxInd] = 0
        partIntersections = intersectionUtil.intraPrimitiveIntersection(predParams, primTypes)
    end
    return predParams, partVols
end
-------------------
-------------------
local function meshFaceInds(predParams, partVols, meshVar, primTypes)
    local nSperFace = 20
    local nV = meshVar.vertices:size(1)
    local nF = meshVar.faces:size(1)
    
    local function pgenFuncTsdf(nP)
        return primitives.primitiveSelector(primTypes, nP)
    end
    
    local tsdfFunc = transformer.partComposition(pgenFuncTsdf, 1, nSperFace*nF)
    local ptsQuery = torch.Tensor(nSperFace*nF,3):fill(0)
    for f = 1,nF do
        local v1 = meshVar.vertices[meshVar.faces[f][1]]
        local v2 = meshVar.vertices[meshVar.faces[f][2]]
        local v3 = meshVar.vertices[meshVar.faces[f][3]]
        for ns = 1,nSperFace do
            --formula from https://math.stackexchange.com/questions/18686/uniform-random-point-
            local r1 = torch.sqrt(torch.uniform(0,1))
            local r2 = torch.uniform(0,1)
            local pt = (1-r1)*v1 + r1*(1-r2)*v2 + r1*r2*v3
            ptsQuery[(f-1)*nSperFace + ns]:copy(pt)
        end
    end

    local tsdf_pts = torch.Tensor(nSperFace*nF,1):fill(10)
    local vol_pts = torch.Tensor(nSperFace*nF,1):fill(0)
    local partInds = torch.Tensor(nSperFace*nF,1):fill(1)
    local faceInds = torch.Tensor(nF):fill(1)
    local ptsQuery = ptsQuery:reshape(1,nSperFace*nF,3)

    for px = 1,#predParams do
        if(partVols[px] > 0) then
            tsdf_p = tsdfFunc:forward({ptsQuery, {predParams[px]}}):reshape(tsdf_pts:size()):clone()
            thisPointsMin = torch.gt(tsdf_pts,tsdf_p):double()
            thisPointsTied = torch.eq(tsdf_pts,tsdf_p):double():cmul(torch.lt(vol_pts, partVols[px]):double())
            thisPoints = thisPointsMin + thisPointsTied
            tsdf_pts = torch.min(torch.cat(tsdf_pts,tsdf_p,2),2)
            vol_pts = vol_pts:clone():cmul(1-thisPoints) + thisPoints*partVols[px]
            partInds = partInds:clone():cmul(1-thisPoints) + thisPoints*px
        end
    end
    for f = 1,nF do
        local sampleInds = partInds:narrow(1,(f-1)*nSperFace+1,nSperFace) --if(f < 3) then print(sampleInds) end
        faceInds[f] = torch.mode(sampleInds:squeeze())
    end
    
    return faceInds

end
-------------------
-------------------
M.removeRedundantParts = removeRedundantParts
M.meshFaceInds = meshFaceInds
return M