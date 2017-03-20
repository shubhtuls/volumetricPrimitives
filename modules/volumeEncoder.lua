require 'nn'
local M = {}

function M.convEncoderSimple3d(nLayers, nChannelsInit, nInputChannels, useBn)
    local nInputChannels = nInputChannels or 1
    local nChannelsInit = nChannelsInit or 8
    local useBn = (useBn~=false) and true
    local nOutputChannels = nChannelsInit
    local encoder = nn.Sequential()
    
    for l=1,nLayers do
        encoder:add(nn.VolumetricConvolution(nInputChannels, nOutputChannels, 3, 3, 3 , 1, 1, 1, 1, 1, 1))
        if useBn then encoder:add(nn.VolumetricBatchNormalization(nOutputChannels)) end
        encoder:add(nn.LeakyReLU(0.2, true))
        encoder:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels*2
    end
    return encoder, nOutputChannels/2 -- division by two offsets the mutiplication in last iteration
end

function M.convEncoderSimple2d(nLayers, nChannelsInit, nInputChannels, useBn)
    local nInputChannels = nInputChannels or 1
    local nChannelsInit = nChannelsInit or 8
    local useBn = (useBn~=false) and true
    local nOutputChannels = nChannelsInit
    local encoder = nn.Sequential()
    
    for l=1,nLayers do
        encoder:add(nn.SpatialConvolution(nInputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then encoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        encoder:add(nn.LeakyReLU(0.2, true))
        encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels*2
    end
    encoder:add(nn.Unsqueeze(5))
    return encoder, nOutputChannels/2 -- division by two offsets the mutiplication in last iteration
end

function M.convEncoderComplex2d(nLayers, nChannelsInit, nInputChannels, useBn)
    local nInputChannels = nInputChannels or 1
    local nChannelsInit = nChannelsInit or 8
    local useBn = useBn or true
    local nOutputChannels = nChannelsInit
    local encoder = nn.Sequential()
    
    for l=1,nLayers do
        encoder:add(nn.SpatialConvolution(nInputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then encoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        encoder:add(nn.LeakyReLU(0.2, true))
        
        encoder:add(nn.SpatialConvolution(nOutputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then encoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        encoder:add(nn.LeakyReLU(0.2, true))
        
        encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels*2
    end
    encoder:add(nn.Unsqueeze(5))
    return encoder, nOutputChannels/2 -- division by two offsets the mutiplication in last iteration
end

function M.convDecoderSimple3d(nLayers, nInputChannels, ndf, nFinalChannels, useBn, normalizeOut)
    --adds nLayers deconv layers + 1 conv layer
    local nFinalChannels = nFinalChannels or 1
    local ndf = ndf or 8 --channels in penultimate layer
    local useBn = useBn~=false and true
    local normalizeOut = normalizeOut~=false and true
    local nOutputChannels = ndf*torch.pow(2,nLayers-1)
    local decoder = nn.Sequential()
    for l=1,nLayers do
        decoder:add(nn.VolumetricFullConvolution(nInputChannels, nOutputChannels, 4, 4, 4, 2, 2, 2, 1, 1, 1))
        if useBn then decoder:add(nn.VolumetricBatchNormalization(nOutputChannels)) end
        decoder:add(nn.ReLU(true))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels/2
    end
    decoder:add(nn.VolumetricConvolution(ndf, nFinalChannels, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    if(normalizeOut) then
        decoder:add(nn.Tanh()):add(nn.AddConstant(1)):add(nn.MulConstant(0.5))
    end
    return decoder
end

function M.deconvDecoderSimple3d(nz, ndf)
    local nc=1
    local netD = nn.Sequential():add(nn.Squeeze())
    netD:add(nn.Linear(nz, 1024))
    netD:add(nn.BatchNormalization(1024)):add(nn.ReLU(true))
    netD:add(nn.Linear(1024, 4*4*4*ndf*8))
    netD:add(nn.View(ndf*8, 4, 4, 4))  
    netD:add(nn.VolumetricBatchNormalization(ndf*8)):add(nn.ReLU(true))

    -- state size: (ngf*8) x 4 x 4 x 4
    netD:add(nn.VolumetricFullConvolution(ndf * 8, ndf * 4, 4, 4, 4, 2, 2, 2, 1, 1, 1))
    netD:add(nn.VolumetricBatchNormalization(ndf * 4)):add(nn.ReLU(true))
    -- state size: (ngf*4) x 8 x 8 x 8 
    netD:add(nn.VolumetricFullConvolution(ndf * 4, ndf * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
    netD:add(nn.VolumetricBatchNormalization(ndf * 2)):add(nn.ReLU(true))
    -- state size: (ngf*2) x 16 x 16 x 16
    netD:add(nn.VolumetricFullConvolution(ndf * 2, nc, 4, 4, 4, 2, 2, 2, 1, 1, 1))
    netD:add(nn.Tanh()):add(nn.AddConstant(1)):add(nn.MulConstant(0.5))
    return netD
end

return M