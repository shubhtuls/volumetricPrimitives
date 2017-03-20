local util = {}

function montage(img)
  if type(img) == 'table' then
    return arrayMontage(img)
  end

  -- img is a collection?
  if img:dim() == 4 or (img:dim() == 3 and img:size(1) > 3) then
    local images = {}
    for i = 1,img:size(1) do
      images[i] = img[i]
    end
    return arrayMontage(images)
  end
  return img
end

function arrayMontage(images)
  local nperrow = math.ceil(math.sqrt(#images))

  local maxsize = {1, 0, 0}
  for i, img in ipairs(images) do
    --if opts.normalize then
    --  img = normalize(img, opts)
    --end
    if img:dim() == 2 then
      img = torch.expand(img:view(1, img:size(1), img:size(2)), maxsize[1], img:size(1), img:size(2))
    end
    images[i] = img
    maxsize[1] = math.max(maxsize[1], img:size(1))
    maxsize[2] = math.max(maxsize[2], img:size(2))
    maxsize[3] = math.max(maxsize[3], img:size(3))
  end

  -- merge all images onto one big canvas
  local numrows = math.ceil(#images / nperrow)
  local canvas = torch.FloatTensor(maxsize[1], maxsize[2] * numrows, maxsize[3] * nperrow):fill(0.5)
  local row = 0
  local col = 0
  for i, img in ipairs(images) do
    canvas:narrow(2, maxsize[2] * row + 1, img:size(2)):narrow(3, maxsize[3] * col + 1, img:size(3)):copy(img)
    col = col + 1
    if col == nperrow then
      col = 0
      row = row + 1
    end
  end
  return canvas
end

function util.imsave(img,path)
    require 'image'
    local saveIm = montage(img)
    -- print(saveIm:size())
    image.save(path,saveIm)
end

function util.RGBAtoRGB(img)
    local alphaMask = img[4]:repeatTensor(3,1,1)
    local imgOut = torch.cmul(img:clone():narrow(1,1,3),alphaMask) + 1 - alphaMask
    return imgOut
end

return util