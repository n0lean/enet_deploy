#!/usr/bin/env qlua

require 'image'
require 'paths'
require 'os'
require 'lfs'
require 'cunn'
require 'cudnn'

-- Local repo files
local opts = require 'opts'

-- Get the input arguments parsed and stored in opt
local opt = opts.parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')
if opt.dev:lower() == 'cuda' then
   cutorch.setDevice(opt.devID)
   print("GPU # " .. cutorch.getDevice() .. " selected")
end

----------------------------------------
-- Network
local network = {}
network.path = opt.dmodel .. opt.model .. '/model-' .. opt.net .. '.net'
assert(paths.filep(network.path), 'Model not present at ' .. network.path)
print("Loading model from: " .. network.path)

network.model = torch.load(network.path)

-- Convert all the modules in nn from cudnn
if #network.model:findModules('cudnn.SpatialConvolution') > 0 then
   if network.model.__typename == 'nn.DataParallelTable' then
      network.model = network.model:get(1)
   end
end

-- Change model type based on device being used for demonstration
if opt.dev:lower() == 'cpu' then
   cudnn.convert(network.model, nn)
   network.model:float()
else
   network.model:cuda()
end

-- Set the module mode 'train = false'
network.model:evaluate()
network.model:clearState()

-- Get mean and std of the dataset used while training
local stat_file = opt.dmodel .. opt.model .. '/' .. 'stat.t7'
if paths.filep(stat_file) then
   network.stat = torch.load(stat_file)
elseif paths.filep(stat_file .. 'ascii') then
   network.stat = torch.load(stat_file .. '.ascii', 'ascii')
else
   print('No stat file found in directory: ' .. opt.dmodel .. opt.model)
   network.stat = {}
   network.stat.mean = torch.Tensor{0, 0, 0}
   network.stat.std = torch.Tensor{1, 1, 1}
end

-- classes and color based on neural net model used:
local classes

--change target based on categories csv file:
function readCatCSV(filepath)
   print(filepath)
   local file = io.open(filepath, 'r')
   local classes = {}
   local targets = {}
   file:read()    -- throw away first line
   local fline = file:read()
   while fline ~= nil do
      local col1, col2 = fline:match("([^,]+),([^,]+)")
      table.insert(classes, col1)
      table.insert(targets, ('1' == col2))
      fline = file:read()
   end
   return classes, targets
end

-- Load categories from the list of categories generated during training
local newcatdir = opt.dmodel .. opt.model .. '/categories.txt'
if paths.filep(newcatdir) then
   print('Loading categories file from: ' .. newcatdir)
   network.classes, network.targets = readCatCSV(newcatdir)
end

if #network.classes == 0 then
   error('Categories file contains no categories')
end

print('Network has this list of categories, targets:')
for i=1,#network.classes do
   if opt.allcat then network.targets[i] = true end
   print(i..'\t'..network.classes[i]..'\t'..tostring(network.targets[i]))
end

classes = network.classes
--local testout = network.model:forward(torch.Tensor(1,3,256,256))
--print(testout[1]:size(1))

local source = {}
source.w = 1024
source.h = 512

-------------------run main loop-------------------
im_path = 'workspace/cfl.jpg'
lock_path = 'workspace/lock.txt'

function wait(ms)
    local start = os.time()
    repeat until os.time() > start + ms*0.001
end

while true
do
    if paths.filep(im_path) and not(paths.filep(lock_path)) then
	io.open('workspace/lock.txt','w').close()
        -------------load image-------------------
        --prev_size = -1
        --new_size = lfs.attributes(im_path, "size")
        --while prev_size ~= new_size
        --do 
        --    wait(0.01)
        --    prev_size = new_size
        --    new_size = lfs.attributes(im_path, "size")
        --    print(prev_size)
        --    print(new_size)
        --end
        --print(os.time())

        wait(5)
        --print(os.time())
        print("found new " .. im_path .. "!")
        img = image.load(im_path)
	os.remove('workspace/lock.txt')
        --print(img:size())
        --os.remove(im_path)
        
        -----------run model----------------------
        if img:dim() == 3 then
            img = img:view(1, img:size(1), img:size(2), img:size(3))
        end
        local scaledImg = torch.Tensor(1, 3, opt.ratio * img:size(3), opt.ratio * img:size(4))

        if opt.ratio == 1 then
            scaledImg[1] = img[1]
        else
            scaledImg[1] = image.scale(img[1],
                                    opt.ratio * source.w,
                                    opt.ratio * source.h,
                                    'bilinear')
        end

        if opt.dev == 'cuda' then
            scaledImgGPU = scaleImgGPU or torch.CudaTensor(scaledImg:size())
            scaledImgGPU:copy(scaledImg)
            scaledImg = scaledImgGPU
        end

        -- compute network on frame:
        distributions = network.model:forward(scaledImg):squeeze()
        print(distributions:size())

        local sflag = false
        -- Assigning classes to each pixels
        if sflag then
            scores, winners = distributions:max(1)
        else
            _, winners = distributions:max(1)
        end

        if opt.dev == 'cuda' then
            cutorch.synchronize()
            winner = winners:squeeze():float()
            if sflag then
                score = scores:squeeze():float()
            end
        else
            winner = winners:squeeze()
            if sflag then
                score = scores:squeeze()
            end
        end
        print(winner:size())

        -- Confirming whether rescaling is even necessary or not
        if opt.ratio * source.h ~= winner:size(1) or
            opt.ratio * source.w ~= winner:size(2) then
            winner = image.scale(winner:float(),
                              opt.ratio * source.w,
                              opt.ratio * source.h,
                              'simple')
            if sflag then
                score = image.scale(winner:float(),
                              opt.ratio * source.w,
                              opt.ratio * source.h,
                              'simple')
            end

        end
        thresh = 10
        vector = torch.Tensor(1, winner:size(2))
        for i=1,vector:size(2) do
           vector[1][i] = 0
           for j=thresh, winner:size(1) do
               id = winner:size(1)-j
               if winner[id][i] ~= 2 then
                   vector[1][i] = id
                   break
               end
           end 
        end
        ------------------saving result------------------------------
        --local out1 = assert(io.open("workspace/winner.csv", "w"))
	io.open('workspace/lock_vector_write.txt','w').close()
        local out3 = assert(io.open("workspace/vector.csv", "w"))
	os.remove('workspace/lock_vector_write.txt')
        --if sflag then
        --    local out2 = assert(io.open("workspace/score.csv", "w"))
        --end
        splitter = "	"
        -- write vector.csv
        for i=1,vector:size(1) do
            for j=1,vector:size(2) do
                out3:write(vector[i][j])
                if j==vector:size(2) then
                    out3:write("\n")
                end
                out3:write(splitter)
            end
        end
        out3:close() 
--[==[
        for i=1,winner:size(1) do
            for j=1,winner:size(2) do
                out1:write(winner[i][j])
                if sflag then
                    out2:write(math.floor(score[i][j]))
                end
                if j == winner:size(2) then
                    out1:write("\n")
                    if sflag then
                        out2:write("\n")
                    end
                else
                    out1:write(splitter)
                    if sflag then
                        out2:write(splitter)
                    end
                end
            end
        end
        out1.close() 
        if sflag then
            out2.close() 
        end
--]==]
        os.execute('mv ' .. im_path .. ' workspace/img.jpg')
        os.execute('python workspace/visualize.py')
        wait(10)
        --image.save('workspace/cfl.png', winner, 1, 'byte')
    end
end
