

require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local trainFile = opt.datapath .. '/train.txt'
local testFile = opt.datapath .. '/test.txt'

----------------------------------------------------------------------

local classes = { 'Nuclei', 'Background'}

local conClasses = {'Nuclei', 'Background'}

print('==> number of classes: ' .. #classes ..', classes: ', classes)

----------------------------------------------------------------------
-- saving training histogram of classes
local histClasses = torch.Tensor(#classes):fill(0)

local trainData
local testData

-- Function to read txt file and return image and ground truth path
function getPath(filepath)
   print("Extracting file names from: " .. filepath)
   local file = io.open(filepath, 'r')
   local imgPath = {}
   local gtPath = {}
   file:read()    -- throw away first line
   local fline = file:read()
   while fline ~= nil do
      local col1, col2 = fline:match("([^,]+),([^,]+)")
      col1 = opt.datapath .. col1
      col2 = opt.datapath .. col2
      table.insert(imgPath, col1)
      table.insert(gtPath, col2)
      fline = file:read()
   end
   return imgPath, gtPath
end

----------------------------------------------------------------------

   ----------------------------------------------------------------------
   -- Acquire image and ground truth paths for training set
   local imgPath, gtPath = getPath(trainFile)

   -- initialize data structures:
   trainData = {
      data = torch.FloatTensor(#imgPath, opt.channels, opt.imHeight , opt.imWidth),
      labels = torch.FloatTensor(#imgPath, opt.labelHeight , opt.labelWidth),
      preverror = 1e10, -- a really huge number
      size = function() return trainData.data:size(1) end
   }

   print "==> Loading traning data"
   for i = 1, #imgPath do
      -- load original image
      local rawImg = image.load(imgPath[i])

      if (opt.imHeight == rawImg:size(2)) and
         (opt.imWidth == rawImg:size(3)) then
         trainData.data[i] = rawImg
      else
         trainData.data[i] = image.scale(rawImg, opt.imWidth, opt.imHeight)
      end

      -- load corresponding ground truth
      rawImg = image.load(gtPath[i], 1, 'byte'):squeeze():float()
     -- local mask = rawImg:eq(3):float()
     -- rawImg = rawImg - mask * #classes

      if (opt.labelHeight == rawImg:size(2)) and
         (opt.labelWidth == rawImg:size(3)) then
         trainData.labels[i] = rawImg
      else
         trainData.labels[i] = image.scale(rawImg, opt.labelWidth, opt.labelHeight, 'simple')
      end
      histClasses = histClasses + torch.histc(trainData.labels[i], #classes, 1, #classes)
      xlua.progress(i, #imgPath)
      collectgarbage()
   end

   ----------------------------------------------------------------------
   -- Acquire image and ground truth paths for testing set
   imgPath, gtPath = getPath(testFile)

   testData = {
      data = torch.FloatTensor(#imgPath, opt.channels, opt.imHeight , opt.imWidth),
      labels = torch.FloatTensor(#imgPath, opt.labelHeight , opt.labelWidth),
      preverror = 1e10, -- a really huge number
      size = function() return testData.data:size(1) end
   }

   print "\n==> Loading testing data"
   for i = 1, #imgPath do
      -- load original image
      local rawImg = image.load(imgPath[i])

      if (opt.imHeight == rawImg:size(2)) and
         (opt.imWidth == rawImg:size(3)) then
         testData.data[i] = rawImg
      else
         testData.data[i] = image.scale(rawImg, opt.imWidth, opt.imHeight)
      end

      -- load corresponding ground truth
      rawImg = image.load(gtPath[i], 1, 'byte'):squeeze():float() 
      --local mask = rawImg:eq(3):float()
     --rawImg = rawImg - mask * #classes

      if (opt.labelHeight == rawImg:size(2)) and
         (opt.labelWidth == rawImg:size(3)) then
         testData.labels[i] = rawImg
      else
         testData.labels[i] = image.scale(rawImg, opt.labelWidth, opt.labelHeight, 'simple')
      end


      xlua.progress(i, #imgPath)
      collectgarbage()
   end

   collectgarbage()
--end

----------------------------------------------------------------------


----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i = 1, opt.channels do
   local trainMean = trainData.data[{ {},i }]:mean()
   trainData.data[{{},i}] = trainData.data[{{},i}]-trainMean
   local trainStd = trainData.data[{ {},i }]:std()
   trainData.data[{{},i}] = trainData.data[{{},i}]/trainStd
   
   local trainmean = trainData.data[{ {},i }]:mean()
   local trainstd = trainData.data[{ {},i }]:std()
   
     local testMean = testData.data[{ {},i }]:mean()
     testData.data[{{},i}] = testData.data[{{},i}]-testMean
   local testStd = testData.data[{ {},i }]:std() 
   testData.data[{{},i}] = testData.data[{{},i}]/testStd

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, channel-'.. i ..', mean: ' .. trainmean)
   print('training data, channel-'.. i ..', standard deviation: ' .. trainstd)

   print('test data, channel-'.. i ..', mean: ' .. testMean)
   print('test data, channel-'.. i ..', standard deviation: ' .. testStd)
end

----------------------------------------------------------------------

