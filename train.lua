require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'LSTM'

require 'lstm_init'
require 'rand_data'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Model options
cmd:option('-dropout', 0.6)
cmd:option('-seqLength', 59)
cmd:option('-lstmHidden', 30)

-- Model options
cmd:option('-numClasses', 11)

-- Optimization options
cmd:option('-numEpochs', 200)
cmd:option('-learningRate', 1e-2)
cmd:option('-lrDecayFactor', 0.5)
cmd:option('-lrDecayEvery', 25)

-- Output options
cmd:option('-printEvery', 1) -- Print the loss after every n epochs
cmd:option('-checkpointEvery', 1000) -- Save model, print train acc
cmd:option('-checkpointName', 'checkpoints/checkpoint') -- Save model

-- Backend options
cmd:option('-cuda', 0)

local opt = cmd:parse(arg)

-- Torch cmd parses user input as strings so we need to convert number strings to numbers
for k, v in pairs(opt) do
  if tonumber(v) then
    opt[k] = tonumber(v)
  end
end


-- Set up GPU
opt.dtype = 'torch.DoubleTensor'
if opt.cuda == 1 then
  require 'cunn'
  opt.dtype = 'torch.CudaTensor'
end



-- Initialize model and criterion and data
utils.printTime("Initializing LSTM")
local model = lstm_init(opt):type(opt.dtype) --torch.load('checkpoints/checkpoint_final.t7').model:type(opt.dtype)  

local criterion = nn.CrossEntropyCriterion():type(opt.dtype) --nn.ClassNLLCriterion()
criterion.nll.sizeAverage = false

local data,labels,x_test,y_test=getData()

--[[
  Input:
    - model: an LSTM

  Trains a fresh LSTM from end to end. Also uses the opt parameters declared above.
]]--
function train(model)
  utils.printTime("Starting training for %d epochs" % {opt.numEpochs})

  local trainLossHistory = {}
  local valLossHistory = {}
  local valLossHistoryEpochs = {}

  local config = {learningRate = opt.learningRate}
  local params, gradParams = model:getParameters()
  
  for i = 1, opt.numEpochs do
    collectgarbage()

    local epochLoss = {}
    local videosProcessed = 0

    if i % opt.lrDecayEvery == 0 then
      local oldLearningRate = config.learningRate
      config = {learningRate = oldLearningRate * opt.lrDecayFactor}
    end
  
    if opt.cuda == 1 then
    data = data:cuda()
    labels = labels:cuda()
    end
    
--    local dataSize=data:size()[1]
--    for j=1,dataSize do
    

    local function feval(x)
    collectgarbage()

        if x ~= params then
          params:copy(x)
        end

        gradParams:zero()

        local modelOut = model:forward(data)
        local frameLoss = criterion:forward(modelOut, labels)
        local gradOutputs = criterion:backward(modelOut, labels)
        local gradModel = model:backward(data, gradOutputs)
        
--[[
	    for j=1,2 do
        utils.printTime("%s,%s       %s      %s" %{modelOut[j][1],modelOut[j][2],labels[j],criterion:forward(modelOut, labels)}) --
        end
        
        for j=1060,1062 do
        utils.printTime("%s       %s      %s" %{modelOut[j],labels[j],criterion:forward(modelOut, labels)}) --
        end
]]--

        return frameLoss, gradParams
        end

    --optim.adam: a function that takes a single input (X), the point of a evaluation, and returns f(X) and df/dX
    local _, loss = optim.adam(feval, params, config)
    
    table.insert(epochLoss, loss[1])
            


    local epochLoss = torch.mean(torch.Tensor(epochLoss))
    table.insert(trainLossHistory, epochLoss)

    -- Print the epoch loss
    if (opt.printEvery > 0 and i % opt.printEvery == 0) then
      utils.printTime("Epoch %d training loss: %f" % {i, epochLoss})
    end
    
    local minLoss=10

    -- Save a checkpoint of the model, its opt parameters, the training loss history, and the validation loss history
    if (opt.checkpointEvery > 0 and i % opt.checkpointEvery == 0) or i == opt.numEpochs or epochLoss<minLoss then
      local valLoss = test(model, 'val', 'loss')
      utils.printTime("Epoch %d validation loss: %f" % {i, valLoss})
      table.insert(valLossHistory, valLoss)
      table.insert(valLossHistoryEpochs, i)

      local checkpoint = {
        opt = opt,
        trainLossHistory = trainLossHistory,
        valLossHistory = valLossHistory
      }

      local filename
      if i == opt.numEpochs or epochLoss<minLoss then
        filename = '%s_%s.t7' % {opt.checkpointName, 'final'}
      else
        filename = '%s_%d.t7' % {opt.checkpointName, i}
      end

      -- Make sure the output directory exists before we try to write it
      paths.mkdir(paths.dirname(filename))

      -- Cast model to float so it can be used on CPU
      model:float()
      checkpoint.model = model
      torch.save(filename, checkpoint)

      -- Cast model back so that it can continue to be used
      model:type(opt.dtype)
      params, gradParams = model:getParameters()
      utils.printTime("Saved checkpoint model and opt at %s" % filename)
      collectgarbage()
    end
    
    if epochLoss<minLoss then
    break
    end
    
    end 
  utils.printTime("Finished training")
  return model  
  end

  




--[[
  Inputs:
    - model: an LSTM
    - split: 'train', 'val', or 'test'
    - task: 'detection', or 'loss'

    Same code used in testing
]]--
function test(model, split, task)
  assert(task == 'detection' or task == 'loss')
  collectgarbage()
  utils.printTime("Starting %s testing on the %s split" % {task, split})

  local evalData = {}
  evalData.loss = 0 -- sum of losses
  evalData.numBatches = 0 -- total number of frames


--    if opt.cuda == 1 then
--      data = data:cuda()
--      labels = labels:cuda()
--    end


      local numData = data:size()[1]
      local scores = model:forward(data)
      evalData.loss = evalData.loss + criterion:forward(scores, labels)
      evalData.numBatches = evalData.numBatches + 1


  if task == 'recognition' or task == 'detection' then
    evalData.predictedLabels = torch.Tensor(evalData.predictedLabels)
    evalData.trueLabels = torch.Tensor(evalData.trueLabels)
    return torch.sum(torch.eq(evalData.predictedLabels, evalData.trueLabels)) / evalData.predictedLabels:size()[1]
  else
    return evalData.loss
  end
end

train(model)
