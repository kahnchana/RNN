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
cmd:option('-dropout', 0.5)
cmd:option('-seqLength', 8)
cmd:option('-lstmHidden', 256)

-- Model options
cmd:option('-numClasses', 1)

-- Optimization options
cmd:option('-numEpochs', 50)
cmd:option('-learningRate', 1e-6)
cmd:option('-lrDecayFactor', 0.5)
cmd:option('-lrDecayEvery', 5)

-- Output options
cmd:option('-printEvery', 1) -- Print the loss after every n epochs
cmd:option('-checkpointEvery', 100) -- Save model, print train acc
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
local model = lstm_init(opt):type(opt.dtype)
local criterion = nn.ClassNLLCriterion():type(opt.dtype)
local data,labels=genData()

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

    -- Save a checkpoint of the model, its opt parameters, the training loss history, and the validation loss history
    if (opt.checkpointEvery > 0 and i % opt.checkpointEvery == 0) or i == opt.numEpochs then
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
      if i == opt.numEpochs then
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

      -- Cast model back so that it can continue to beu sed
      model:type(opt.dtype)
      params, gradParams = model:getParameters()
      utils.printTime("Saved checkpoint model and opt at %s" % filename)
      collectgarbage()
    end
    end
  utils.printTime("Finished training")  
  end

  




--[[
  Inputs:
    - model: an LRCN
    - split: 'train', 'val', or 'test'
    - task: 'recognition', 'detection', or 'loss'

  Performs either action recognition accuracy, action detection accuracy, or 
  loss for a split based on what task the user inputs.

  Action recognition is done by calculating the scores for each frame. The 
  score for a video is the max of the average of its sequence of frames.

  Action detection is done by calculating the scores for each frame and then 
  getting the max score for each frame.
]]--
function test(model, split, task)
  assert(task == 'recognition' or task == 'detection' or task == 'loss')
  collectgarbage()
  utils.printTime("Starting %s testing on the %s split" % {task, split})

  local evalData = {}
  if task == 'recognition' or task == 'detection' then
    evalData.predictedLabels = {} -- predicted video or frame labels
    evalData.trueLabels = {} -- true video or frame labels
  else
    evalData.loss = 0 -- sum of losses
    evalData.numBatches = 0 -- total number of frames
  end


  while batch ~= nil do
    if opt.cuda == 1 then
      batch.data = batch.data:cuda()
      batch.labels = batch.labels:cuda()
    end

    if task == 'recognition' then
      local numData = batch:size() / checkpoint.opt.seqLength
      local scores = model:forward(batch.data)

      for i = 1, numData do
        local startIndex = (i - 1) * checkpoint.opt.seqLength + 1
        local endIndex = i * checkpoint.opt.seqLength
        local videoFrameScores = scores[{ {startIndex, endIndex}, {} }]
        local videoScore = torch.sum(videoFrameScores, 1) / checkpoint.opt.seqLength
        local maxScore, predictedLabel = torch.max(videoScore[1], 1)
        table.insert(evalData.predictedLabels, predictedLabel[1])
        table.insert(evalData.trueLabels, batch.labels[i])
      end
    elseif task == 'detection' then
      local numData = batch:size()
      local scores = model:forward(batch.data)

      for i = 1, numData do
        local videoFrameScores = scores[i]
        local _, predictedLabel = torch.max(videoFrameScores, 1)
        table.insert(evalData.predictedLabels, predictedLabel[1])
        table.insert(evalData.trueLabels, batch.labels[i])
      end
    else
      local numData = batch:size()
      local scores = model:forward(batch.data)

      evalData.loss = evalData.loss + criterion:forward(scores, batch.labels)
      evalData.numBatches = evalData.numBatches + 1
    end

    batch = nil
  end

  if task == 'recognition' or task == 'detection' then
    evalData.predictedLabels = torch.Tensor(evalData.predictedLabels)
    evalData.trueLabels = torch.Tensor(evalData.trueLabels)
    return torch.sum(torch.eq(evalData.predictedLabels, evalData.trueLabels)) / evalData.predictedLabels:size()[1]
  else
    return evalData.loss / evalData.numBatches
  end
end

train(model)
