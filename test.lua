require 'torch'
require 'nn'
require 'image'

require 'LSTM'

require 'lstm_init'
require 'rand_data'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Options
cmd:option('-checkpoint', 'checkpoint_final.t7')
cmd:option('-split', 'test')

local opt = cmd:parse(arg)

assert(opt.checkpoint ~= '', "Need a trained network file to load.")

-- Set up GPU
opt.dtype = 'torch.DoubleTensor'
if opt.cuda == 1 then
	require 'cunn'
  opt.dtype = 'torch.CudaTensor'
end

-- Initialize model and criterion
utils.printTime("Initializing model")
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:type(opt.dtype)
local criterion = nn.ClassNLLCriterion():type(opt.dtype)

--[[
	Inputs:
		- model: an LSTM
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
  if task == 'detection' then
  	evalData.predictedLabels = {} -- predicted video or frame labels
  	evalData.trueLabels = {} -- true video or frame labels
  end

  local x,y,data,labels=getData('x_train_1.mat','y_train_1.mat','x_test_1.mat','y_test_1.mat')

    if opt.cuda == 1 then
      data = data:cuda()
      labels = labels:cuda()
    end


	  if task == 'detection' then
	    local numData = 459
	    local scores = model:forward(data)
	    scores:resize(459) 

	    for i = 1, numData do
	      local predictedLabel = scores[i]
	      --local _, predictedLabel = torch.max(videoFrameScores, 1)
	      table.insert(evalData.predictedLabels, predictedLabel)
	      table.insert(evalData.trueLabels, labels[i])
	    end
	  end


  if task == 'recognition' or task == 'detection' then
  	  evalData.predictedLabels = torch.round(torch.Tensor(evalData.predictedLabels))
	  evalData.trueLabels = torch.Tensor(evalData.trueLabels)
	  return torch.sum(torch.eq(evalData.predictedLabels,evalData.trueLabels))/459 --evalData.predictedLabels:size()[1]

  end
end

local testDetectionAcc = test(model, 'test', 'detection')
utils.printTime("Action detection accuracy on the test set: %f" % testDetectionAcc)
--local testRecognitionAcc = test(model, 'test', 'recognition')
--utils.printTime("Action recognition accuracy on the test set: %f" % testRecognitionAcc)
