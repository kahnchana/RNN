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
cmd:option('-checkpoint', 'checkpoints/checkpoint_final.t7')
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
local criterion = nn.CrossEntropyCriterion():type(opt.dtype)
criterion.nll.sizeAverage = false

--[[
	Inputs:
		- model: an LSTM
		- split: 'train', 'val', or 'test'
		- task: 'detection', or 'loss'

]]--

function test(model, split, task)
	assert(task == 'detection' or task == 'loss')
  collectgarbage()
  utils.printTime("Starting %s testing on the %s split" % {task, split})

  local evalData = {}
  if task == 'detection' then
  	evalData.predictedLabels = {} -- predicted video or frame labels
  	evalData.trueLabels = {} -- true video or frame labels
  end

  local data,labels,x,y=getData()
    
    if opt.cuda == 1 then
      data = data:cuda()
      labels = labels:cuda()
    end


	  if task == 'detection' then
	    numData = labels:size()[1]
	    local scores = model:forward(data):exp()
	    scores:resize(numData,11)

	    for i = 1, numData do
	      local _, predictedLabel = torch.max(scores[i],1)
	      predictedLabel=predictedLabel[1]
	      utils.printTime("%d       %d" %{predictedLabel,labels[i]})
	      table.insert(evalData.predictedLabels, predictedLabel)
	      table.insert(evalData.trueLabels, labels[i])
	    end
	  end


  if task == 'detection' then
  	  evalData.predictedLabels = torch.Tensor(evalData.predictedLabels)
	  evalData.trueLabels = torch.Tensor(evalData.trueLabels)
	  return torch.sum(torch.eq(evalData.predictedLabels,evalData.trueLabels))/numData*100
  end

end

local testDetectionAcc = test(model, 'test', 'detection')
utils.printTime("Action detection accuracy on the test set: %f" % testDetectionAcc)

