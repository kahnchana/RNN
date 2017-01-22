require 'nn'
require 'LSTM'

local utils = require 'util.utils'


function lstm_init(kwargs)

  assert(kwargs ~= nil)
  
  local model = nn.Sequential()
  local dropout = utils.getKwarg(kwargs, 'dropout')
  local numClasses = utils.getKwarg(kwargs, 'numClasses')
  local seqLength = utils.getKwarg(kwargs, 'seqLength')
  local lstmHidden = utils.getKwarg(kwargs, 'lstmHidden')
  local inputDim = 1000 --pre specified input dimension
  local lstm = {}
  
  lstm.numHidden = lstmHidden -- 30 default
  
  -- Reshape for LSTM; N items x T sequence length x H hidden size
  model:add(nn.LSTM(inputDim, lstm.numHidden))
  model:add(nn.Transpose(dim2, dim3))
  model:add(nn.View(-1, seqLength))
  if dropout > 0 then
    model:add(nn.Dropout(dropout))
  end
  model:add(nn.Linear(seqLength,1))
  model:add(nn.View(-1, lstm.numHidden))
  model:add(nn.Linear(lstm.numHidden, numClasses))  
--  model:add(nn.LogSoftMax())


  return model
end
