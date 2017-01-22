function genData()  --random data for testing
    --random set of 20 samles with sample size 1000 and 8 time steps
    local x=torch.rand(20,59,1000)

    --random outputs of 1/2 for the 20 samples
    local y=torch.Tensor(20)
    y:random(1,2)

    return x,y,x,y
end



--size of data => train:1072x16x1000 | test:459x16x1000

function getData()
--input the names of the .mat files to function
    local matio= require 'matio'
    --local data='/media/kanchana/Work/Machine Learning/DataSets/data/64/data.mat'
    --local labels='/media/kanchana/Work/Machine Learning/DataSets/data/64/labels.mat'
    x=matio.load('data/data.mat','data')
    y=matio.load('data/labels.mat','labels')
    x=x:narrow(2,1,59)
    --y=torch.eq(y,11)+1
    y:resize(1532)
    
    local dl = require 'dataload'
    dataloader = dl.TensorLoader(x,y)
    torch.manualSeed(0) 
    local indices = torch.LongTensor():randperm(dataloader:size())
    dataloader.inputs = torchx.recursiveIndex(nil, dataloader.inputs, 1, indices)
    dataloader.targets = torchx.recursiveIndex(nil, dataloader.targets, 1, indices)

    d1,d2=dataloader:split(0.7)
    x=d1.inputs
    y=d1.targets
    x_t=d2.inputs
    y_t=d2.targets
            
    return x,y,x_t,y_t

end
