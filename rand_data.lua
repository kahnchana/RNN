function genData()
    --random set of 20 samples with sample size 1000 and 8 time steps
    x=torch.rand(20,8,1000)

    --random outputs of 1/2 for the 20 samples
    y=torch.Tensor(20)
    y:random(1,2)

    return x,y
end
