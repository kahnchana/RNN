function genData()
    --random set of 20 samples with sample size 1000 and 8 time steps
    local x=torch.rand(20,8,1000)

    --random outputs of 1/2 for the 20 samples
    local y=torch.Tensor(20)
    y:random(1,2)

    return x,y
end


--size of data => train:1072x59x1000 | test:459x59x1000

function getData(x_train,y_train,x_test,y_test)
--input the names of the .mat files to function
    local matio= require 'matio'
    x=matio.load(x_train,'x_train_1')
    y=matio.load(y_train,'y_train_1') 
    y:resize(1072)
    x_t=matio.load(x_test,'x_test_1') 
    x_t:resize(1072,59,1000)   
    y_t=matio.load(y_test,'y_test_1')
    y_t:resize(459)
        
    return x,y,x_t,y_t

end
