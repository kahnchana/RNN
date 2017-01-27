cd 'E:\Machine Learning\DataSets\hollywood\prob 59' ; %change path for different file sets
x=[]; 
y=[];
a=dir('*.mat');
class={'AnswerPhone','DriveCar','Eat','FightPerson','GetOutCar','HandShake','HugPerson','Kiss','Run','SitDown','SitUp','StandUp'};


for i=1:numel(a)    %extract file names and import the .mat files
    %x contains the values of each 1x1000 vector
    %y contains the label names of each vector
    temp=load(a(i).name);
    temp=struct2cell(temp);
    temp=cell2mat(temp);
    x=cat(1,x,temp); 
    
    temp=a(i).name;
    tempInd0=strfind(temp,'0');
    tempInd_=strfind(temp,'_');
    temp1=temp(1:tempInd0(1)-1);
    [~,temp1]=max(strcmp(temp1,class));

    temp2=temp(tempInd0(1):tempInd_(1)-1);
    temp3=temp(tempInd_(1)+1:tempInd_(1)+2);

    y=[y ;temp1 str2double(temp2) str2double(temp3)];
end

%create data of size (N x 10 x 1000) and labels (N x 1)
data=zeros(1000,10,1000);
labels=zeros(1000,1);
vectorNum=1;

data(1,1,:)=x(1,:);
labels(1)= y(1,1);

for i=2:5237
    if y(i,2)==y(i-1,2)
        data(vectorNum,y(i,3),:)=x(i,:);
    else
        vectorNum=vectorNum+1;
        data(vectorNum,y(i,3),:)=x(i,:);
        labels(vectorNum)=y(i,1);
    end
end

%remove additional parts of data and labels matrices
temp=find(labels==0);
data=data(1:temp-1,:,:);
labels=labels(1:temp-1);

save('data.mat','data'); %size (1532 x 10 x 1000) 
save('labels.mat','labels'); % (1532 x 1)

cd 'E:\Machine Learning\DataSets' ;

