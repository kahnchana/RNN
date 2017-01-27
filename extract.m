cd 'E:\Machine Learning\DataSets\58' ; %change path for different file sets
x=[]; 
y=[];
a=dir('*.mat');
class={'biking','diving','golf','juggle','jumping','riding','shooting','spiking','swing','tennis','walk'};

for i=1:numel(a)    %extract file names and import the .mat files
    %x contains the values of each 1x1000 vector
    %y contains the label names of each vector
    temp=load(a(i).name);
    temp=struct2cell(temp);
    temp=cell2mat(temp);
    x=cat(1,x,temp); 
    
    temp=a(i).name;
    tempInd=strfind(temp,'_');
    temp1=temp(tempInd(1)+1:tempInd(2)-1);
    [q,temp1]=max(strcmp(temp1,class));
    clear q;
    if temp1~=11 %11th class has two words
        temp2=temp(tempInd(2)+1:tempInd(2)+2);
        temp3=temp(tempInd(3)+1:tempInd(3)+2);
        temp4=temp(tempInd(4)+1:tempInd(4)+2);
    else
        temp2=temp(tempInd(3)+1:tempInd(3)+2);
        temp3=temp(tempInd(4)+1:tempInd(4)+2);
        temp4=temp(tempInd(5)+1:tempInd(5)+2);
    end
    y=[y ;temp1 str2double(temp2) str2double(temp3) str2double(temp4)];
end

%create data of size (N x 59 x 1000) and labels (N x 1)
data=zeros(2000,59,1000);
labels=zeros(2000,1);
vectorNum=1;

data(1,1,:)=x(1,:);
labels(1)= y(1,1);

for i=2:15565
    if y(i,3)==y(i-1,3)
        data(vectorNum,y(i,4),:)=x(i,:);
    else
        vectorNum=vectorNum+1;
        data(vectorNum,y(i,4),:)=x(i,:);
        labels(vectorNum)=y(i,1);
    end
end

%remove additional parts of data and labels matrices
temp=find(labels==0);
data=data(1:temp-1,:,:);
labels=labels(1:temp-1);

save('data.mat','data'); %size (1532 x 59 x 1000) 
save('labels.mat','labels'); % (1532 x 1)

cd 'E:\Machine Learning\DataSets' ;

