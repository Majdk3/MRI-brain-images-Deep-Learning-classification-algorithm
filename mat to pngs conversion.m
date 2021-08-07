import mlreportgen.dom.*
for i=1:3000
    
    i=mat2str(i);   
    
    i=strcat(i,'.mat');
    x='C:\Users\ROG\Desktop\brainTumorRetrieval-master\imageData\';
    
    
    z=load(strcat(x,i));
    
    z2=mat2str(z.cjdata.label);
    if z2=="1"
    y='C:\Users\ROG\Desktop\brainTumorRetrieval-master\imageData\1\';
    e=strcat(i,'label=',z2,'.png');
    z1=strcat(y,e);

    imwrite(mat2gray(z.cjdata.image),z1);
    elseif z2=="2"
    y='C:\Users\ROG\Desktop\brainTumorRetrieval-master\imageData\2\';
    e=strcat(i,'label=',z2,'.png');
    z1=strcat(y,e);

    imwrite(mat2gray(z.cjdata.image),z1);
    elseif z2=="3"
     y='C:\Users\ROG\Desktop\brainTumorRetrieval-master\imageData\3\';
    e=strcat(i,'label=',z2,'.png');
    z1=strcat(y,e);
    imwrite(mat2gray(z.cjdata.image),z1);
    end
    
end;

