import torch



def get_batch(dataset,batch_size,context_length,device):
    x=torch.zeros((batch_size,context_length),dtype=torch.long)
    y=torch.zeros((batch_size,context_length),dtype=torch.long)
    for i in range(batch_size):
        start_index=torch.randint(0,len(dataset)-context_length,(1,)).item() # 这里使用item获取单值张良的值，实际可以使用random库生成随机数
        x[i]=torch.tensor(dataset[start_index:start_index+context_length],dtype=torch.long)
        y[i]=torch.tensor(dataset[start_index+1:start_index+context_length+1],dtype=torch.long)
    return x.to(device),y.to(device)