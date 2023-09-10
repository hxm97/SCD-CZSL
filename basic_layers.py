import torch.nn as nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(
            self,
            inp_dim,          
            latent_dim,         
            out_dim,            
            num_layers=2,      
            bias=True,         
            batchnorm=True,    
            layernorm=False,  
            dropout=0,          
            end_relu=False,  
            drop_input=0,    
            drop_output=0, 
            final_linear_bias=True
        ):
        super(MLP, self).__init__()
        mod = []

        if drop_input > 0:
            mod.append(nn.Dropout(drop_input))

        mod.append(nn.Linear(inp_dim, latent_dim, bias=bias))
        if batchnorm:
            mod.append(nn.BatchNorm1d(latent_dim))
        if layernorm:
            mod.append(nn.LayerNorm(latent_dim))
        mod.append(nn.ReLU(True))

        for L in range(num_layers-2):
            mod.append(nn.Linear(latent_dim, latent_dim, bias=bias))
            if batchnorm:
                mod.append(nn.BatchNorm1d(latent_dim))
            if layernorm:
                mod.append(nn.LayerNorm(latent_dim))
            mod.append(nn.ReLU(True))
        
        if dropout > 0:
            mod.append(nn.Dropout(dropout))

        mod.append(nn.Linear(latent_dim, out_dim, bias=final_linear_bias))

        if end_relu:
            mod.append(nn.ReLU(True))

        if drop_output > 0:
            mod.append(nn.Dropout(drop_output))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output


class DyCls(nn.Module):
    def __init__(self, input_dim, out_dim, device, bias=True, tmp=1):
        super(DyCls, self).__init__()
        self.cls = nn.Linear(input_dim, out_dim, bias = bias)
        self.att_dims = input_dim
        self.tmp = tmp
        self.device = device
    
    def forward(self, x, mod):
        cls_para = self.cls.weight      
        batch_size = x.shape[0]         
        cls_num = cls_para.shape[0]         
        att = F.softmax(mod/self.tmp, dim=1).reshape(-1,1,self.att_dims)
        att = att+torch.ones_like(att)
        att = att.repeat(1,cls_num,1) 
        
        cls_para = cls_para.reshape(1,-1,self.att_dims)
        cls_para = cls_para.repeat(batch_size,1,1)     
        cls_para = att*cls_para  

        result = torch.zeros(batch_size, cls_num).to(self.device)
        for i in range(batch_size):
            result[i] = torch.matmul(cls_para[i],x[i])
        
        return result