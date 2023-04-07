from lib import *

def scaled_dot_product(q,k,v,mask=None):
    d_k         =q.size()[-1]
    attention   =torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(d_k)

    if mask != None:
        attention+=mask
    
    attention   =torch.softmax(attention,-1)   
    value   =torch.matmul(attention,v)
    
    return value

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_head):
        super(MultiHeadAttention,self).__init__()
        self.head_dim       = d_model//num_head
        self.num_head       = num_head
        self.qkv_layer      = nn.Linear(d_model,d_model*3)
        self.output_layer   = nn.Linear(d_model,d_model)
        
    def forward(self,x,mask=None):
        batch_size,sequence_length,input_dim=x.size() #(batch,length,dim)
        qkv=self.qkv_layer(x) #(batch,length,dim*3)
        qkv=qkv.reshape(batch_size,sequence_length,self.num_head,self.head_dim*3)
        qkv=qkv.permute(0,2,1,3) #(batch,num_head,length,head_dim*3)
        q,k,v=qkv.chunk(3,dim=-1) #seperate to 3 tensor (batch,head,length,head_dim)
        value=scaled_dot_product(q,k,v,mask) # (batch,head,length,head_dim)
        value=value.reshape(batch_size,sequence_length,self.num_head*self.head_dim)
        out= self.output_layer(value)
        return out
        
        
class MultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model) # 1024
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size() 
        kv = self.kv_layer(x) 
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)  
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)  
        kv = kv.permute(0, 2, 1, 3) 
        q = q.permute(0, 2, 1, 3) 
        k, v = kv.chunk(2, dim=-1) 
        values= scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)  
        return out  
        
        
# input_dim=1024
# d_model=512
# num=8
# model=MultiHeadAttention(input_dim,d_model,num)
# x=torch.rand((30,5,input_dim))
# out=model(x)
# print(out.size())
# print(out)
                                      
        