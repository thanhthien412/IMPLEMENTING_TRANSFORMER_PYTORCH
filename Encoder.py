from lib import*

from MultipleHeaderAttention import MultiHeadAttention
from Layernorm import Layernormalization
from FeedForward import PositionwiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self,d_input,num_head,ffn,pro_dropout):
        super().__init__()
        self.dim            =d_input
        self.MultiheadAtention=MultiHeadAttention(d_input,num_head)
        self.layernorm1      =Layernormalization(d_input)
        self.layernorm2      =Layernormalization(d_input)
        self.ffn             =PositionwiseFeedForward(d_input,ffn,pro_dropout)
        self.drop1           =nn.Dropout(pro_dropout)
        self.drop2           =nn.Dropout(pro_dropout)
        
    def forward(self,x):
        attention_matrix    =self.MultiheadAtention(x)
        attention_matrix    =self.drop1(attention_matrix)
        attention_matrix   +=x
        #attention_matrix    =nn.LayerNorm(self.dim)(attention_matrix)
        attention_matrix    =self.layernorm1(attention_matrix)
        residual            =attention_matrix
       
        attention_matrix    = self.ffn(attention_matrix)
        
        attention_matrix   +=residual
        #attention_matrix    =nn.LayerNorm(self.dim)(attention_matrix)
        attention_matrix    =self.layernorm2(attention_matrix)
        
        return attention_matrix
    


class Encoder(nn.Module):
    def __init__(self,d_input,num_head,stack_block,ffn=2048,dropout_pro=0.1):
        super().__init__()
        self.model=nn.Sequential(*[EncoderBlock(d_input,num_head,ffn,dropout_pro) for _ in range(stack_block)])
        
    def forward(self,x):
        x=self.model(x)
        
        return x
        

# d_model=512
# num=8
# model=Encoder(d_model,num,3)
# #model=Layernormalization(512)
# x=torch.rand((30,5,d_model))
# summary(model,(200,512),1)
# #out=model(x)