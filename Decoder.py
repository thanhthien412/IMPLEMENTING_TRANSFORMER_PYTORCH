from lib import*

from MultipleHeaderAttention import MultiHeadAttention,MultiHeadCrossAttention
from Layernorm import Layernormalization
from FeedForward import PositionwiseFeedForward

class DecoderBlock(nn.Module):
    def __init__(self,d_input,num_head,ffn,drop_prob=0.1):
        super(DecoderBlock,self).__init__()
        self.ffn                    = PositionwiseFeedForward(d_input,ffn,drop_prob)
        self.layernorm1             = Layernormalization(d_input)
        self.layernorm2             = Layernormalization(d_input)
        self.layernorm3             = Layernormalization(d_input)
        self.MaskMultiheadattention = MultiHeadAttention(d_input,num_head)
        self.CrossMultiheadattention= MultiHeadCrossAttention(d_input,num_head)
        
    def forward(self,x,encoder_output,decoder_mask):
        residual1       =x
        attention       =self.MaskMultiheadattention(x,decoder_mask)
        attention       +=residual1
        attention       =self.layernorm1(attention)
        residual2       =attention
        attention       =self.CrossMultiheadattention(encoder_output,attention)
        attention       +=residual2
        attention       =self.layernorm2(attention)
        residual3       =attention
        attention       =self.ffn(attention)
        attention       +=residual3
        attention       =self.layernorm3(attention)
        
        return attention

class SequentialDecoder(nn.Sequential):
    def forward(self,*inputs):
        x,encoder_output,mask=inputs
        
        for module in self._modules.values():
            x=module(x,encoder_output,mask)
        
        return x


class Decoder(nn.Module):
    def __init__(self,d_input,num_head,stack_block,ffn=2048,dropout_pro=0.1):
        super(Decoder,self).__init__()
        self.model=SequentialDecoder(*[DecoderBlock(d_input,num_head,ffn,dropout_pro) for _ in range(stack_block)])
        
    def forward(self,x,encoder_output,mask):
        attention= self.model(x,encoder_output,mask)
        return attention

# d_model = 512
# num_heads = 8
# drop_prob = 0.1
# batch_size = 1
# max_sequence_length = 10
# ffn_hidden = 2048
# num_layers = 5

# x = torch.randn( (batch_size, max_sequence_length, d_model) ) # English sentence positional encoded 
# y = torch.randn( (batch_size, max_sequence_length, d_model) ) # Kannada sentence positional encoded 
# mask = torch.full([max_sequence_length, max_sequence_length] , float('-inf'))
# mask = torch.triu(mask, diagonal=1)
# decoder = Decoder(d_model,num_heads,1,ffn_hidden, drop_prob)
# # summary(decoder,[(10,512),(10,512),(10,10)],1)

# x=decoder(x,y,mask)
# print(x.size())
# print(x)
