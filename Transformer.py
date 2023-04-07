from lib import*

from Encoder import Encoder
from Decoder import Decoder
from Positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self,d_dict_ori,d_dict_translate,d_model,max_length,num_head=8,stack_encoder=3,stack_decoder=3,ffn=2048,dropout_prob=0.1):
        super(Transformer,self).__init__()
        self.embedding_input    =nn.Embedding(d_dict_ori,d_model)
        self.embedding_output   =nn.Embedding(d_dict_translate,d_model)
        self.positionalencode   =PositionalEncoding(d_model,max_length)
        self.encoder            =Encoder(d_model,num_head,stack_encoder,ffn,dropout_prob)
        self.decoder            =Decoder(d_model,num_head,stack_decoder,ffn,dropout_prob)
        self.mask               =torch.full([max_length,max_length],float('-inf'))
        self.mask               =torch.triu(self.mask,diagonal=1)
        self.linear             =nn.Linear(d_model,d_dict_translate)
        self.softmax            =nn.Softmax(dim=-1)
        
    def forward(self,inputs,signs):
        position                =self.positionalencode()
        attention_encoder       = self.embedding_input(inputs)
        
        attention_encoder       += position
        
        attention_encoder       = self.encoder(attention_encoder)
        
        embeeding_signs         =self.embedding_output(signs)
        
        attention_decoder       = self.decoder(embeeding_signs,attention_encoder,self.mask)
        
        translation_space       = self.linear(attention_decoder)
        
        probability             = self.softmax(translation_space)
        
        return probability
    

        
        
        
        
        
        
        
        