from lib import *

class PositionalEncoding(nn.Module):
    def __init__(self,d_embedding,max_length_sentence):
        super(PositionalEncoding,self).__init__()
        self.max_length_sequence    =max_length_sentence
        self.d_model                =d_embedding
        
    def forward(self):
        even_i_model    =torch.arange(0,self.d_model,2)
        denominator     =torch.pow(10000,even_i_model/self.d_model)
        position        =torch.arange(self.max_length_sequence).reshape(-1,1)
        PE_even         =torch.sin(position/denominator)
        PE_odd          =torch.cos(position/denominator)
        PE_total        =torch.stack([PE_even,PE_odd],dim=2).reshape(self.max_length_sequence,-1)
        
        return PE_total