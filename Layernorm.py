from lib import*

class  Layernormalization(nn.Module):
    def __init__(self,d_input):
        super(Layernormalization,self).__init__()
        self.eps    =1e-5
        self.gamma  =nn.Parameter(torch.ones(d_input)) 
        self.beta   =nn.Parameter(torch.zeros(d_input))
        
    def forward(self,x):
        means    =x.mean(-1,keepdim=True)
        var     =((x-means)**2).mean(-1,keepdim=True)
        std     = (var+self.eps).sqrt()
        value   =(x-means)/std
        out     =self.gamma*value + self.beta
        return out
