from lib import*

from Transformer import Transformer
from DataLoader import Dataset,my_collate_fn


from GPUtil import showUtilization as gpu_usage
from numba import cuda
import torch.optim as optim
torch.manual_seed(1234)
import time 
import argparse

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



ap = argparse.ArgumentParser()
ap.add_argument('-d_ori','--d_ori',type=int,required=True)
ap.add_argument('-d_trans','--d_trans',type=int,required=True)
ap.add_argument('-d_model','--d_model',type=int, default=512)
ap.add_argument('-max_l','--max_length',type=int,default=20)
ap.add_argument('-head','--num_head',type=int,default=8)
ap.add_argument('-en_block','--encoder_block',type=int,default=3)
ap.add_argument('-de_block','--decoder_block',type=int,default=3)
ap.add_argument('-hidden','--hidden_layer',type=int,default=2048)
ap.add_argument('-drop','--dropout_prob',type=float,default=0.1)
ap.add_argument('-epoch','--epoch',type=float,default=100)
args = vars(ap.parse_args())

if(args['dropout_prob']<0 or args['dropout_prob'] >=1):
    raise ValueError('Value of dropout_prob must be in range [0,1)')

if(args['d_model']%args['num_head']!=0):
    raise ValueError('d_model should be diviable by num_head')

free_gpu_cache()
print('device: ', device)


model=Transformer(args['d_ori'],args['d_trans'],args['d_model'],args['max_length'],args['num_head'],args['encoder_block'],args['decoder_block'],args['hidden_layer'],args['dropout_prob'])


optimizer = optim.SGD(model.parameters(),lr=1e-3,momentum=0.9,weight_decay=5e-6)



'''
dataloader_dict: 
for example

train_dataset = Dataset(train_ori_sentence,train_trans_sentence,ori_to_index,trans_to_index,max_length
val_dataset = Dataset(train_ori_sentence,val_trans_sentence,ori_to_index,trans_to_index,max_length)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

'''



def train_model(model,optimizer,num_epochs,dataloader_dict,Validation=False):
        model.to(device)
        state=['train']
        if(Validation):
            state.append('val')
            
        for epoch in range(num_epochs):
            iteration=0
            epoch_train_loss=0.0
            epoch_val_loss =0.0
            t_epoch_start = time.time()
            t_iter_start  = time.time()
            print('---'*20)
            print('Epoch{}/{}'.format(epoch+1,num_epochs))
            print('---'*20)

            for phase in state:
                if(phase=='train'):
                    model.train()
                    print("TRAINING")
                else:
                    model.eval()
                    print("---"*10)
                    print("VALIDATION")
                
                for inputs,signs,targets in dataloader_dict[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    signs   = signs.to(device)
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase=='train'):
                        outputs =model(inputs,signs)
                        # output shape (batch,max_length,d_dict_translate)
                        # target shape (batch,max_length)
                        # signs shape (batch,max_length) 
                        # but target have only start sign otherwise signs have end sign
                        # compute loss
                        loss = F.cross_entropy(outputs.permute(0,2,1),targets) # adding ignore_index depend on the index of padding_sign in trans_dict
                        
                        if phase == 'train':
                            loss.backward()
                            nn.utils.clip_grad_value_(model.parameters(),clip_value=1.0)
                            optimizer.step()
                            
                            if(iteration%10):
                                t_iter_end=time.time()
                                duration = t_iter_end - t_iter_start
                                print("Iteration {}/{} || Loss: {:.4f} || 10iter: {:.4f} sec".format(iteration,len(dataloader_dict[phase]),loss.item(),duration))
                                t_iter_start = time.time()
                            
                            epoch_train_loss += loss.item()
                            iteration +=1
                        
                        else:
                            epoch_val_loss  += loss.item()
        
        t_epoch_end=time.time()
        print('---'*20)
        print('Epoch {} || epoch_train_loss: {:.4f} || epoch_val_loss: {:.4f}'.format(epoch+1,epoch_train_loss,epoch_val_loss))
        print('Duration : {:.4f} sec'.format(t_epoch_start-t_epoch_end))
        t_epoch_start=time.time()
        if(epoch+1%10==0):
            torch.save(model.state_dict(),'./weight/epoch_{}.pth'.format(epoch+1))

        
        
'''
Can add list to store lost to plot :)))
'''
            

