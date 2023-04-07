from lib import*

from torch.utils.data import Dataset

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

#in this situation it work with char so if want to work with word just modify tokennize process and input sentence to dataloader
def tokenize(sentence, language_to_index,max_length,start_token=True, end_token=True):
    sentence_word_indicies = [language_to_index[token] for token in list(sentence)]
    if start_token:
        sentence_word_indicies.insert(0, language_to_index[START_TOKEN])
    if end_token:
        sentence_word_indicies.append(language_to_index[END_TOKEN])
    for _ in range(len(sentence_word_indicies), max_length):
        sentence_word_indicies.append(language_to_index[PADDING_TOKEN])
    return torch.tensor(sentence_word_indicies)


#sentences must be processed before put into Dataset
class Dataset(Dataset):
    def __init__(self, ori_sentences, trans_sentences,ori_to_index,trans_to_index,max_length):
        self.ori_sentences = ori_sentences
        self.trans_sentences = trans_sentences
        self.ori_to_index  = ori_to_index
        self.trans_sentences = trans_to_index
        self.max_length        = max_length
        self.tokenize          = tokenize #function to add signs

    def __len__(self):
        return len(self.ori_sentences)

    def __getitem__(self, idx):
        # Before return we will process to add padding and start end sign or we can process it before and after batch
        ori_sentence    = self.tokenize(self.ori_sentences[idx],self.ori_to_index,self.max_length,False,False)
        decoder_input   = self.tokenize(self.trans_sentences[idx],self.trans_sentences,self.max_length,False,True)
        trans_sentence    = self.tokenize(self.trans_sentences[idx],self.trans_sentences,self.max_length,True,False)
        return ori_sentence,decoder_input,trans_sentence

def my_collate_fn(batch):
    targets = []
    signs  = []
    inputs = []

    for sample in batch:
        inputs.append(sample[0])
        signs.append(sample[1])
        targets.append(sample[2]) 
    
    inputs=torch.stack(inputs,0)
    signs=torch.stack(inputs,0)
    targets=torch.stack(targets,0)
    #(batch,max_length)
    return inputs,signs,targets

'''

for example:

train_dataset = Dataset(ori_sentence,trans_sentence,ori_to_index,trans_to_index,max_length)

train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=my_collate_fn)


'''