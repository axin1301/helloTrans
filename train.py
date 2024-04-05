from PIL import Image
import torch
import clip
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from model import *
import pandas as pd
from transformers import BertModel, BertTokenizer

BATCH_SIZE = 64
EPOCH = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
print(torch.cuda.is_available())
# model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
model = Pair_CLIP_SI().cuda()
_, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

class image_title_dataset(Dataset):
    def __init__(self, data_csv, si_global_path, si_local_path, stv_path):
        self.si_global_path = si_global_path
        self.si_local_path = si_local_path
        self.stv_path = stv_path
        self.data = pd.read_csv(data_csv)
        self.img_name_list = list(self.data['stv_img_name1'])
        self.text_list = list(self.data['text'])
        self.model_name = 'bert-base-uncased' #'uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12' #/bert-base-uncased
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        # self.title  = clip.tokenize() #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sat_global_path = self.si_global_path + self.img_name_list[idx].split('_')[0] + '_' + self.img_name_list[idx].split('_')[1] + '.png'
        sat_local_path = self.si_local_path + self.img_name_list[idx].split('_')[0] + '_' + self.img_name_list[idx].split('_')[1] \
          + '_' + self.img_name_list[idx].split('_')[2] + '_' + self.img_name_list[idx].split('_')[3] + '_patch.png'
        stv_path = self.stv_path + self.img_name_list[idx].split('_',4)[-1]

        si_global_image = preprocess(Image.open(sat_global_path)) # Image from PIL module
        si_local_image = preprocess(Image.open(sat_local_path)) # Image from PIL module
        stv_image = preprocess(Image.open(stv_path)) # Image from PIL module

        # title = clip.tokenize(self.text_list[idx])
        text = self.text_list[idx]
        # 加载预训练的BERT模型和分词器
        tmp = self.tokenizer.batch_encode_plus([text], add_special_tokens=True, max_length=512,padding='max_length', return_attention_mask=True)
        title = torch.tensor(tmp['input_ids']).squeeze(0)
        attention_mask = torch.tensor(tmp['attention_mask'])
        return si_global_image.to(device),si_local_image.to(device),stv_image.to(device),title.to(device),attention_mask.to(device)

train_file = 'SAT_STV_concat_text_single_sat_global.csv'
# use your own data
dataset = image_title_dataset(train_file,'SAT_STV_file_sep/SAT/','../../SAT_STV_file_all/SAT_STV_file_all/sat_patch/','SAT_STV_file_sep/STV/')
train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE,shuffle=True,drop_last=True) #Define your own dataloader

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

# loss_img = nn.CrossEntropyLoss()
# loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-2,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# add your own code to track the training progress.
for epoch in range(EPOCH):
  for batch in train_dataloader :
      optimizer.zero_grad()

      images_g,images_l,images_stv,texts,att_text = batch
      # print(texts.shape)
      # print(att_text.shape)
      loss_1, loss_2, loss2_1, loss2_2 = model(images_g,images_l,images_stv,texts,att_text)
      total_loss = loss_1+ loss_2+ loss2_1+ loss2_2
      total_loss = total_loss/BATCH_SIZE
    
      # images_g= images_g.to(device)
      # images_l= images_l.to(device)
      # images_stv= images_stv.to(device)
      # texts = texts.to(device)
    
      # # logits_per_image, logits_per_text = model(images, texts)
      # logits_per_image = model.encode_image(images).float()
      # logits_per_text = model.encode_text(texts).float()
      ##################引入 KnowClip

      # ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

      # total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
      total_loss.backward()
      print(loss_1.item(), loss_2.item(), loss2_1.item(), loss2_2.item(),total_loss.item())


torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"model_checkpoint/model_test.pt") #just change to your preferred folder/filename