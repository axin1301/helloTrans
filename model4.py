import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
# from compGCN import CompGraphConv
import torch.nn as nn
# import clip
import collections
from transformers import BertModel, BertTokenizer
import copy
import open_clip
from PIL import Image
# 设置输出选项，取消省略号
torch.set_printoptions(threshold=float('inf'))

''' 
Load state_dict in pre_model to model
Solve the problem that model and pre_model have some different keys
'''
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

class Pair_CLIP_SI(nn.Module):
    def __init__(self): #, **kwargs
        super(Pair_CLIP_SI, self).__init__()
        # self.model,_ = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
        # self.image_encoder_SI = copy.deepcopy(self.model.encode_image)
        model_name = 'ViT-B-32' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
        self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        ckpt = torch.load(f"checkpoints/RemoteCLIP-{model_name}.pt", map_location="cpu")
        # message = self.model.load_state_dict(ckpt)
        self.model = self.model.to("cuda:0")
        self.image_encoder_SI = copy.deepcopy(self.model.encode_image)

        self.projector_si = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 64, bias=False),
        )
        torch.nn.init.xavier_normal_(self.projector_si[0].weight.data)
        torch.nn.init.xavier_normal_(self.projector_si[2].weight.data)


        self.image_encoder_SV = copy.deepcopy(self.model.encode_image)
        self.projector_sv = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 64, bias=False),
        )
        torch.nn.init.xavier_normal_(self.projector_sv[0].weight.data)
        torch.nn.init.xavier_normal_(self.projector_sv[2].weight.data)

        # # 配置微调参数
        # model_name = "gpt2"  # 预训练模型的名称
        # self.encoder_TEXT = GPT2Model.from_pretrained(model_name)
        # self.model_name = "gpt2"  # 预训练模型的名称
        # self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name, pad_token='<|pad|>')#, eos_token='<|endoftext|>'
        # self.encoder_TEXT.resize_token_embeddings(len(self.tokenizer))

        # # # 加载预训练的BERT模型和分词器
        # # self.model_name = 'bert-base-uncased' #'uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12' #
        # # # self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        # # self.encoder_TEXT = BertModel.from_pretrained(self.model_name)
        # # # 去掉最后的分类层
        # # # self.bert_model.config.num_labels = None
        # # # self.bert_model.classifier = None
        # self.projector_TEXT = nn.Sequential(
        #     nn.Linear(768, 768, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(768, 64, bias=False),
        # )
        # torch.nn.init.xavier_normal_(self.projector_TEXT[0].weight.data)
        # torch.nn.init.xavier_normal_(self.projector_TEXT[2].weight.data)

    def forward(self, si_global, si_local, stv, _):#,att
        # # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # si_global = self.image_encoder_SI(si_global).float()
        # si_global_features = self.projector_si(si_global).float()

        si_local = self.image_encoder_SI(si_local).float()
        si_local_features = self.projector_si(si_local).float()

        stv = self.image_encoder_SV(stv).float()
        stv_features = self.projector_sv(stv).float()


        # # 微调模型
        # outputs = self.encoder_TEXT(**text_global) #,att
        # # 获取最终的CLS嵌入
        # last_hidden_state = outputs.last_hidden_state
        # # print(last_hidden_state.shape)
        # text_features = last_hidden_state[:,0,-1,:]
        # # print(text_features.shape)
        # text_features = self.projector_TEXT(text_features).float()

        # print(si_global_features.shape)
        # print(text_features.shape)
# 
        # print(si_global_features)
        # print(text_features)

        # # calculate loss for kge-sv-si contrastive
        # score = torch.einsum('ai, ci->ac', si_global_features, text_features)
        # # [nb, nb]
        # score_1 = F.softmax(score, dim=1)
        # diag_1 = torch.diag(score_1)
        # loss_1 = -torch.log(diag_1 + 1e-10).sum()
        # print(score_1)
        # # [nb, nb]
        # score_2 = F.softmax(score, dim=0)
        # diag_2 = torch.diag(score_2)
        # loss_2 = -torch.log(diag_2 + 1e-10).sum()
        # print(score_2)

        score2 = torch.einsum('ai, ci->ac', si_local_features, stv_features)
        # [nb, nb]
        score2_1 = F.softmax(score2, dim=1)
        diag2_1 = torch.diag(score2_1)
        loss2_1 = -torch.log(diag2_1 + 1e-10).sum()
        # [nb, nb]
        score2_2 = F.softmax(score2, dim=0)
        diag2_2 = torch.diag(score2_2)
        loss2_2 = -torch.log(diag2_2 + 1e-10).sum()
        return  loss2_1, loss2_2 #, loss_1, loss_2