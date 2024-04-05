import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
# from compGCN import CompGraphConv
import torch.nn as nn
import torch
import clip
import collections
from transformers import BertModel, BertTokenizer

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


''' 
Load state_dict in pre_model to model
Solve the problem that model and pre_model have some different keys
'''
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
class CLIP_SI(nn.Module):
    def __init__(self): #, **kwargs
        super(Pair_CLIP_SI, self).__init__()

        self.model,_ = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
        self.image_encoder_SI = self.model.encode_image
        self.projector_si = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.ReLU(),
            nn.Linear(768, 512, bias=False),
        )
        torch.nn.init.xavier_normal_(self.projector_si[0].weight.data)
        torch.nn.init.xavier_normal_(self.projector_si[2].weight.data)

        self.image_encoder_SV = self.model.encode_image
        self.projector_sv = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.ReLU(),
            nn.Linear(768, 512, bias=False),
        )
        torch.nn.init.xavier_normal_(self.projector_sv[0].weight.data)
        torch.nn.init.xavier_normal_(self.projector_sv[2].weight.data)

        self.encoder_TEXT = self.model.encode_text
        self.projector_TEXT = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.ReLU(),
            nn.Linear(768, 512, bias=False),
        )
        torch.nn.init.xavier_normal_(self.projector_TEXT[0].weight.data)
        torch.nn.init.xavier_normal_(self.projector_TEXT[2].weight.data)

    def forward(self, si_global, text_global, si_local, stv):
        si_global = self.image_encoder_SI(si_global)
        si_global_features = self.projector_si(si_global)

        si_local = self.image_encoder_SI(si_local)
        si_local_features = self.projector_si(si_local)

        stv = self.image_encoder_SV(stv)
        stv_features = self.projector_sv(stv)

        text_global = self.encoder_TEXT(text_global)
        text_features = self.projector_TEXT(stv)

        # calculate loss for kge-sv-si contrastive
        score = torch.einsum('ai, ci->ac', si_global_features, text_features)
        # [nb, nb]
        score_1 = F.softmax(score, dim=1)
        diag_1 = torch.diag(score_1)
        loss_1 = -torch.log(diag_1 + 1e-10).sum()
        # [nb, nb]
        score_2 = F.softmax(score, dim=0)
        diag_2 = torch.diag(score_2)
        loss_2 = -torch.log(diag_2 + 1e-10).sum()

        score2 = torch.einsum('ai, ci->ac', si_local_features, stv_features)
        # [nb, nb]
        score2_1 = F.softmax(score2, dim=1)
        diag2_1 = torch.diag(score2_1)
        loss2_1 = -torch.log(diag2_1 + 1e-10).sum()
        # [nb, nb]
        score2_2 = F.softmax(score2, dim=0)
        diag2_2 = torch.diag(score2_2)
        loss2_2 = -torch.log(diag2_2 + 1e-10).sum()
        return loss_1 + loss_2 + loss2_1 + loss2_2

    # def get_feature(self, si):
    #     si_features = self.si_encoder(si)
    #     return si_features
    
    # def get_feature(self, kg_idx,att):
    #     si_features = self.bert_model(kg_idx,att)
    #     last_hidden_states = torch.mean(si_features.last_hidden_state, dim=1)
    #     return last_hidden_states


# class Pair_CLIP_SV(nn.Module):
#     def __init__(self, node_emb, rel_emb, layer_size, layer_dropout, **kwargs):
#         super(Pair_CLIP_SV, self).__init__()
#         d = kwargs['d']
#         self.g = kwargs['g']
#         self.layer_size = layer_size
#         self.layer_dropout = layer_dropout
#         self.num_layer = len(layer_size)

#         model_simCLR_resnet18_path = "../data/model_pretrain/checkpoint_100.tar"
#         resnet18_pretrain_sv = torch.load(model_simCLR_resnet18_path)
#         mlp_pretrain_sv = torch.load(model_simCLR_resnet18_path)
#         self.sv_encoder = get_resnet(name='resnet18', pretrained=False)
#         self.sv_encoder.fc = Identity()
#         self.sv_encoder = load_pretrain(self.sv_encoder, resnet18_pretrain_sv)
#         self.projector_sv = nn.Sequential(
#             nn.Linear(512, 512, bias=False),
#             nn.ReLU(),
#             nn.Linear(512, 64, bias=False),
#         )
#         self.projector_sv[0].weight.data = mlp_pretrain_sv['projector.0.weight']
#         self.projector_sv[2].weight.data = mlp_pretrain_sv['projector.2.weight']

#         # CompGCN layers
#         self.layers = nn.ModuleList()
#         self.layers.append(CompGraphConv(64, self.layer_size[0]))
#         for i in range(self.num_layer - 1):
#             self.layers.append(CompGraphConv(self.layer_size[i], self.layer_size[i + 1]))

#         # Initial relation embeddings
#         self.rel_embds = nn.Embedding.from_pretrained(rel_emb, freeze=True)
#         # Node embeddings
#         self.n_embds = nn.Embedding.from_pretrained(node_emb, freeze=True)
#         # Dropout after compGCN layers
#         self.dropouts = nn.ModuleList()
#         for i in range(self.num_layer):
#             self.dropouts.append(nn.Dropout(self.layer_dropout[i]))
#         # CompGCN +mlp_projector
#         self.projector_ent = nn.Sequential(
#             nn.Linear(64, 64, bias=False),
#             nn.ReLU(),
#             nn.Linear(64, 64, bias=False),
#         )

#         torch.nn.init.xavier_normal_(self.projector_ent[0].weight.data)
#         torch.nn.init.xavier_normal_(self.projector_ent[2].weight.data)

#     def forward(self, sv, kg_idx):
#         sv = sv.reshape(len(kg_idx) * 10, 3, 224, 224)
#         sv = self.sv_encoder(sv)
#         sv_features = sv.reshape(len(kg_idx), 10, 512)

#         sv_features = torch.mean(sv_features, dim=1)

#         sv_features = self.projector_sv(sv_features)

#         n_feats = self.n_embds.weight
#         r_feats = self.rel_embds.weight
#         for layer, dropout in zip(self.layers, self.dropouts):
#             n_feats, r_feats = layer(self.g, n_feats, r_feats)
#             n_feats = dropout(n_feats)
#         kge_features = n_feats[kg_idx, :]
#         kge_features = self.projector_ent(kge_features)

#         score = torch.einsum('ai, bi->ab', kge_features, sv_features)
#         # [nb, nb]
#         score_1 = F.softmax(score, dim=1)
#         diag_1 = torch.diag(score_1)
#         loss_1 = -torch.log(diag_1 + 1e-10).sum()
#         # [nb, nb]
#         score_2 = F.softmax(score, dim=0)
#         diag_2 = torch.diag(score_2)
#         loss_2 = -torch.log(diag_2 + 1e-10).sum()
#         return loss_1 + loss_2

#     def get_feature(self, sv):
#         batch_size = sv.shape[0]
#         sv = sv.reshape(batch_size * 10, 3, 224, 224)
#         sv = self.sv_encoder(sv)
#         sv_features = sv.reshape(batch_size, 10, 512)
#         sv_features = torch.mean(sv_features, dim=1)

#         return sv_features

class Pair_CLIP_SI(nn.Module):
    def __init__(self): #, **kwargs
        super(Pair_CLIP_SI, self).__init__()

        self.model,_ = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
        self.image_encoder_SI = self.model.encode_image
        self.projector_si = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 64, bias=False),
        )
        torch.nn.init.xavier_normal_(self.projector_si[0].weight.data)
        torch.nn.init.xavier_normal_(self.projector_si[2].weight.data)

        self.image_encoder_SV = self.model.encode_image
        self.projector_sv = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 64, bias=False),
        )
        torch.nn.init.xavier_normal_(self.projector_sv[0].weight.data)
        torch.nn.init.xavier_normal_(self.projector_sv[2].weight.data)

        # 加载预训练的BERT模型和分词器
        self.model_name = 'bert-base-uncased' #'uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12' #
        # self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.encoder_TEXT = BertModel.from_pretrained(self.model_name)
        # 去掉最后的分类层
        # self.bert_model.config.num_labels = None
        # self.bert_model.classifier = None
        self.projector_TEXT = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.ReLU(),
            nn.Linear(768, 64, bias=False),
        )
        torch.nn.init.xavier_normal_(self.projector_TEXT[0].weight.data)
        torch.nn.init.xavier_normal_(self.projector_TEXT[2].weight.data)

    def forward(self, si_global, si_local, stv, text_global,attention_mask):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        si_global = self.image_encoder_SI(si_global).float()
        si_global_features = self.projector_si(si_global).float()

        si_local = self.image_encoder_SI(si_local).float()
        si_local_features = self.projector_si(si_local).float()

        stv = self.image_encoder_SV(stv).float()
        stv_features = self.projector_sv(stv).float()


        bert = self.encoder_TEXT(text_global,attention_mask)
        # 获取最后一个隐藏层的输出
        # last_hidden_states = torch.mean(bert.last_hidden_state, dim=1)  #之前的都是
        last_hidden_states = (bert.last_hidden_state[:, 0])
        # last_hidden_states = bert.last_hidden_state[:, 0, :]
        # print("last_hidden_states .size():", last_hidden_states.size())
        text_features = self.projector_TEXT(last_hidden_states).float()

        # calculate loss for kge-sv-si contrastive
        score = torch.einsum('ai, ci->ac', si_global_features, text_features)
        # [nb, nb]
        score_1 = F.softmax(score, dim=1)
        diag_1 = torch.diag(score_1)
        loss_1 = -torch.log(diag_1 + 1e-10).sum()
        # [nb, nb]
        score_2 = F.softmax(score, dim=0)
        diag_2 = torch.diag(score_2)
        loss_2 = -torch.log(diag_2 + 1e-10).sum()

        score2 = torch.einsum('ai, ci->ac', si_local_features, stv_features)
        # [nb, nb]
        score2_1 = F.softmax(score2, dim=1)
        diag2_1 = torch.diag(score2_1)
        loss2_1 = -torch.log(diag2_1 + 1e-10).sum()
        # [nb, nb]
        score2_2 = F.softmax(score2, dim=0)
        diag2_2 = torch.diag(score2_2)
        loss2_2 = -torch.log(diag2_2 + 1e-10).sum()
        return loss_1, loss_2, loss2_1, loss2_2
