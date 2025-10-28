import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from modules.transformer import TransformerEncoder
from torchvision import models


class ClassEmbedding(nn.Module):
    def __init__(self, cfg, trainable=True):
        super(ClassEmbedding, self).__init__()
        idx2vocab = utils.load_files(cfg["DATASET"]["IDX2VOCAB"])
        self.n_token = len(idx2vocab)
        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]

        self.emb = nn.Embedding(self.n_token, self.word_emb_size)
        weight_init = utils.load_files(cfg["DATASET"]["GLOVE"]).astype(np.float32)
        weight_mat = torch.from_numpy(weight_init)
        self.emb.load_state_dict({"weight": weight_mat})

        if not trainable:
            self.emb.weight.requires_grad = False

    def forward(self, x):
        emb = self.emb(x)
        return emb


class AnswerSelector(nn.Module):
    def __init__(self, cfg):
        super(AnswerSelector, self).__init__()
        self.av2i = utils.load_files(cfg["DATASET"]["AVOCAB2IDX"])
        self.len_avocab = len(self.av2i)

        self.glove_cands = utils.load_files(cfg["DATASET"]["GLOVE_ANS_CAND"]).astype(
            np.float32
        )
        self.glove_cands = torch.from_numpy(self.glove_cands).cuda()  #501*300

    def forward(self, inputs):
        # print("self.glove_cands on ",self.glove_cands.device)
        similarity = torch.matmul(inputs, self.glove_cands.transpose(0, 1))
        pred = F.log_softmax(similarity, dim=1)
        return pred


class HypergraphTransformer(nn.Module):
    def __init__(self, cfg, args):
        super(HypergraphTransformer, self).__init__()

        self.cfg = cfg
        self.args = args
        self.n_hop = args.n_hop

        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]
        self.max_num_hqnode = cfg["MODEL"]["NUM_MAX_QNODE"]
        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.max_num_hknode = cfg["MODEL"]["NUM_MAX_KNODE_{}H".format(self.n_hop)]
        self.n_out = cfg["MODEL"]["NUM_OUT"]
        self.n_ans = cfg["MODEL"]["NUM_ANS"]
        self.abl_only_ga = args.abl_only_ga
        self.abl_only_sa = args.abl_only_sa


        #visual encodeing
        self.conv_down_outchannel1= cfg["MODEL"]["CONV_DOWN_OUTCHANNEL1"]
        self.conv_down_outchannelh= cfg["MODEL"]["CONV_DOWN_OUTCHANNELh"]
        self.img_dim=cfg["MODEL"]["IMAGE_DIM"]
        self.kernel1_size=cfg["MODEL"]["KERNEL1_SIZE"]
        self.pool1_size=cfg["MODEL"]["POOL1_SIZE"]
        self.feature_map_drop=cfg["MODEL"]["FEATURE_MAP_DROP"]
        self.vq_c_out1=cfg["MODEL"]["VQ_C_OUT1"]
        self.vq_c_outh=cfg["MODEL"]["VQ_C_OUTh"]
        self.alpha=cfg["MODEL"]["ALPHA"]
        

        self.backbone1=nn.Sequential(*list(models.resnet50(pretrained=True).children()))[0:-2]
        self.backbone1_end=nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),nn.Flatten(),nn.Linear(2048,self.img_dim))

        self.conv_down1=nn.Conv2d(2048, self.conv_down_outchannel1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1=nn.BatchNorm2d(self.conv_down_outchannel1)
        self.fc1=nn.Linear(self.vq_c_out1 * self.pool1_size * self.pool1_size,self.img_dim)

        self.conv_downh=nn.Conv2d(2048, self.conv_down_outchannelh, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bnh=nn.BatchNorm2d(self.conv_down_outchannelh)
        self.fch=nn.Linear(self.vq_c_outh * self.pool1_size * self.pool1_size,self.img_dim)

        self.pool1=nn.AdaptiveAvgPool2d(self.pool1_size)
        self.feature_map_dropout=nn.Dropout2d(self.feature_map_drop)

        self.proj_v1=nn.Linear(self.img_dim,2*self.n_hidden)
        self.proj_v2=nn.Linear(self.img_dim,2*self.n_hidden)

        
        self.trans_v_mem = self.get_network(self_type="v_mem", layers=3)

        

        self.i2e = ClassEmbedding(cfg)

        self.q2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hqnode, self.n_hidden
        )

        self.k2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hknode, self.n_hidden
        )

        self.ques_trans = torch.nn.Linear(
            self.word_emb_size, self.n_hidden
        )

        if self.abl_only_sa != True:
            self.trans_k_with_q = self.get_network(self_type="kq")
            self.trans_q_with_k = self.get_network(self_type="qk")
            

        if self.abl_only_ga != True:
            self.trans_k_mem = self.get_network(self_type="k_mem", layers=3)
            self.trans_q_mem = self.get_network(self_type="q_mem", layers=3)


        self.dropout = nn.Dropout(p=self.cfg["MODEL"]["INP_DROPOUT"])
        self.v_dropout = nn.Dropout(p=self.cfg["MODEL"]["V_DROPOUT"])
        self.out_dropout = 0.0

        if self.args.abl_ans_fc != True:
            self.proj1 = nn.Linear(4 * self.n_hidden, 2*self.n_hidden) 
            self.proj2 = nn.Linear(2*self.n_hidden, self.n_out)
            self.ans_selector = AnswerSelector(cfg)
        else:
            self.proj1 = nn.Linear(4 * self.n_hidden, 2*self.n_hidden) 
            self.proj2 = nn.Linear(2*self.n_hidden, self.n_ans)  

    def get_network(self, self_type="", layers=-1):
        if self_type in ["kq","kque", "k_mem","k_que_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_K"]
        elif self_type in ["qk", "q_mem","que_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_Q"]
        elif self_type =="v_mem":
            embed_dim = 512
            attn_dropout = 0.2
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.cfg["MODEL"]["NUM_HEAD"],
            layers=max(self.cfg["MODEL"]["NUM_LAYER"], layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.cfg["MODEL"]["RELU_DROPOUT"],
            res_dropout=self.cfg["MODEL"]["RES_DROPOUT"],
            embed_dropout=self.cfg["MODEL"]["EMB_DROPOUT"],
            attn_mask=self.cfg["MODEL"]["ATTN_MASK"],
            fc_hid_coeff=self.cfg["MODEL"]["FC_HID_COEFF"],
        )

    def forward(self, batch):
        ques=batch[0]
        he_ques = batch[1]
        he_kg = batch[2]
        img=batch[3]
        

        num_batch = he_ques.shape[0]
        num_he_ques = he_ques.shape[1]
        num_he_kg = he_kg.shape[1]
        
        
        he_ques = torch.reshape(self.i2e(he_ques), (num_batch, num_he_ques, -1)) 
        he_kg = torch.reshape(self.i2e(he_kg), (num_batch, num_he_kg, -1))   

        he_ques = self.q2h(he_ques)  
        he_kg = self.k2h(he_kg)  

        
        ques_emb=self.i2e(ques)
        ques_emb=self.ques_trans(ques_emb)   

        
        v1=self.backbone1(img)  
        v2=self.backbone1_end(v1)  
        v2=self.v_dropout(v2)

        v11=self.conv_down1(v1)  
        v11=self.bn1(v11)
        v11=F.relu(v11)

        v1h=self.conv_downh(v1)  
        v1h=self.bnh(v1h)
        v1h=F.relu(v1h)

        v_q1_list=[]
        v_qh_list=[]
        for i in range(num_batch):
            v=v11[i]
            q_rh=he_ques[i].contiguous().view(self.vq_c_outh,self.conv_down_outchannelh,self.kernel1_size,self.kernel1_size) 
            q_r1=ques_emb[i].contiguous().view(self.vq_c_out1,self.conv_down_outchannel1,self.kernel1_size,self.kernel1_size)   
            v_q1=F.conv2d(v.unsqueeze(dim=0),q_r1,stride=1,padding=2)
            v_q1=F.relu(v_q1)
            v_q1=self.pool1(v_q1) 
            v_q1=self.feature_map_dropout(v_q1)  
            v_q1=torch.flatten(v_q1,1)  
            v_q1=self.fc1(v_q1)   
            v_q1_list.append(v_q1)

            vh=v1h[i]
            v_qh=F.conv2d(vh.unsqueeze(dim=0),q_rh,stride=1,padding=2)
            v_qh=F.relu(v_qh)
            v_qh=self.pool1(v_qh)  
            v_qh=self.feature_map_dropout(v_qh)  
            v_qh=torch.flatten(v_qh,1)  
            v_qh=self.fch(v_qh)   
            v_qh_list.append(v_qh)

        v1_q=torch.cat(v_q1_list,dim=0)  
        v1_qh=torch.cat(v_qh_list,dim=0) 
        v1_sum= torch.add(v1_q,v1_qh)
        v1_sum=self.v_dropout(v1_sum)

        
        
        v_emb=torch.add(self.proj_v2(v2),self.proj_v1(v1_sum),alpha=self.alpha)  
        
        v_emb=self.trans_v_mem(v_emb.unsqueeze(dim=0)).squeeze()

        he_ques = self.dropout(he_ques)
        he_kg = self.dropout(he_kg)

        he_ques = he_ques.permute(1, 0, 2)
        he_kg = he_kg.permute(1, 0, 2)

        if self.args.abl_only_ga == True:
            h_k_with_q = self.trans_k_with_q(he_kg, he_ques, he_ques)
            h_ks_sum = torch.sum(h_k_with_q, axis=0)
            h_q_with_k = self.trans_q_with_k(he_ques, he_kg, he_kg)
            h_qs_sum = torch.sum(h_q_with_k, axis=0)
            last_kq = torch.cat([h_ks_sum, h_qs_sum], dim=1)
            

        elif self.args.abl_only_sa == True:
            h_ks = self.trans_k_mem(he_kg)
            h_ks_sum = torch.sum(h_ks, axis=0)
            h_qs = self.trans_q_mem(he_ques)
            h_qs_sum = torch.sum(h_qs, axis=0)
            last_kq = torch.cat([h_ks_sum, h_qs_sum], dim=1)

        else: 
            h_k_with_q = self.trans_k_with_q(he_kg, he_ques, he_ques)  
            h_ks = self.trans_k_mem(h_k_with_q)  
            h_ks_sum = torch.sum(h_ks, axis=0)  


            h_q_with_k = self.trans_q_with_k(he_ques, he_kg, he_kg)
            h_qs = self.trans_q_mem(h_q_with_k)
            h_qs = self.trans_q_mem(he_ques)
            h_qs_sum = torch.sum(h_qs, axis=0)

            
            last_kq = torch.cat([v_emb,h_ks_sum, h_qs_sum], dim=1) 

        if self.args.abl_ans_fc != True:
            output = self.proj2(    
                F.dropout(
                    F.relu(self.proj1(last_kq)),
                    p=self.out_dropout,
                    training=self.training,
                )
            )
            pred = self.ans_selector(output)
        else:
            output = self.proj2(
                F.dropout(
                    F.relu(self.proj1(last_kq)),
                    p=self.out_dropout,
                    training=self.training,
                )
            )
            pred = F.log_softmax(output, dim=1)
        return pred

