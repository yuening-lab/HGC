import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import os, math
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torch
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
 
try:
    import dgl
except:
    print("<<< dgl is not imported >>>")
    pass
 
 
class causality_enhanced_message_passing(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False, device=torch.device("cpu")):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.device        = device
        self.use_norm    = use_norm
        self.time_emb    = None
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        self.cau_k_linears   = nn.ModuleDict()
        self.cau_q_linears   = nn.ModuleDict()
        self.cau_v_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            self.cau_k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.cau_q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.cau_v_linears[t] = nn.Linear(in_dim,   out_dim)
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)

        self.relation_msg = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        for etype in etypes:
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, 1))

        self.relation_msg_cau = nn.ParameterDict()
        for etype in ['tw','tt','td']:
            self.relation_msg_cau[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.cau_filter = nn.Parameter(torch.Tensor(3, n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def edge_attention(self, etype, inp_key):
        def msg_func(edges):
            relation_att = self.relation_att[etype] 
            query_prior =  (edges.data['weight'] * relation_att).permute(2,0,1).contiguous()
            relation_msg = self.relation_msg[etype] 
            att   = (((edges.dst['q']+query_prior) * edges.src['k'] ).sum(dim=-1) / self.sqrt_dk )#+ att0
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            if etype in ['tw','tt','td']:
                cau_types = edges.src['cau_type'] # 0,1,2,3  learn and mask out 0 type
                relation_msg_cau = self.relation_msg_cau[etype] 
                effect_mask = self.cau_filter[cau_types]
                n, n_head, d_k, _ = effect_mask.size()
                mul1 = edges.src['ck'].reshape(-1,1,d_k)
                mul2 = effect_mask.reshape(-1,d_k,d_k)
                masked_effect = torch.bmm(mul1,mul2)
                masked_effect = masked_effect.reshape(n,n_head,d_k) 
                cau_att   = (edges.dst['cq'] * masked_effect).sum(dim=-1)/ self.sqrt_dk
                cau_val   = torch.bmm(edges.src['cv'].transpose(1,0) + self.time_emb, relation_msg_cau).transpose(1,0)
                return {'a': att, 'v': val, 'ca':cau_att,'cv':cau_val}
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        if 'ca' in edges.data:
            return {'v': edges.data['v'], 'a': edges.data['a'], 'ca':edges.data.pop('ca'),'cv':edges.data.pop('cv')}
        return {'v': edges.data['v'], 'a': edges.data['a']}
     
    
    def reduce_func(self,etype):
        def reduce(nodes):
            att = F.softmax(nodes.mailbox['a'], dim=1)
            h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
            if 'ca' in nodes.mailbox:
                cau_att = F.softmax(nodes.mailbox['ca'], dim=1) 
                cau_h   = torch.sum(cau_att.unsqueeze(dim = -1) * nodes.mailbox['cv'], dim=1)
                h += cau_h
            return {'t': h.view(-1, self.out_dim)}
        return reduce

    def forward(self, G, inp_key, out_key):
        self.time_emb = G.time_emb
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype not in self.etypes:
                continue 
            edge_dict.append(etype)
            k_linear = self.k_linears[srctype]
            v_linear = self.v_linears[srctype] 
            q_linear = self.q_linears[dsttype]
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            if etype in ['tw','tt','td']:
                cau_k_linear = self.cau_k_linears[srctype]
                cau_v_linear = self.cau_v_linears[srctype] 
                cau_q_linear = self.cau_q_linears[dsttype]
                G.nodes[srctype].data['ck'] = cau_k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
                G.nodes[srctype].data['cv'] = cau_v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
                G.nodes[dsttype].data['cq'] = cau_q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.apply_edges(func=self.edge_attention(etype, inp_key), etype=etype)
           
        G.multi_update_all({etype : (self.message_func, self.reduce_func(etype)) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        
        for ntype in G.ntypes: 
            trans_out = G.nodes[ntype].data.pop('t') 
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)
 
 
class TemporalEncoding(nn.Module):
    def __init__(self, n_inp, max_len = 7, dropout = 0.2):
        super(TemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_inp, 2) *
                             -(math.log(10000.0) / n_inp))
        emb = nn.Embedding(max_len, n_inp)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_inp)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_inp)
        emb.requires_grad = False
        self.emb = emb
    def forward(self, t):
        return self.emb(t)

 
class causality_enhanced_hetero_graph_model(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.time_emb = TemporalEncoding(n_hid // n_heads, seq_len) 
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        }) 
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wd','td','tt','ww','tw','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(causality_enhanced_message_passing(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
            
        self.out_layer = nn.Linear(n_hid*3, 1) 
        self.threshold = 0.5
        self.criterion = F.binary_cross_entropy_with_logits 
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, g_list, y_data): 
        bg = dgl.batch(g_list).to(self.device)  
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        effect = bg.nodes['topic'].data['effect']#.to_dense()
        effect = (effect >0)*1. + (effect < 0)*(-1.)
        effect = effect.sum(-1)
        effect = ((effect > 0)*1.) + ((effect < 0)*2.)
 
        bg.nodes['topic'].data['cau_type'] = effect.long() 
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,
                                        }, 
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID]  
            time_emb = self.time_emb(torch.tensor(curr_time).to(self.device))
            sub_bg.time_emb = time_emb
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            for ntype in ['word','topic']:
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)

        y_pred = self.out_layer(global_info)
        loss = self.criterion(y_pred.view(-1), y_data) 
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred


class message_passing_ablation_no_cau(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False, device=torch.device("cpu")):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.device        = device
        self.use_norm    = use_norm
        self.time_emb    = None
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)

        self.relation_msg = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        for etype in etypes:
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, 1))

        self.drop           = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def edge_attention(self, etype, inp_key):
        def msg_func(edges):
            relation_att = self.relation_att[etype] 
            query_prior =  (edges.data['weight'] * relation_att).permute(2,0,1).contiguous()
            relation_msg = self.relation_msg[etype] 
            att   = (((edges.dst['q']+query_prior) * edges.src['k'] ).sum(dim=-1) / self.sqrt_dk )#+ att0
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}
     
    def reduce_func(self,etype):
        def reduce(nodes):
            att = F.softmax(nodes.mailbox['a'], dim=1)
            h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
            return {'t': h.view(-1, self.out_dim)}
        return reduce

    def forward(self, G, inp_key, out_key):
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype not in self.etypes:
                continue 
            edge_dict.append(etype)
            k_linear = self.k_linears[srctype]
            v_linear = self.v_linears[srctype] 
            q_linear = self.q_linears[dsttype]
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.apply_edges(func=self.edge_attention(etype, inp_key), etype=etype)
           
        G.multi_update_all({etype : (self.message_func, self.reduce_func(etype)) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        
        for ntype in G.ntypes: 
            trans_out = G.nodes[ntype].data.pop('t') 
            trans_out = F.relu(trans_out)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)


class model_ablation_no_cau(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.seq_len = seq_len
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1, n_hid))
        self.time_emb = TemporalEncoding(n_hid // n_heads, seq_len) 
        self.temp_skip = nn.ParameterDict({
                'word': nn.Parameter(torch.ones(1)),
                'topic': nn.Parameter(torch.ones(1)),
        }) 
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wd','td','tt','ww','tw','dw']

        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(message_passing_ablation_no_cau(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm, device=self.device))
        self.out_layer = nn.Sequential(
                nn.Linear(n_hid*3, 1) 
        )
        self.threshold = 0.5
        self.criterion = F.binary_cross_entropy_with_logits #soft_cross_entropy
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, g_list, y_data): 
        bg = dgl.batch(g_list).to(self.device)  
        word_emb = self.word_embeds[bg.nodes['word'].data['id'].long()].view(-1, self.word_embeds.shape[1])
        topic_emb = self.topic_embeds[bg.nodes['topic'].data['id'].long()].view(-1, self.topic_embeds.shape[1])
        doc_emb = self.doc_gen_embeds.repeat(bg.number_of_nodes('doc'),1)
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        bg.nodes['topic'].data['h0'] = topic_emb
        bg.nodes['doc'].data['h0'] = doc_emb 
        # word and topic take info from last time step
        out_key_dict = {'doc':'ht','topic':'ht-1','word':'ht-1'}
     
        for ntype in ['word','topic']: 
            bg.nodes[ntype].data['ht-1'] = torch.zeros(bg.nodes[ntype].data['h0'].size()).to(self.device)
        
        bg.nodes['doc'].data['ht'] = bg.nodes['doc'].data['h0']

        tt_edges_idx = list(range(len(bg.edges(etype='tt'))))
        for curr_time in range(self.seq_len):
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wt_edges_idx = (bg.edges['wt'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            wd_edges_idx = (bg.edges['wd'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            td_edges_idx = (bg.edges['td'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        ('topic', 'tt', 'topic'): tt_edges_idx,
                                        ('word', 'wt', 'topic'): wt_edges_idx,
                                        ('topic', 'td', 'doc'): td_edges_idx,
                                        ('word', 'wd', 'doc'):wd_edges_idx,
                                        ('topic', 'tw', 'word'): wt_edges_idx,},)
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID]  
            for i in range(self.n_layers):
                if i == 0:
                    self.gcs[i](sub_bg, 'h0', 'ht')
                else:
                    self.gcs[i](sub_bg, 'ht', 'ht')
            for ntype in ['word','topic']:
                alpha = torch.sigmoid(self.temp_skip[ntype])
                sub_bg.nodes[ntype].data['ht-1'] = alpha * sub_bg.nodes[ntype].data['ht'] + (1-alpha) * sub_bg.nodes[ntype].data['ht-1']
            
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                bg.nodes[ntype].data[key][orig_node_ids[ntype].long()] = sub_bg.nodes[ntype].data[key]
            
        if self.pool == 'max':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.max_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)
        elif self.pool == 'mean':
            global_info = []
            for ntype in out_key_dict:
                key = out_key_dict[ntype]
                global_info.append( dgl.mean_nodes(bg, feat=key,ntype=ntype))
            global_info = torch.cat(global_info,-1)

        y_pred = self.out_layer(global_info)
        loss = self.criterion(y_pred.view(-1), y_data) 
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred
