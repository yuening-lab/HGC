import  time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.nn import GATConv


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h, ntype, etype):
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        g.nodes[ntype].data['h'] = h
        g.update_all(
                        fn.u_mul_e('h', 'weight', 'm'),
                        fn.sum(msg='m', out='h'),etype=etype)
        h = g.nodes[ntype].data.pop('h')
        # normalization by square root of dst degree
        h = h * g.nodes[ntype].data['norm'].unsqueeze(1)
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_dim={}, out_dim={})'.format(
            self.__class__.__name__, self.in_feats, self.out_feats)

 
class GATNet(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation,allow_zero_in_degree=True))
        # hidden layers
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation,allow_zero_in_degree=True))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        return h


class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, heads, activation, device, seq_len=7, vocab_size=15000,dropout=0.5,pool='max'):
        super().__init__()
        self.in_feats = in_feats
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.device = device
        self.pool = pool
        self.word_embeds = None 
        self.layers = nn.ModuleList()
        heads_list = [heads for i in range(n_layers)]
        self.conv = GATNet(n_layers,in_feats, n_hidden, heads_list,activation,dropout,dropout,0.2,False)
        self.out_layer = nn.Linear(n_hidden*heads, 1) 
        self.threshold = 0.5
        self.out_func = torch.sigmoid
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
        h = self.word_embeds[bg.nodes['word'].data['id']].view(-1, self.word_embeds.shape[1])
        sub_g = dgl.edge_type_subgraph(bg, [('word', 'ww', 'word')])
        h_sub_g = dgl.to_homogeneous(sub_g)
        h = self.conv(h_sub_g, h)
        bg.nodes['word'].data['h'] = h
        if self.pool == 'max':
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
        elif self.pool == 'mean':
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
        y_pred = self.out_layer(global_word_info)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred) 
        return loss, y_pred


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout, etypes, ntypes):
        super().__init__()
        self.etypes = etypes
        self.ntypes = ntypes
        self.weight = nn.ModuleDict()
        for etype in etypes:
            self.weight[etype] = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.drop = nn.Dropout(dropout)

    def forward(self, G, feat_dict):

        for ntype in feat_dict:
            G.nodes[ntype].data['h'] = feat_dict[ntype]
        funcs={}
        for srctype, etype, dsttype in G.canonical_etypes:
            if etype not in self.etypes:
                continue
            Wh = self.weight[etype](G.nodes[srctype].data['h'])   #   feat_dict[srctype]
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')

        for ntype in self.ntypes:
            feat = G.nodes[ntype].data['h']
            G.nodes[ntype].data['h'] = self.drop(self.activation(feat))

        return {ntype : G.nodes[ntype].data['h'] for ntype in self.ntypes}


class RGCN(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, activation, device, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super(RGCN, self).__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.device = device
        self.pool = pool
        self.activation = activation
        self.word_embeds = None
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1,n_hid))
        self.hetero_layers = nn.ModuleList()
        self.adapt_ws = nn.Linear(n_inp, n_hid)
        for _ in range(n_layers):
            self.hetero_layers.append(HeteroRGCNLayer(n_hid, n_hid, activation, dropout, ['ww','wt','tt','wd','td','tw','dw','dt'],['word','topic','doc']))
        self.out_layer = nn.Linear(n_hid*3, 1)  
        self.threshold = 0.5
        self.out_func = torch.sigmoid
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
        feat_dict = {'word':word_emb, 'topic':topic_emb, 'doc':doc_emb}
        for layer in self.hetero_layers:
            feat_dict = layer(bg, feat_dict) 
        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='h',ntype='doc')
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
            global_topic_info = dgl.max_nodes(bg, feat='h',ntype='topic')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='h',ntype='doc')
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
            global_topic_info = dgl.mean_nodes(bg, feat='h',ntype='topic')
        global_info = torch.cat((global_doc_info, global_word_info, global_topic_info),-1)
        y_pred = self.out_layer(global_info)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

 
class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False):
        super().__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        self.skip = nn.ParameterDict() 
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            self.skip[t] = nn.Parameter(torch.ones(1))
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        
        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def edge_attention(self, etype):
        def msg_func(edges):
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': F.relu(h.view(-1, self.out_dim))}
        
    def forward(self, G, inp_key, out_key):
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            edge_dict.append(etype)
            # print(srctype, etype, dsttype) 
            k_linear = self.k_linears[srctype]
            v_linear = self.v_linears[srctype] 
            q_linear = self.q_linears[dsttype]
            
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            G.apply_edges(func=self.edge_attention(etype), etype=etype)
           
        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        for ntype in G.ntypes:
            alpha = torch.sigmoid(self.skip[ntype])
            trans_out = self.a_linears[ntype](G.nodes[ntype].data.pop('t'))
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)


class TempHGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout = 0.5, use_norm = False):
        super().__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.etypes        = etypes
        self.ntypes        = ntypes
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.use_norm    = use_norm
        self.k_linears   = nn.ModuleDict()
        self.q_linears   = nn.ModuleDict()
        self.v_linears   = nn.ModuleDict()
        self.a_linears   = nn.ModuleDict()
        self.norms       = nn.ModuleDict()
        self.skip = nn.ParameterDict() 
        for t in ntypes:
            self.k_linears[t] = nn.Linear(in_dim,   out_dim)
            self.q_linears[t] = nn.Linear(in_dim,   out_dim)
            self.v_linears[t] = nn.Linear(in_dim,   out_dim)
            self.a_linears[t] = nn.Linear(out_dim,  out_dim)
            self.skip[t] = nn.Parameter(torch.ones(1))
            if use_norm:
                self.norms[t] = nn.LayerNorm(out_dim)
        
        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in etypes:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def edge_attention(self, etype):
        def msg_func(edges):
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
            key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
            att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
            return {'a': att, 'v': val}
        return msg_func
    
    def message_func(self, edges):
        if 'timeh' in edges.data:
            edges.data['v'] += edges.data['timeh'].unsqueeze(1)
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': F.relu(h.view(-1, self.out_dim))}
        
    def forward(self, G, inp_key, out_key):
        edge_dict = []
        for srctype, etype, dsttype in G.canonical_etypes:
            edge_dict.append(etype)
            # print(srctype, etype, dsttype) 
            k_linear = self.k_linears[srctype]
            v_linear = self.v_linears[srctype] 
            q_linear = self.q_linears[dsttype]
            
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            G.apply_edges(func=self.edge_attention(etype), etype=etype)
           
        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        for ntype in G.ntypes:
            alpha = torch.sigmoid(self.skip[ntype])
            trans_out = self.a_linears[ntype](G.nodes[ntype].data.pop('t'))
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[ntype](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

# https://github.com/acbull/pyHGT/blob/f7c4be620242d8c1ab3055f918d4c082f5060e07/pyHGT/conv.py#L283
class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 7, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    def forward(self, t):
        # return x + self.lin(self.emb(t))
         return self.lin(self.emb(t))

 
class TempHGT(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, n_heads, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max', use_norm = True):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_topic = num_topic
        self.device = device
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.word_embeds = None
        self.topic_embeds = nn.Parameter(torch.Tensor(num_topic, n_hid))
        self.doc_gen_embeds = nn.Parameter(torch.Tensor(1,n_hid))
        self.time_emb = RelTemporalEncoding(n_hid//n_heads,seq_len)
        self.adapt_ws  = nn.Linear(n_inp,  n_hid)
        etypes = ['wt','wd','td','tt','ww','tw','dt','dw']
        ntypes = ['word','topic','doc']
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(TempHGTLayer(n_hid, n_hid, ntypes, etypes, n_heads, use_norm = use_norm))
        self.out_layer =  nn.Linear(n_hid*3, 1) 
        self.threshold = 0.5
        self.out_func = torch.sigmoid
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
        bg.nodes['word'].data['h'] = word_emb
        bg.nodes['topic'].data['h'] = topic_emb
        bg.nodes['doc'].data['h'] = doc_emb
        ww_time = self.time_emb(bg.edges['ww'].data['time'].long())
        bg.edges['ww'].data['timeh'] = ww_time
        wd_time = self.time_emb(bg.edges['wd'].data['time'].long())
        bg.edges['wd'].data['timeh'] = wd_time
        bg.edges['dw'].data['timeh'] = wd_time
        wt_time = self.time_emb(bg.edges['wt'].data['time'].long())
        bg.edges['wt'].data['timeh'] = wt_time
        bg.edges['tw'].data['timeh'] = wt_time
        td_time = self.time_emb(bg.edges['td'].data['time'].long())
        bg.edges['td'].data['timeh'] = self.time_emb(bg.edges['td'].data['time'].long())
        bg.edges['dt'].data['timeh'] = td_time
        for i in range(self.n_layers):
            self.gcs[i](bg, 'h', 'h')

        if self.pool == 'max':
            global_doc_info = dgl.max_nodes(bg, feat='h',ntype='doc')
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
            global_topic_info = dgl.max_nodes(bg, feat='h',ntype='topic')
        elif self.pool == 'mean':
            global_doc_info = dgl.mean_nodes(bg, feat='h',ntype='doc')
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
            global_topic_info = dgl.mean_nodes(bg, feat='h',ntype='topic')
        global_info = torch.cat((global_doc_info, global_word_info, global_topic_info),-1)
        y_pred = self.out_layer(global_info)
        loss = self.criterion(y_pred.view(-1), y_data)
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred
 

class EvolveGCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(EvolveGCNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.

    def forward(self, g, h, ntype, etype, weight):
        h = torch.mm(h, weight)
        g.nodes[ntype].data['h'] = h
        g.update_all(
                    fn.u_mul_e('h', 'weight', 'm'),
                    fn.sum(msg='m', out='h'),etype=etype)
        h = g.nodes[ntype].data.pop('h')
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_dim={}, out_dim={})'.format(
            self.__class__.__name__, self.in_feats, self.out_feats)


# EvolveGCN
# only use words, keep nodes
class EvolveGCN(nn.Module):
    def __init__(self, n_inp, n_hid, n_layers, activation, device, seq_len, num_topic=50, vocab_size=15000, dropout=0.5, pool='max'):
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
         
        self.adapt_ws  = nn.Linear(n_inp,  n_hid) 
        # input layer
        self.layers = nn.ModuleList()
        # self.weights = nn.ModuleList()
        self.lstmCell = nn.ModuleList()
        self.weights = nn.Parameter(torch.Tensor(self.n_layers,n_hid, n_hid))
        for i in range(self.n_layers):
            self.layers.append(EvolveGCNLayer(n_hid, n_hid, activation, dropout))
            self.lstmCell.append(nn.LSTMCell(n_hid, n_hid))
        self.out_layer = nn.Linear(n_hid, 1) 
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
        word_emb = self.adapt_ws(word_emb)
        bg.nodes['word'].data['h0'] = word_emb
        bg.nodes['word'].data['h'] = word_emb
        hx_list = self.weights
        cx_list = []
        for curr_time in range(0,self.seq_len):
            ww_edges_idx = (bg.edges['ww'].data['time']==curr_time).nonzero(as_tuple=False).view(-1).cpu().detach().tolist()
            if len(ww_edges_idx) <= 0:
                continue
            bg_cpu = bg.to('cpu')
            sub_bg = dgl.edge_subgraph(bg_cpu, 
                                        {('word', 'ww', 'word'): ww_edges_idx,
                                        }
                                        )
            sub_bg = sub_bg.to(self.device)
            orig_node_ids = sub_bg.ndata[dgl.NID] 
            h = sub_bg.nodes['word'].data['h']
            new_hx_list = []
            
            for i in range(self.n_layers):
                hx = hx_list[i] # hx1
                if curr_time == 0 or len(cx_list) == 0:
                    hx, cx = self.lstmCell[i](hx) 
                else:
                    cx = cx_list[i]
                    hx, cx = self.lstmCell[i](hx, (hx, cx))
                new_hx_list.append(hx)
                cx_list.append(cx)
                h = self.layers[i](sub_bg, h, 'word','ww', hx)
            hx_list = new_hx_list
            bg.nodes['word'].data['h'][orig_node_ids['word'].long()] = h
            
        if self.pool == 'max':
            global_word_info = dgl.max_nodes(bg, feat='h',ntype='word')
        elif self.pool == 'mean':
            global_word_info = dgl.mean_nodes(bg, feat='h',ntype='word')
        y_pred = self.out_layer(global_word_info)
        loss = self.criterion(y_pred.view(-1), y_data) 
        y_pred = torch.sigmoid(y_pred)
        return loss, y_pred

