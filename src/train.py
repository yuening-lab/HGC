def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse, os, time, csv, pickle
import numpy as np
from sklearn.utils import shuffle

 
parser = argparse.ArgumentParser(description='')
parser.add_argument("--dp", type=str, default="../data", help="data path")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--n-hidden", type=int, default=32, help="number of hidden units")
parser.add_argument("--n-layers", type=int, default=2, help="number of hidden layers")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight for L2 loss")
parser.add_argument("-d", "--dataset", type=str, default='THA_w7h7_minday3', help="dataset to use")
parser.add_argument("-df", "--datafiles", type=str, default='dyn_tf_2014-2015_900,dyn_tf_2015-2016_900,dyn_tf_2016-2017_900', help="")
parser.add_argument("-cf", "--causalfiles", type=str, default='', help="causality to use")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--n-epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--seq-len", type=int, default=7)
parser.add_argument("--horizon", type=int, default=7)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--pool", type=str, default='mean')
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--seed", type=int, default=999, help='random seed')
parser.add_argument("--runs", type=int, default=5, help='number of runs')
parser.add_argument("-m","--model", type=str, default="hgc", help="model name")
parser.add_argument("--train", type=float, default=0.6, help="")
parser.add_argument("--val", type=float, default=0.2, help="")
parser.add_argument('--shuffle', action="store_false")
parser.add_argument("--note", type=str, default="", help="")
parser.add_argument("--n-topics", type=int, default=50, help='number of topics')
parser.add_argument("--n-heads", type=int, default=4, help='number of attention heads')

 

args = parser.parse_args()
print(args)


os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from model_baselines import *
from model_ours import *
from utils import *
from data import *

use_cuda = args.gpu >= 0 and torch.cuda.is_available()

print("cuda",use_cuda)

torch.manual_seed(args.seed) 
random.seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
print('device',device)
with open('{}/{}/word_emb_300.pkl'.format(args.dp,args.dataset), 'rb') as f:
    word_embeds = pickle.load(f)
print('load word_emb_300.pkl')
word_embeds = torch.FloatTensor(word_embeds)
vocab_size = word_embeds.size(0)
emb_size = word_embeds.size(1)

 
static_graph_dataset = GraphData(args.dp, args.dataset,args.datafiles, args.horizon,args.causalfiles)

dataset_size = len(static_graph_dataset)
indices = list(range(dataset_size))
split1 = int(np.floor(args.train * dataset_size))
split2 = int(np.floor((args.val+args.train) * dataset_size))
if args.shuffle:
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]
else:
    test_indices = indices[split2:]
    train_val_indices = indices[:split2]
    np.random.seed(args.seed)
    np.random.shuffle(train_val_indices)
    train_indices, val_indices = train_val_indices[:split1], train_val_indices[split1:split2]
    
print(len(train_indices),len(val_indices),len(test_indices))
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(static_graph_dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_2,sampler=train_sampler)
valid_loader = DataLoader(static_graph_dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_2,sampler=valid_sampler)
test_loader = DataLoader(static_graph_dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_2, sampler=test_sampler)
train_loader.len = len(train_indices)
valid_loader.len = len(val_indices)
test_loader.len = len(test_indices)
 
def prepare(args,word_embeds,device): 
    if args.model == 'gat':
        model = GAT(in_feats=emb_size, n_hidden=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, heads=args.n_heads,device=device, dropout=args.dropout,pool=args.pool)
    elif args.model == 'rgcn':
        model = RGCN(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, 
        vocab_size=vocab_size, device=device, num_topic=args.n_topics, dropout=args.dropout,pool=args.pool) 
    elif args.model == 'temphgt':
        model = TempHGT(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, seq_len=args.seq_len,device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool, use_norm = True)
    elif args.model == 'evolvegcn':
        model = EvolveGCN(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, activation=F.relu, seq_len=args.seq_len,device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout,pool=args.pool)
    elif args.model == 'hgc':
        model = causality_enhanced_hetero_graph_model(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, seq_len=args.seq_len,device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout, pool=args.pool, use_norm = True)
    elif args.model == 'hgc_no_cau':
        model = model_ablation_no_cau(n_inp=emb_size, n_hid=args.n_hidden, n_layers=args.n_layers, n_heads=args.n_heads, seq_len=args.seq_len,device=device, 
        num_topic=args.n_topics, vocab_size=vocab_size, dropout=args.dropout, pool=args.pool, use_norm = True)
f
    model_name = model.__class__.__name__
    # print(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#params:', total_params)
    token = '{}_seed{}topic{}w{}h{}lr{}bs{}p{}hid{}l{}tr{}va{}po{}nh{}dp{}'.format(model_name, args.seed, args.n_topics, args.seq_len, args.horizon,args.lr,args.batch_size,args.patience,args.n_hidden,args.n_layers,args.train,args.val,args.pool,args.n_heads,args.dropout)
    if args.shuffle is False:
        token += '_noshuf'
    if args.note != "":
        token += args.note
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/' + args.dataset, exist_ok=True)
    os.makedirs('models/{}/{}'.format(args.dataset, token), exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/' + args.dataset, exist_ok=True)

    result_file = 'results/{}/{}.csv'.format(args.dataset,token)
    model_state_file = 'models/{}/{}.pth'.format(args.dataset, token)

    if use_cuda:
        model.cuda()
        word_embeds = word_embeds.cuda()
    model.word_embeds = word_embeds
    return model, optimizer, result_file, token
 

epoch = 0
def train(train_loader):
    model.train()
    total_loss = 0
    t0 = time.time()
    for i, batch in enumerate(train_loader):
        g_data, y_data = batch
        y_data = torch.stack(y_data, dim=0).to(device)
        loss, y_pred = model(g_data, y_data) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    t2 = time.time()
    reduced_loss = total_loss / (train_loader.len / args.batch_size)
    print("Epo {:04d} | Loss {:.6f} | time {:.2f} {}".format(epoch, reduced_loss, t2 - t0, time.ctime()))
    return reduced_loss

@torch.no_grad()
def eval(data_loader, set_name='valid'):
    model.eval()
    y_true_l, y_pred_l = [], []
    total_loss = 0
    for i, batch in enumerate(data_loader):
        g_data, y_data = batch
        y_data = torch.stack(y_data, dim=0).to(device)
        loss, y_pred = model(g_data, y_data) 
        y_true_l.append(y_data)
        y_pred_l.append(y_pred)
        total_loss += loss.item()

    y_true_l = torch.cat(y_true_l,0).cpu().detach().numpy() 
    y_pred_l = torch.cat(y_pred_l,0).cpu().detach().numpy() 
    eval_dict = eval_bi_classifier(y_true_l, y_pred_l)
    reduced_loss = total_loss / (data_loader.len / args.batch_size)
    return reduced_loss, eval_dict

 

for i in range(args.runs):
    model, optimizer, result_file, token = prepare(args, word_embeds, device)
    print('========= Run i = {} on Dataset {} {} ========='.format(i,args.dataset,token))
    model_state_file = 'models/{}/{}/{}.pth'.format(args.dataset, token, i)
    if i == 0 and os.path.exists(result_file):  
        os.remove(result_file)
        
    bad_counter = 0
    loss_small = float('inf')
    value_large = float('-inf')
    try:
        print('begin training ...')
        for epoch in range(0, args.n_epochs):
            epoch_start_time = time.time()
            train_loss = train(train_loader)
            valid_loss, eval_dict = eval(valid_loader, set_name='val')
            small_value = valid_loss 
            if small_value < loss_small:
                loss_small = small_value
                bad_counter = 0
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                print('Epo {:04d} tr_los:{:.5f} val_los:{:.5f} '.format(epoch, train_loss, valid_loss),'|'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
        print("training done")
            
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early, epoch',epoch)
    
    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    f = open(result_file,'a')
    wrt = csv.writer(f)
    
    print("Test using best epoch: {}".format(checkpoint['epoch']))
    val_loss, eval_dict = eval(valid_loader, 'val')
    print('Val','|'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))

    _, eval_dict = eval(test_loader, 'test')
    print('Test','|'.join(['{}:{:.4f}'.format(k, eval_dict[k]) for k in eval_dict]))
    test_res = [eval_dict[k] for k in eval_dict]
    wrt.writerow([val_loss] + [0] + test_res)
    f.close()


with open(result_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    arr = []
    for row in csv_reader:
        arr.append(list(map(float, row)))
arr = np.array(arr)
line_count = arr.shape[0]
mean = [round(float(v),3) for v in arr.mean(0)]
std = [round(float(v),3) for v in arr.std(0)]
res = [str(mean[i]) +' ' + str(std[i]) for i in range(len(mean))]
print(res)


all_res_file = 'results/{}/{}.csv'.format(args.dataset,args.model)
f = open(all_res_file,'a')
wrt = csv.writer(f)
wrt.writerow([token] + [line_count] + res)
f.close()
print(token)
 