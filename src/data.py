import pickle
import torch
from torch.utils import data
import pickle


class GraphData(data.Dataset):
      def __init__(self, path, dataset, datafiles, horizon, causalfiles=''):
            datafile_list = datafiles.split(',')
            datafile_list.sort()
            y_data = []
            g_data = []
            t_data = []
            for datafile in datafile_list:
                  with open('{}/{}/{}.pkl'.format(path, dataset,datafile),'rb') as f:
                        graph_list = pickle.load(f)
                  g_data += graph_list
                  
                  tmp = datafile.split('_')
                  tmp[0] = 'attr'
                  attr_file = '_'.join(tmp)
                  with open('{}/{}/{}.pkl'.format(path, dataset, attr_file),'rb') as f:
                        attr_dict = pickle.load(f)
                  y_data.append(attr_dict['y'])
                  t_data +=  attr_dict['date']
            y_data = torch.cat(y_data,dim=0)
            y_data = y_data[:,:horizon].sum(-1)

            y_data = torch.where(y_data > 0,1.,0.)
            self.len = len(y_data)
            self.y_data = y_data
            self.g_data = g_data
            self.t_data = t_data
            print(len(self.g_data),'self.g_data', 'self.y_data',self.y_data.shape)
            print('positive',y_data.mean()) 
            #load causal
            splitted_date_lists = [
            '2013-01-01','2013-04-01','2013-07-01','2013-10-01',
            '2014-01-01','2014-04-01','2014-07-01','2014-10-01',
            '2015-01-01','2015-04-01','2015-07-01','2015-10-01',
            '2016-01-01','2016-04-01','2016-07-01','2016-10-01',
            '2017-01-01','2017-04-01'
            ]
            self.splitted_date_lists = splitted_date_lists
            if causalfiles != '':
                  print('{}/{}/{}.pkl'.format(path, dataset,causalfiles),'=====')
                  with open('{}/{}/{}.pkl'.format(path, dataset,causalfiles),'rb') as f:
                        causal_time_dict = pickle.load(f)
                  self.causal_time_dict = causal_time_dict
            else:
                  self.causal_time_dict = {}
      def __len__(self):
            return self.len

      def __getitem__(self, index):
            if self.causal_time_dict == {}:
                  return self.g_data[index], self.y_data[index]

            g = self.g_data[index]
            date = self.t_data[index]
            for end_date in self.splitted_date_lists:  
                  if date < end_date:
                        cur_end_date = end_date
                        break
            causal_weight = self.causal_time_dict[cur_end_date]
            causal_weight_tensor = torch.from_numpy(causal_weight) 
            if isinstance(g, list):
                  for i in range(len(g)):
                        g[i].nodes['topic'].data['effect'] = causal_weight_tensor[g[i].nodes('topic').numpy()]#.to_sparse()
            else:
                  g.nodes['topic'].data['effect'] = causal_weight_tensor[g.nodes('topic').numpy()] 
            return g, self.y_data[index]


def collate_2(batch):
    g_data = [item[0] for item in batch]
    y_data = [item[1] for item in batch]
    return [g_data, y_data]