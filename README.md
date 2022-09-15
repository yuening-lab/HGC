# HGC 

A dynamic heterogeneous graph model with causality enhanced node representations.

This is the source code for paper [Causality Enhanced Societal Event Forecasting
With Heterogeneous Graph Learning](https://icdm22.cse.usf.edu/) appeared in IEEE ICDM22

## Prerequisites
The code has been successfully tested in the following environment. (For older dgl versions, you may need to modify the code)
- Python 3.7.9
- PyTorch 1.7.0+cu92
- dgl 0.5.2
- Sklearn 0.23.2 

## Data
The experiments are conducted on four event datasets collected from Integrated Conflict Early Warning System (ICEWS). These events are encoded into 20 main categories (e.g., protest, demand, appeal) using Conflict and Mediation Event Observations (CAMEO) event codes.
Please find example datasets in this [Google Drive Link](https://drive.google.com/drive/folders/xxxxx?usp=sharing)(TODO). A brief introduction of the data files is as follows:
- `dyn_tf_2014-2015_900.pkl` A list of dynamic heterogeneous graphs constructed for samples from 2014 to 2015. The number of word nodes does not exceed 900.
<!-- - `sta_tf_2014-2015_900.pkl` A list of static heterogeneous graphs constructed for samples from 2014 to 2015. The number of word nodes does not exceed 900. -->
- `attr_tf_2014-2015_900.pkl` Date and target event (label) information for heterogeneous graphs.
- `causal_topics_0.01.pkl`  Evolving and Multi-view Causal Topics. 0.01 means the significance level is  99%.
- `word_emb_300.pkl` Word embeddings.



## Getting Started
### Prepare your code
Clone this repo.
```bash
git clone https://github.com/yuening-lab/HGC
cd HGC
```
### Prepare your data
Download the dataset (e.g., `THA_w7h7`) from the given link and store them in `data` filder. Or prepare your own dataset in a similar format. The folder structure is as follows:
```sh
- HGC
	- data
		- THA_w7h7
		- ...
	- src
```

### Training and testing
Please run following commands for training and testing under the `src` folder. We take the dataset `THA_w7h7` as the example.

**Evaluate baseline models (Examples)**
<!-- *GAT*
```python
python train.py --dataset THA_w7h7 --datafiles sta_tf_2014-2015_900,sta_tf_2015-2016_900,sta_tf_2016-2017_900 --horizon 5 --gpu 0 -m gat --n-hidden 64 --n-layers 2 --note "" --train 0.4 --patience 15
``` -->
*EvolveGCN*
```python
python train.py --dataset THA_w7h7 --datafiles dyn_tf_2014-2015_900,dyn_tf_2015-2016_900,dyn_tf_2016-2017_900 --horizon 5 --gpu 1 -m evolvegcn --n-hidden 64 --n-layers 1 --note "" --train 0.4 --patience 15
```
*HGT*
```python
python train.py --dataset THA_w7h7 --datafiles dyn_tf_2014-2015_900,dyn_tf_2015-2016_900,dyn_tf_2016-2017_900 --horizon 5 --gpu 1 -m temphgt --n-hidden 64 --n-layers 1 --note "" --train 0.4 --patience 15
```

**Evaluate the HGC model**

*Full model*
```python
python train.py --dataset THA_w7h7 --datafiles dyn_tf_2014-2015_900,dyn_tf_2015-2016_900,dyn_tf_2016-2017_900 --horizon 5 --gpu 5 -m hgc --n-hidden 64 --n-layers 1 --note "cau0.05" --train 0.4 --n-topics 50 --causalfiles causal_topics_0.05 --patience 15
```

*Variant model wthout causal*
```python
python train.py --dataset THA_w7h7 --datafiles dyn_tf_2014-2015_900,dyn_tf_2015-2016_900,dyn_tf_2016-2017_900 --horizon 5 --gpu 6 -m hgc_no_cau --n-hidden 64 --n-layers 1 --note "" --train 0.4 --n-topics 50  --patience 15
```


## Cite

Please cite our paper if you find this code useful for your research:
```
The reference will be updated soon.
```
