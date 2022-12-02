## Environment
```
python 3.7
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```
and then install matplotlib, sklearn and so on.

## Dataset
Consider ```./``` as ```/src```.
Datasets can be found in ```../dataset/```

## Run Experiment
### Learning node representation
First run
```
python main.py --experiment_type cf
```
make directory ```./models_save/``` if you don't.

and then
```
python main.py --experiment_type train
```
The subgraphs will be generated under ```./graphFair_subgraph/``` at the first time of running. If the files already exist, the subgraph data will be directly loaded. The true counterfactuals for evaluation are in ```./graphFair_subgraph/cf/```, and the augmented data is in ```./graphFair_subgraph/aug/```. The trained GEAR model can be saved in ```./models_save/```.

### Refenrences
The code is the implementation of this paper:
```
[1] J. Ma, R. Guo, M. Wan, L. Yang, A. Zhang, and J. Li. Learning fair node representations with graph counterfactual fairness. In Proceedings of the 15th WSDM, 2022
```
