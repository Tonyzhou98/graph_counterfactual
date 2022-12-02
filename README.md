## Environment
```
python 3.7
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge
```
and then install matplotlib, sklearn and so on.

## Dataset
Consider ```./``` as ```/src```.
Datasets can be found in ```../dataset/```

## Run Experiment
### Learning node representation
First run
```
python main.py --experiment_type train
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
