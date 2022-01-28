# Out-of-scope intent recognition through hierarchical modeling
User queries for a real-world dialog system may sometimes fall outside the scope of the system's capabilities.
This project focuses on out-of-scope intent classification in dialog systems,
and presents a hierarchical joint model to classify domain and intent simultaneously,
where the novelties include: (1) sharing out-of-scope signals in joint modeling of domain and intent classification; and
(2) introducing a hierarchical model that learns the intent and domain representations in the higher and lower layers respectively.

## 1. Source Code
The main source code contains models.py, train.py and utils.py:
(1) The BERT and BERT-Joint model is defined in models.py, covering various model structures.
(2) The joint training procedure, early stopping and hyperparamters are coded in train.py.
(3) The threshold-based post-processing method, the evaluation metrics and other utitlity methods are defined in utils.py.

## 2. Dataset
The dataset folder contains the four variants of the OOS dataset. The training, validation and testing set are organized in separate csv files.
The original dataset in JSON is from https://github.com/clinc/oos-eval, please cite the corresponding paper if you use the dataset.

## 3. How to Run
### Install
Please first install the necessary packages using requirements.txt.
```
pip install -r requirements.txt
```

To train and evaluate on the oos\_full dataset, please run the following commands:
### Train a Model
```
make train dataset=oos_full
```
### Evaluate a Model
```
make predict dataset=oos_full
```


## 4. Citation
If you find this project useful or use it in your own work, please cite the following paper:
```
@article{liu2021hierarchical,
  title={Out-of-Scope Domain and Intent Classification through Hierarchical Joint Modeling},
  author={Liu, Pengfei and Li, Kun and Meng, Helen},
  journal={arXiv preprint arXiv:2104.14781},
  year={2021}
}
```

## 5. License
This repository is released under the GNU General Public License v3.0.
