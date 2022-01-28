# Copyright (c) 2021 Pengfei Liu. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import torch
import random
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Fix random seed to reproduce same results
SEED = 2020


def set_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# threshold-based post-processing method
def get_label(logits, oos_idx, threshold):
    softmax_out = torch.softmax(logits, dim=1)
    max_args = torch.argmax(softmax_out, dim=1)

    if threshold > 0 and oos_idx > 0:
        max_p = torch.max(softmax_out, dim=1)[0]
        default_idx = torch.tensor([oos_idx] * logits.size()[0]).to(logits.device)
        return torch.where(max_p > threshold, max_args, default_idx)
    else:
        return max_args


# evaluation method
def evaluate(labels, preds, oos_idx):
    labels = np.array(labels)
    preds = np.array(preds)

    acc = (preds == labels).mean()
    oos_labels, oos_preds = [], []
    ins_labels, ins_preds = [], []
    for i in range(len(preds)):
        if labels[i] != oos_idx:
            ins_preds.append(preds[i])
            ins_labels.append(labels[i])

        oos_labels.append(int(labels[i] == oos_idx))
        oos_preds.append(int(preds[i] == oos_idx))

    ins_preds = np.array(ins_preds)
    ins_labels = np.array(ins_labels)
    ins_acc = (ins_preds == ins_labels).mean()

    # for oos samples
    oos_preds = np.array(oos_preds)
    oos_labels = np.array(oos_labels)

    TP = (oos_labels & oos_preds).sum()
    FP = ((oos_preds - oos_labels) > 0).sum()
    FN = ((oos_labels - oos_preds) > 0).sum()
    print('TP: {}, FP: {}, FN:{}'.format(TP, FP, FN))

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f_score = 2 * precision * recall / (precision + recall)
    results = {'acc': acc, 'ins_acc': ins_acc, 'oos_precision': precision, 'oos_recall': recall, 'f1': f_score}

    return results


# compute metrics using sklearn
def print_metrics(y_true_list, y_pred_list, vocab, prefix=None):
    accuary = accuracy_score(y_true_list, y_pred_list)
    print('Accuracy: {:.6f}'.format(accuary))

    print(confusion_matrix(y_true_list, y_pred_list))
    print(classification_report(y_true_list, y_pred_list, target_names=vocab.itos, digits=6))

    if 'oos' in vocab.stoi:
        results = evaluate(y_true_list, y_pred_list, vocab.stoi['oos'])
        if prefix:
            print(prefix, end=' ')
        for key, val in results.items():
            print('{}: {:.6f}'.format(key, val), end=' ')
        print('')


# utility methods
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as vocab_file:
        vocab = torch.load(vocab_file, encoding='ascii')
    return vocab
