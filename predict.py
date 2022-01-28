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


import torch

from argparse import ArgumentParser

from transformers import BertConfig

from dataset import get_iterator
from models import BertClassifier, BertDomainIntent
from utils import load_vocab, get_label, print_metrics, str2bool, set_seed


def run_single(args):
    intent_vocab = load_vocab(args.intent_vocab)
    num_intents = len(intent_vocab)

    config = BertConfig.from_pretrained(args.pretrained)
    model = BertClassifier(config=config, num_labels=num_intents)
    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)
    model.eval()

    y_true_intents, y_pred_intents = [], []

    intent_vocab = load_vocab(args.intent_vocab)

    if args.validation:
        data_loader = get_iterator(args.data_path + '/valid.csv', args.val_batch_size, args.device)
    else:
        data_loader = get_iterator(args.data_path + '/test.csv', args.test_batch_size, args.device)

    for batch in data_loader:
        x, y_true_intent = batch.text, batch.intent

        with torch.no_grad():
            intent_out = model(x)

        intent_oos_idx = intent_vocab.stoi['oos'] if 'oos' in intent_vocab.stoi else -1
        y_pred_intent = get_label(intent_out, intent_oos_idx, args.threshold)

        y_true_intents.extend(y_true_intent.tolist())
        y_pred_intents.extend(y_pred_intent.tolist())

    print_metrics(y_true_intents, y_pred_intents, intent_vocab, prefix='intent: ')


def run_joint(args):
    domain_vocab = load_vocab(args.domain_vocab)
    intent_vocab = load_vocab(args.intent_vocab)
    num_domains = len(domain_vocab)
    num_intents = len(intent_vocab)

    config = BertConfig.from_pretrained(args.pretrained)
    model = BertDomainIntent.from_pretrained(args.pretrained, config=config, num_domains=num_domains, num_intents=num_intents,
                                             subspace=args.subspace, hierarchy=args.hierarchy, domain_first=args.domain_first)

    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)
    model.eval()

    y_true_domains, y_true_intents, y_pred_domains, y_pred_intents = [], [], [], []

    if args.validation:
        data_loader = get_iterator(args.data_path + '/valid.csv', args.val_batch_size, args.device)
    else:
        data_loader = get_iterator(args.data_path + '/test.csv', args.test_batch_size, args.device)

    for batch in data_loader:
        x, y_true_domain, y_true_intent = batch.text, batch.domain, batch.intent
        with torch.no_grad():
            domain_out, intent_out = model(x)

        domain_oos_idx = domain_vocab.stoi['oos'] if 'oos' in domain_vocab.stoi else -1
        intent_oos_idx = intent_vocab.stoi['oos'] if 'oos' in intent_vocab.stoi else -1
        y_pred_domain = get_label(domain_out, domain_oos_idx, args.threshold)
        y_pred_intent = get_label(intent_out, intent_oos_idx, args.threshold)

        y_true_domains.extend(y_true_domain.tolist())
        y_pred_domains.extend(y_pred_domain.tolist())

        y_true_intents.extend(y_true_intent.tolist())
        y_pred_intents.extend(y_pred_intent.tolist())

    print_metrics(y_true_domains, y_pred_domains, domain_vocab, prefix='domain: ')
    print_metrics(y_true_intents, y_pred_intents, intent_vocab, prefix='intent: ')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--single", default=False, action='store_true', help="single or joint model")
    parser.add_argument("--validation", default=False, action='store_true', help="validation or testing")
    parser.add_argument("--data_path", type=str, default="dataset/oos-eval/", help="dataset path")
    parser.add_argument("--val_batch_size", type=int, default=32, help="input batch size for validation (default: 32)")
    parser.add_argument("--test_batch_size", type=int, default=32, help="input batch size for testing (default: 32)")
    parser.add_argument("--model_path", type=str, default="best_model.pt", help="model path")
    parser.add_argument("--domain_vocab", type=str, default="domain_vocab.pkl", help="path of domain vocabulary")
    parser.add_argument("--intent_vocab", type=str, default="intent_vocab.pkl", help="path of intent vocabulary")
    parser.add_argument("--pretrained", type=str, default="bert-base-uncased", help="model or path of pretrained model")
    parser.add_argument("--threshold", type=float, default=0.7, help="threshold for oos cutoff")
    parser.add_argument("--subspace", type=str, default='false', help="use subspace layers")
    parser.add_argument("--hierarchy", type=str, default='true', help="use hierarchical structure")
    parser.add_argument("--domain_first", type=str, default='false', help="domain representation first")

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.subspace = str2bool(args.subspace)
    args.hierarchy = str2bool(args.hierarchy)
    args.domain_first = str2bool(args.domain_first)

    set_seed()

    if args.single:
        run_single(args)
    else:
        run_joint(args)
