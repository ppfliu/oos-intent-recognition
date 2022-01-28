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

from argparse import ArgumentParser

import torch
from torch import nn

from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.engine import Engine, Events

from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import EarlyStopping, Checkpoint, DiskSaver, global_step_from_engine

from transformers import AdamW, BertConfig

from dataset import get_iterator
from models import BertClassifier
from utils import set_seed


def train(args):
    set_seed()

    train_loader, num_domains, num_intents = get_iterator(args.data_path + '/train.csv', args.train_batch_size, args.device, train=True)
    val_loader = get_iterator(args.data_path + '/valid.csv', args.val_batch_size, args.device)

    config = BertConfig.from_pretrained(args.pretrained)
    model = BertClassifier.from_pretrained(args.pretrained, config=config, num_labels=num_intents, dropout=args.dropout)
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    criterion = nn.CrossEntropyLoss()

    def update(engine, batch):
        model.train()

        x_text, y_domain, y_intent = batch.text, batch.domain, batch.intent
        y_pred_intent = model(x_text)
        loss = criterion(y_pred_intent, y_intent)

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()


    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x_text, y_domain, y_intent = batch.text, batch.domain, batch.intent
            y_pred_intent = model(x_text)
            return {'intent': (y_pred_intent, y_intent)}


    trainer = Engine(update)
    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    train_evaluator = Engine(inference)
    valid_evaluator = Engine(inference)

    def attach_metrics(evaluator):
        Accuracy(output_transform=lambda out: out['intent']).attach(evaluator, 'intent_acc')
        Loss(criterion, output_transform=lambda out: out['intent']).attach(evaluator, 'intent_loss')

    attach_metrics(train_evaluator)
    attach_metrics(valid_evaluator)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    def log_message(msg_type, epoch, metrics):
        pbar.log_message(
            "{} - Epoch: {}  intent acc/loss: {:.4f}/{:.4f}".format(
                msg_type, epoch, metrics['intent_acc'], metrics['intent_loss']
            )
        )

    #@trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        log_message('Training', engine.state.epoch, metrics)

    #@trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        valid_evaluator.run(val_loader)
        metrics = valid_evaluator.state.metrics
        log_message('Validation', engine.state.epoch, metrics)

        pbar.n = pbar.last_print_n = 0

    def score_function(engine):
        #return (engine.state.metrics['domain_acc'] + engine.state.metrics['intent_acc']) * 0.5
        return engine.state.metrics['intent_acc']

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

    to_save = {'model': model}
    checker = Checkpoint(to_save, DiskSaver('checkpoints', create_dir=True, require_empty=False),
                         n_saved=args.n_saved, filename_prefix='best', score_function=score_function, score_name='intent_acc',
                         global_step_transform=global_step_from_engine(trainer))
    stopper = EarlyStopping(patience=args.patience, score_function=score_function, trainer=trainer)

    valid_evaluator.add_event_handler(Events.COMPLETED, checker)
    valid_evaluator.add_event_handler(Events.COMPLETED, stopper)

    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/oos-eval/", help="dataset path")
    parser.add_argument("--train_batch_size", type=int, default=100, help="input batch size for training (default: 100)")
    parser.add_argument("--val_batch_size", type=int, default=100, help="input batch size for validation (default: 100)")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument("--patience", type=int, default=5, help="number of more epochs before early stopping")
    parser.add_argument("--n_saved", type=int, default=1, help="number of models saved")
    parser.add_argument("--lr", type=float, default=4.00E-05, help="learning rate (default: 0.01)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for model")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--pretrained", type=str, default="bert-base-uncased", help="model/path of pretrained model")

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(args)
