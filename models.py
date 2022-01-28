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
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel


class BertClassifier(BertPreTrainedModel):

    def __init__(self, config, num_labels, dropout=0.1):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)

        hidden_dim = config.hidden_size
        self.classifier = nn.Linear(hidden_dim, num_labels)

        self.init_weights()

    def forward(self, input_ids):
        outputs = self.bert(input_ids)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        output = self.classifier(pooled_output)

        return output


class BertDomainIntent(BertPreTrainedModel):

    def __init__(self, config, num_domains, num_intents, dropout=0.3,
                 subspace=False, hierarchy=False, domain_first=False):
        super().__init__(config)

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        hidden_dim = config.hidden_size

        self.linear1 = nn.Linear(config.hidden_size, hidden_dim)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(config.hidden_size, hidden_dim)
        self.relu2 = nn.ReLU()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.subspace = subspace
        self.hierarchy = hierarchy
        self.domain_first = domain_first

        self.domain_out = nn.Linear(hidden_dim, num_domains)
        self.intent_out = nn.Linear(hidden_dim, num_intents)

        self.init_weights()

    def forward(self, input_ids):
        outputs = self.bert(input_ids)

        pooled_output = torch.mean(outputs[0], dim=1)
        #pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # subspace layers
        if self.subspace:
            domain_rep = self.relu1(self.linear1(pooled_output))
            domain_rep = pooled_output + self.dropout1(domain_rep)
            if self.training:
                domain_rep = self.norm1(domain_rep)

            intent_rep = self.relu2(self.linear2(pooled_output))
            intent_rep = pooled_output + self.dropout2(intent_rep)
            if self.training:
                intent_rep = self.norm2(intent_rep)
        else:
            domain_rep = pooled_output
            intent_rep = pooled_output

        # hierarchical layers
        if self.hierarchy:
            if self.domain_first:
                domain_rep = self.relu1(self.linear1(pooled_output))
                domain_rep = pooled_output + self.dropout1(domain_rep)
                if self.training:
                    domain_rep = self.norm1(domain_rep)

                intent_rep = domain_rep + pooled_output
                intent_rep = self.relu2(self.linear2(intent_rep))

                intent_rep = domain_rep + self.dropout2(intent_rep)
                if self.training:
                    intent_rep = self.norm2(intent_rep)
            else:
                intent_rep = self.relu1(self.linear1(pooled_output))
                intent_rep = pooled_output + self.dropout1(intent_rep)
                if self.training:
                    intent_rep = self.norm1(intent_rep)

                domain_rep = intent_rep + pooled_output
                domain_rep = self.relu2(self.linear2(domain_rep))

                domain_rep = intent_rep + self.dropout2(domain_rep)
                if self.training:
                    domain_rep = self.norm2(domain_rep)

        # output layers
        domain_rep = self.dropout1(domain_rep)
        output1 = self.domain_out(domain_rep)

        intent_rep = self.dropout2(intent_rep)
        output2 = self.intent_out(intent_rep)

        return output1, output2


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, domain_loss, intent_loss):
        loss = self.w * domain_loss + (1-self.w) * intent_loss
        return loss


if __name__ == '__main__':
    from transformers import BertConfig

    config = BertConfig.from_pretrained('bert-base-uncased')
    model1 = BertClassifier(config, 10)
    model2 = BertDomainIntent(config, 10, 100)
    print(model1)
    print(model2)
