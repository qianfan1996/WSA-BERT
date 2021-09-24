# -*-coding:utf-8-*- 
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss

from transformers import BertTokenizer, BertModel, XLNetTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from hyparams import HyParams as hp
from utils import set_random_seed
from load_data import load_pkl


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_words_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--valid_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--backbone_model", type=str, default="bert-base-uncased")
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=5862)

args = parser.parse_args()


def get_dataset(examples, max_words_length, tokenizer):
    words, acoustic, visual, label = examples

    dic = tokenizer(words, padding='max_length', truncation=True, max_length=max_words_length, return_tensors='pt')
    input_ids = dic['input_ids']
    input_mask = dic['attention_mask']
    segment_ids = dic['token_type_ids']
    label = torch.tensor([[label]], dtype=torch.float).transpose(0, 2)

    dataset = TensorDataset(
        input_ids,
        input_mask,
        segment_ids,
        label,
    )

    return dataset


def get_tokenizer(backbone_model):
    if backbone_model == "bert-base-uncased":
        return BertTokenizer.from_pretrained(backbone_model)
    elif backbone_model == "xlnet-base-cased":
        return XLNetTokenizer.from_pretrained(backbone_model)
    else:
        raise ValueError("Expected 'bert-base-uncased' or 'xlnet-base-cased, but received {}".format(backbone_model))


def set_up_data_loader():
    train_data = load_pkl(f"data/{args.dataset}_aligned.pkl", 'train')
    valid_data = load_pkl(f"data/{args.dataset}_aligned.pkl", 'valid')
    test_data = load_pkl(f"data/{args.dataset}_aligned.pkl", 'test')

    tokenizer = get_tokenizer(args.backbone_model)

    train_dataset = get_dataset(train_data, args.max_words_length, tokenizer)
    valid_dataset = get_dataset(valid_data, args.max_words_length, tokenizer)
    test_dataset = get_dataset(test_data, args.max_words_length, tokenizer)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    ) # all epoches number of training

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.valid_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


class BertClassifier(nn.Module):
    def __init__(self, num_labels=1, dropout=0.5, hidden_size=768):
        super(BertClassifier, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(args.backbone_model)

        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)


    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2: ]  # add hidden states and attention if they are here

        return outputs


def prep_for_training(num_train_optimization_steps):
    model = BertClassifier()

    model.to(hp.DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_optimization_steps,
        num_training_steps=args.warmup_proportion * num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model, train_dataloader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(hp.DEVICE) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        outputs = model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
        )

        logits = outputs[0]
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model, valid_dataloader):
    model.eval()
    valid_loss = 0
    nb_valid_examples, nb_valid_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(valid_dataloader, desc="Iteration")):
            batch = tuple(t.to(hp.DEVICE) for t in batch)

            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = model(
                input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )
            logits = outputs[0]

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            valid_loss += loss.item()
            nb_valid_steps += 1

    return valid_loss / nb_valid_steps


def test_epoch(model, test_dataloader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(hp.DEVICE) for t in batch)

            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = model(
                input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model, test_dataloader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)

    return acc, mae, corr, f_score


def train(
    model,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
):

    max_valid_loss = 999
    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, valid_dataloader)

        print("epoch:{}, train_loss:{}, valid_loss:{}".format(epoch_i, train_loss, valid_loss))

        if valid_loss < max_valid_loss:
            max_valid_loss = valid_loss
            print('Saving model ...')
            torch.save(model, 'saved_models/BERT_{}.pt'.format(args.dataset))


    model = torch.load('saved_models/BERT_{}.pt'.format(args.dataset))


    test_acc, test_mae, test_corr, test_f_score = test_score_model(model, test_dataloader)
    print("Accuracy: {}, MAE: {}, Corr: {}, F1_score: {}".format(test_acc, test_mae, test_corr, test_f_score))


def main():
    set_random_seed(args.seed)
    (
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        num_train_optimization_steps
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()