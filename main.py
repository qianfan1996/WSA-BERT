# -*-coding:utf-8-*- 
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss

from transformers import BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from models import BertForSequenceClassification

from hyparams import HyParams as hp
from utils import set_random_seed
from load_data import load_pkl


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--model", type=str, choices=["WM-BERT", "MAG-BERT"],default="WM-BERT")
parser.add_argument("--max_words_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--valid_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--backbone_model", type=str, default="bert-base-uncased")
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=5862)

args = parser.parse_args()


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def get_dataset(examples, max_words_length, tokenizer):
    words, acoustic, visual, label = examples

    dic = tokenizer(words, padding='max_length', truncation=True, max_length=max_words_length, return_tensors='pt')
    input_ids = dic['input_ids']
    input_mask = dic['attention_mask']
    segment_ids = dic['token_type_ids']
    visual = torch.tensor(visual, dtype=torch.float)
    acoustic = torch.tensor(acoustic, dtype=torch.float)
    label = torch.tensor([[label]], dtype=torch.float).transpose(0, 2)

    dataset = TensorDataset(
        input_ids,
        visual,
        acoustic,
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


def prep_for_training(num_train_optimization_steps):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    if args.backbone_model == "bert-base-uncased":
        model = BertForSequenceClassification.from_pretrained(
            args.backbone_model, multimodal_config=multimodal_config, model=args.model, num_labels=1
        )

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
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        outputs = model(
            input_ids,
            visual,
            acoustic,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
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

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
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

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
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
            if args.model == "WM-BERT":
                torch.save(model, 'saved_models/WM-BERT_{}.pt'.format(args.dataset))
            else:
                torch.save(model, 'saved_models/MAG-BERT_{}.pt'.format(args.dataset))


    if args.model == "WM-BERT":
        model = torch.load('saved_models/WM-BERT_{}.pt'.format(args.dataset))
    else:
        model = torch.load('saved_models/MAG-BERT_{}.pt'.format(args.dataset))

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