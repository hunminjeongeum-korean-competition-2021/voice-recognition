import argparse
import os
from glob import glob
import gc
import pickle

import pandas as pd
import numpy as np

import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader

import nsml
from nsml import HAS_DATASET, DATASET_PATH

from stt_model import Transformer
from data import CustomTokenizer, CustomDataset

print('torch version: ',torch.__version__)

def evaluate(model, imgs):
    model.to(device)
    # as the target is english, the first word to the transformer should be the
    # english start token.
    tokenizer = dict_for_infer['tokenizer']
    decoder_input = torch.tensor([tokenizer.txt2idx['<sos>']] * imgs.size(0), dtype=torch.long).to(device)
    output = decoder_input.unsqueeze(1).to(device)
    enc_output = None
    for i in range(max_length + 1):
        # predictions.shape == (batch_size, seq_len, vocab_size)
        with torch.no_grad():
            # predictions, attention_weights, enc_output = transformer([imgs, output, enc_output])
            predictions, attention_weights, enc_output = model([imgs, output, enc_output])
        # select the last token from the seq_len dimension
        predictions_ = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = torch.tensor(torch.argmax(predictions_, axis=-1), dtype=torch.int32)

        output = torch.cat([output, predicted_id], dim=-1)
    output = output.cpu().numpy()

    result_list = []
    token_list = []
    for token in output:
        summary = tokenizer.convert(token)
        result_list.append(summary)
        token_list.append(token)

    return result_list, token_list


def train_step(batch_item, training):
    src = batch_item['magnitude'].to(device)
    tar = batch_item['target'].to(device)
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    if training is True:
        # transformer.train()
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output, _, _ = model([src, tar_inp, None])
            # output, _, _ = transformer([src, tar_inp, None])
            loss = loss_function(tar_real, output)
        acc = accuracy_function(tar_real, output)
        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        return loss, acc, round(lr, 10)
    else:
        # transformer.eval()
        model.eval()
        with torch.no_grad():
            output, _, _ = model([src, tar_inp, None])
            # output, _, _ = transformer([src, tar_inp, None])
            loss = loss_function(tar_real, output)
        acc = accuracy_function(tar_real, output)
        return loss, acc


def loss_function(real, pred):
    mask = torch.logical_not(torch.eq(real, 0))
    loss_ = criterion(pred.permute(0, 2, 1), real)
    mask = torch.tensor(mask, dtype=loss_.dtype)
    loss_ = mask * loss_

    return torch.sum(loss_) / torch.sum(mask)


def accuracy_function(real, pred):
    accuracies = torch.eq(real, torch.argmax(pred, dim=2))
    mask = torch.logical_not(torch.eq(real, 0))
    accuracies = torch.logical_and(mask, accuracies)
    accuracies = torch.tensor(accuracies, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)

    return torch.sum(accuracies) / torch.sum(mask)

def path_loader(root_path, is_test= False):

    if is_test:
        file_list = sorted(glob(os.path.join(root_path, 'test_data', '*')))

        return file_list

    if args.mode == 'train' :
        train_path = os.path.join(root_path, 'train')
        file_list = sorted(glob(os.path.join(train_path, 'train_data', '*')))
        label = pd.read_csv(os.path.join(train_path, 'train_label'))

    return file_list, label

def save_checkpoint(checkpoint, dir):

    torch.save(checkpoint, os.path.join(dir))

def bind_model(model, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'checkpoint')
        save_checkpoint(dict_for_infer, save_dir)
        
        with open(os.path.join(dir_name, "dict_for_infer"), "wb") as f:
            pickle.dump(dict_for_infer, f)

        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):

        save_dir = os.path.join(dir_name, 'checkpoint')

        global checkpoint
        checkpoint = torch.load(save_dir)

        model.load_state_dict(checkpoint['model'])

        global dict_for_infer
        with open(os.path.join(dir_name, "dict_for_infer"), 'rb') as f:
            dict_for_infer = pickle.load(f)
        
        print("로딩 완료!")

    def infer(test_path, **kwparser):
        device = checkpoint['device']
        test_file_list = path_loader(test_path, is_test=True)
        test_dataset = CustomDataset(test_file_list, None, 160000, 'test')
        test_data_loader = DataLoader(test_dataset,
                                      batch_size=10)
        result_list = []
        
        for step, batch in enumerate(test_data_loader):
            inp = batch['magnitude'].to(device)
            output, _ = evaluate(model, inp)
            result_list.extend(output)

        prob = [1] * len(result_list)
        
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        # return list(zip(pred.flatten(), clipped.flatten()))
        return list(zip(prob, result_list))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    args = parser.parse_args()

    max_length = 30
    batch_size = 32
    num_layers = 6
    d_model = 512
    dff = 2048
    num_heads = 8
    dropout_rate = 0.1
    epochs = args.epochs
    learning_rate = 5e-5
    device = torch.device("cuda:0")
    max_vocab_size = 5000

    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        target_size=max_vocab_size + 4,
        pe_target=max_length + 1,
        device=device,
        rate=dropout_rate
    )
    
    bind_model(model=model, parser=args)
    if args.pause :
        nsml.paused(scope=locals())

    if args.mode == 'train' :
        file_list, label = path_loader(DATASET_PATH)

        split_num = int(len(label) * 0.9)
        train_file_list = file_list[:split_num]
        val_file_list = file_list[split_num:]

        train_label = label.iloc[:split_num]
        val_label = label.iloc[split_num:]

        tokenizer = CustomTokenizer(max_length=max_length, max_vocab_size=max_vocab_size)
        tokenizer.fit(train_label.text)

        target_size = len(tokenizer.txt2idx)

        train_tokens = tokenizer.txt2token(train_label.text)
        val_tokens = tokenizer.txt2token(val_label.text)
        train_dataset = CustomDataset(train_file_list, train_tokens)
        valid_dataset = CustomDataset(val_file_list, val_tokens)

        train_dataloader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)

        valid_dataloader = DataLoader(valid_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            gc.collect()
            total_train_loss, total_valid_loss = 0, 0
            total_train_acc, total_valid_acc = 0, 0

            training = True
            for batch in train_dataloader:
                batch = batch
                batch_loss, batch_acc, lr = train_step(batch, training)
                total_train_loss += batch_loss
                total_train_acc += batch_acc


            training = False
            for batch in valid_dataloader:
                batch = batch
                batch_loss, batch_acc = train_step(batch, training)
                total_valid_loss += batch_loss
                total_valid_acc += batch_acc


            print('=================loss=================')
            print(f'total_train_loss: {total_train_loss}')
            print(f'total_valid_loss: {total_valid_loss}')
            print('\n')

            print('=================acc=================')
            print(f'total_train_acc : {total_train_acc}')
            print(f'total_valid_acc : {total_valid_acc}')
            print('\n')


            dict_for_infer = {
                'model' : model.state_dict(),
                'max_length' : max_length,
                'target_size' : target_size,
                'num_layers' : num_layers,
                'd_model' : d_model,
                'dff' : dff,
                'num_heads': num_heads,
                'dropout_rate': dropout_rate,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'tokenizer' : tokenizer,
                'device' : device
            }

            
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)
