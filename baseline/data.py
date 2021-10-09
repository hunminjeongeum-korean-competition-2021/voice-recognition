from torch.utils.data import Dataset
import torch
import numpy as np
import librosa


class CustomTokenizer():
    def __init__(self, max_length, max_vocab_size=-1):
        self.txt2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2txt = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.max_length = max_length
        self.char_count = {}
        self.max_vocab_size = max_vocab_size

    def fit(self, sentence_list):
        for sentence in sentence_list:
            for char in sentence:
                try:
                    self.char_count[char] += 1
                except:
                    self.char_count[char] = 1
        self.char_count = dict(sorted(self.char_count.items(), key=self.sort_target, reverse=True))

        self.txt2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2txt = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        if self.max_vocab_size == -1:
            for i, char in enumerate(list(self.char_count.keys())):
                self.txt2idx[char] = i + 4
                self.idx2txt[i + 4] = char
        else:
            for i, char in enumerate(list(self.char_count.keys())[:self.max_vocab_size]):
                self.txt2idx[char] = i + 4
                self.idx2txt[i + 4] = char

    def sort_target(self, x):
        return x[1]

    def txt2token(self, sentence_list):
        tokens = []
        for j, sentence in enumerate(sentence_list):
            token = [0] * (self.max_length + 2)
            token[0] = self.txt2idx['<sos>']
            for i, c in enumerate(sentence):
                if i == self.max_length:
                    break
                try:
                    token[i + 1] = self.txt2idx[c]
                except:
                    token[i + 1] = self.txt2idx['<unk>']
            try:
                token[i + 2] = self.txt2idx['<eos>']
            except:
                pass
            tokens.append(token)
        return tokens

    def convert(self, token):
        sentence = []
        for i in token:
            if i == self.txt2idx['<eos>'] or i == self.txt2idx['<pad>']:
                break
            elif i != 0:
                try:
                    sentence.append(self.idx2txt[i])
                except:
                    sentence.append('<unk>')
        sentence = "".join(sentence)
        sentence = sentence[5:]

        return sentence


class CustomDataset(Dataset):
    def __init__(self, path_list, target_list, sound_max_length=160000, mode='train'):
        self.hop_length = 512
        self.n_fft = 512
        self.sr = 16000
        self.hop_length_duration = float(self.hop_length) / self.sr
        self.n_fft_duration = float(self.n_fft) / self.sr
        self.sound_max_length = sound_max_length

        self.mode = mode
        self.path_list = path_list

        if self.mode == 'train':
            self.target_list = target_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, i):
        data, rate = librosa.load(self.path_list[i])
        sound = np.zeros(self.sound_max_length)
        if len(data) <= self.sound_max_length:
            sound[:data.shape[0]] = data
        else:
            sound = data[:self.sound_max_length]
        stft = librosa.stft(sound, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft).astype(np.float32)
        magnitude_ = np.zeros([magnitude.shape[0], magnitude.shape[1], 3])
        magnitude_[:, :, 0] = magnitude
        magnitude_[:, :, 1] = magnitude
        magnitude_[:, :, 2] = magnitude

        magnitude = np.transpose(magnitude_, (2, 0, 1))

        if self.mode == 'train':
            target = self.target_list[i]
            return {
                'magnitude': torch.tensor(magnitude, dtype=torch.float32),
                'target': torch.tensor(target, dtype=torch.long)
            }
        else:
            return {
                'magnitude': torch.tensor(magnitude, dtype=torch.float32)
            }
