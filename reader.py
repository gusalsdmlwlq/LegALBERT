import os
import random
import numpy as np
from copy import deepcopy
import torch
from tokenizers import BertWordPieceTokenizer


class Reader:
    def __init__(self, config):
        self.tokenizer = BertWordPieceTokenizer(config.vocab_path, lowercase=False)
        self.train_data = []
        self.dev_data = []
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.max_position_embeddings = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.mask_token_id = config.mask_token_id
        self.masking_rate = config.masking_rate
        self.masking_unk_rate = config.masking_unk_rate
        self.masking_random_rate = config.masking_random_rate

    def load_data(self):
        train_data = open(os.path.join(self.data_path, "train_data.txt"), "r").read().split("\n")[:-1]
        for row in train_data:
            self.train_data.append(self.tokenizer.encode(row).ids)
        dev_data = open(os.path.join(self.data_path, "dev_data.txt"), "r").read().split("\n")[:-1]
        for row in dev_data:
            self.dev_data.append(self.tokenizer.encode(row).ids)

    def make_batch(self, mode="train"):
        if mode == "train":
            data = self.train_data
        else:
            data = self.dev_data
        all_batches = []
        batch = []
        for row in data:
            batch.append(row)
            if len(batch) == self.batch_size:
                all_batches.append(batch)
                batch = []
        if len(batch) > 0:
            all_batches.append(batch)
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch
                
    def make_input(self, batch):
        batch_size = len(batch)
        inputs = torch.zeros(batch_size, self.max_position_embeddings, dtype=torch.int64).cuda()
        labels = torch.zeros(batch_size, self.max_position_embeddings, dtype=torch.int64).cuda()
        max_length = 0
        for batch_idx in range(batch_size):
            tokens = batch[batch_idx]
            length = len(tokens)
            max_length = max(max_length, length)

            # masking
            masked_tokens = deepcopy(tokens)
            for token_idx in range(length):
                prob = np.random.random()
                if prob < self.masking_rate:
                    prob = prob / self.masking_rate
                    if prob < self.masking_unk_rate:  # change to [MASK]
                        masked_tokens[token_idx] = self.mask_token_id
                    elif prob < self.masking_unk_rate + self.masking_random_rate:  # change to random token
                        masked_tokens[token_idx] = np.random.randint(5, self.vocab_size)
                    labels[batch_idx] = tokens[token_idx]  # label original token id
            inputs[batch_idx, :length] = torch.tensor(masked_tokens)

        inputs = inputs[:, :max_length]
        labels = labels[:, :max_length]
        
        return inputs, labels