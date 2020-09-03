import os
import logging
import time
import random
import math
import re
from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Process
from tqdm import tqdm
import torch.distributed as dist
from tokenizers import BertWordPieceTokenizer
from transformers import AlbertForMaskedLM, AlbertConfig

from config import Config
from reader import Reader


def distribute_data(batches, num_gpus):
    distributed_data = []
    if len(batches) % num_gpus == 0:
        batch_size = int(len(batches) / num_gpus)
        for idx in range(num_gpus):
            distributed_data.append(batches[batch_size*idx:batch_size*(idx+1)])
    else:
        batch_size = math.ceil(len(batches) / num_gpus)
        expanded_batches = deepcopy(batches) if type(batches) == list else batches.clone()
        while True:
            expanded_batches = expanded_batches + deepcopy(batches) if type(batches) == list else torch.cat([expanded_batches, batches.clone()], dim=0)
            if len(expanded_batches) >= batch_size*num_gpus:
                expanded_batches = expanded_batches[:batch_size*num_gpus]
                break
        for idx in range(num_gpus):
            distributed_data.append(expanded_batches[batch_size*idx:batch_size*(idx+1)])

    return distributed_data

def init_process(local_rank, backend, config, albert_config, logger):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.cuda.set_device(local_rank)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if local_rank != 0:
        logger.setLevel(logging.WARNING)
    
    if local_rank == 0:
        writer = SummaryWriter()
        if not os.path.exists("save"):
            os.mkdir("save")
        save_path = "save/model_{}.pt".format(re.sub("\s+", "_", time.asctime()))

    reader = Reader(config)
    start = time.time()
    logger.info("Loading data...")
    reader.load_data()
    end = time.time()
    logger.info("Loaded. {} secs".format(end-start))

    model = AlbertForMaskedLM(albert_config).cuda()
    optimizer = Adam(model.parameters(), lr=config.lr)

    if config.save_path is not None:
        load(model, optimizer, config.save_path, local_rank)

    train.global_step = 0
    train.max_iter = len(list(reader.make_batch("train")))
    validate.max_iter = len(list(reader.make_batch("dev")))

    min_loss = 1e+10
    early_stop_count = config.early_stop_count

    # logger.info("Validate...")
    # loss = validate(model, reader, config, local_rank)
    # logger.info("loss: {:.4f}".format(loss))

    for epoch in range(config.max_epochs):
        logger.info("Train...")
        start = time.time()

        if local_rank == 0:
            train_test(model, reader, optimizer, config, local_rank, writer)
        else:
            train_test(model, reader, optimizer, config, local_rank)
        
        exit(0)

        end = time.time()
        logger.info("epoch: {}, {:.4f} secs".format(epoch+1, end-start))

        logger.info("Validate...")
        loss = validate(model, reader, config, local_rank)
        logger.info("loss: {:.4f}".format(loss))
        
        if local_rank == 0:
            writer.add_scalar("Val/loss", loss, epoch+1)

        if loss < min_loss:  # save model
            if local_rank == 0:
                save(model, optimizer, save_path)
                logger.info("Saved to {}.".format(os.path.abspath(save_path)))
            
            min_loss = loss
            early_stop_count = config.early_stop_count
        else:  # ealry stopping
            if early_stop_count == 0:
                if epoch < config.min_epochs:
                    early_stop_count += 1
                    logger.info("Too early to stop training.")
                    logger.info("early stop count: {}".format(early_stop_count))
                else:
                    logger.info("Early stopped.")
                    break
            elif early_stop_count == 2:
                lr = lr / 2
                logger.info("learning rate schedule: {}".format(lr))
                for param in optimizer.param_groups:
                    param["lr"] = lr
            early_stop_count -= 1
            logger.info("early stop count: {}".format(early_stop_count))
    logger.info("Training finished.")

def train_test(model, reader, optimizer, config, local_rank, writer=None):
    for i in tqdm(range(100), total=100, ncols=150):
        model.zero_grad()
        inputs = torch.randint(0, 30000, size=(config.batch_size, 512), dtype=torch.int64).cuda()
        labels = torch.randint(0, 30000, size=(config.batch_size, 512), dtype=torch.int64).cuda()
        pad_mask = torch.ones(config.batch_size, 512, dtype=torch.bool).cuda()
        loss, logits = model(inputs, masked_lm_labels=labels, attention_mask=pad_mask)
        loss.backward()
        optimizer.step()
        del loss, logits
        torch.cuda.empty_cache()

def train(model, reader, optimizer, config, local_rank, writer=None):
    iterator = reader.make_batch("train")

    if local_rank == 0:  # only one process prints something
        t = tqdm(enumerate(iterator), total=train.max_iter, ncols=150, position=0, leave=True)
    else:
        t = enumerate(iterator)

    for batch_idx, batch in t:
        try:
            inputs, labels = reader.make_input(batch)
            batch_size = inputs.size(0)
            length = inputs.size(1)
            distributed_batch_size = math.ceil(batch_size / config.num_gpus)

            # distribute batches to each gpu
            inputs = distribute_data(inputs, config.num_gpus)[local_rank]
            labels = distribute_data(labels, config.num_gpus)[local_rank]

            model.zero_grad()
            pad_mask = (inputs != reader.pad_token_id)
            label_mask = (labels == reader.pad_token_id)
            label_mask.masked_fill_(label_mask, value=-100)  # use only masked tokens for loss
            loss, logits = model(inputs, masked_lm_labels=labels, attention_mask=pad_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            if local_rank == 0:
                writer.add_scalar("Train/loss", loss.item(), train.global_step)
                t.set_description("iter: {}, loss: {:.4f}".format(batch_idx+1, loss.item()))
                time.sleep(1)
            
            del loss, logits
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(e)
            print("batch size: {}, length: {}".format(batch_size, length))
            error_save_path = "save/model_error_{}.pt".format(re.sub("\s+", "_", time.asctime()))
            print("model saved to {}".format(error_save_path))
            save(model, optimizer, error_save_path)
            exit(0)

        except KeyboardInterrupt as e:
            print(e)
            stop_save_path = "save/model_stop_{}.pt".format(re.sub("\s+", "_", time.asctime()))
            print("model saved to {}".format(stop_save_path))
            save(model, optimizer, stop_save_path)
            exit(0)

def validate(model, reader, config, local_rank):
    model.eval()
    loss = 0
    batch_count = 0

    with torch.no_grad():
        iterator = reader.make_batch("dev")

        if local_rank == 0:
            t = tqdm(enumerate(iterator), total=validate.max_iter, ncols=150, position=0, leave=True)
        else:
            t = enumerate(iterator)

        for batch_idx, batch in t:
            inputs, labels = reader.make_input(batch)
            batch_size = inputs.size(0)
            length = inputs.size(1)
            distributed_batch_size = math.ceil(batch_size / config.num_gpus)

            # distribute batches to each gpu
            inputs = distribute_data(inputs, config.num_gpus)[local_rank].cuda().contiguous()
            labels = distribute_data(labels, config.num_gpus)[local_rank].cuda().contiguous()

            pad_mask = (inputs != reader.pad_token_id)
            loss_, logits = model.forward(inputs, masked_lm_labels=labels, attention_mask=pad_mask)

            loss += loss_ * distributed_batch_size
            batch_count += distributed_batch_size

            if local_rank == 0:
                t.set_description("iter: {}".format(batch_idx+1))
                time.sleep(1)

            del loss_, logits
            torch.cuda.empty_cache()

    val_loss = loss.item() / batch_count

    del loss
    torch.cuda.empty_cache()

    model.train()

    return val_loss

def save(model, optimizer, save_path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)

def load(model, optimizer, save_path, local_rank):
    checkpoint = torch.load(save_path, map_location = lambda storage, loc: storage.cuda(local_rank))
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

if __name__ == "__main__":
    os.environ["KMP_WARNINGS"] = "0"
    # torch.multiprocessing.set_start_method("spawn")

    config = Config()
    parser = config.parser
    config = parser.parse_args()

    config_dict = vars(config)
    albert_config = {
        "attention_probs_dropout_prob": 0,
        "bos_token_id": 2,
        "classifier_dropout_prob": 0.1,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0,
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 16384,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "albert",
        "num_attention_heads": 64,
        "num_hidden_groups": 1,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30000
    }
    for parameter in albert_config.keys():
        albert_config[parameter] = config_dict[parameter]
    albert_config = AlbertConfig.from_dict(albert_config)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    init_process(0, "gloo", config, albert_config, logger)