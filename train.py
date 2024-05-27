import os
import gc
import random
import json
import configparser
import argparse
import time

import numpy as np

import torch

from UT_model import UTModel
from Trainer import Trainer, TrainerConfig
from prepare_dataset import CharE, Tokenizer, CharDataset
from utils import * 

import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
logger = logging.getLogger(__name__)


#command line parser for config file
config = configparser.ConfigParser()
parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("-c","--config",dest="filename",help="Pass config file",metavar="FILE")
parser.add_argument('--context_size', help="Length of the sequence to process at once", type=int, required=False, default=512)
parser.add_argument('--num_generated_tokens', help="Number of tokens to generate at inference", type=int, required=False, default=512)
parser.add_argument("--act", help="Flag for whether to enable Adaptive Computation Time (ACT)", action="store_true", required=False)
parser.add_argument('--ponder_penalty', help="Set in range 0-1 only if act is enabled and want act ton contribute to the loss", required=False, type=float, default=0.0)    
parser.add_argument('--train_batch_size', help="Train batch size", required=False, type=int, default=32)
parser.add_argument('--eval_batch_size', help="Eval batch size", required=False, type=int, default=64)    
parser.add_argument('--epoch', help="Number of training epoch", required=False, type=int, default=100)
parser.add_argument('--learning_rate', help="Training Learning rate", required=False, type=float, default=3e-4)    
parser.add_argument('--seed', help="Set Randomised seed", required=False, type=int, default=42)    


args = parser.parse_args()
config.read(args.filename)

SEED = args.seed
seed_everything(SEED)

    #################################
    #       Model                   #
    #################################
# Difference between max context and problem length below is that: problem length just for the length of binary numbers in the operations,
# and max context counts for Binary number total length and =, */+ , SOR/EOR as our input format
max_context = args.context_size
embed_size = int(config['model_config']['embed_size'])
ff_size = int(config['model_config']['ff_size'])
num_layers = int(config['model_config']['num_layers'])
num_heads = int(config['model_config']['num_heads'])
act = args.act

    #################################
    #       Dataset                 #
    #################################

dataset_path = config['dataset']['dataset_path']

text = open(dataset_path, 'r').read()
n = len(text)
char_encoding = CharE(text[:int(n*0.9)])
char_encoding.form_token_map()

tokenizer = Tokenizer()
vocab_size = tokenizer.get_vocab_size()

train_data = text[:int(n*0.8)]
val_data = text[int(n*0.8):int(n*0.9)]

train_dataset = CharDataset(train_data, args.context_size, tokenizer)
eval_dataset = CharDataset(val_data, args.context_size, tokenizer)

    #################################
    #       Train Args              #
    #################################
max_epoch = args.epoch
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
num_workers = int(config['training_config']['num_workers'])
learning_rate = args.learning_rate
weight_decay = float(config['training_config']['weight_decay'])
beta_1 = float(config['training_config']['beta_1'])
beta_2 = float(config['training_config']['beta_2'])
ckpt_path = f"./checkpoints/{'UT' if args.act else 'vanilla'}"

num_generated_tokens = args.num_generated_tokens
generatation_save_dic_path = f"./generation/train"
save_file_name = f"UT_vs_Vanilla_generated_texts.json"

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(generatation_save_dic_path):
    os.makedirs(generatation_save_dic_path)

    #################################
    #       Training                #
    #################################

model = UTModel(vocab_size, max_context,
                embed_size, ff_size,
                num_layers, num_heads,
                act=act)

training_config = TrainerConfig(
        max_epoch = max_epoch,
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        num_workers = num_workers,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        AdamwBetas = (beta_1, beta_2),
        ckpt_path = ckpt_path,
        max_context = max_context,
        num_layers = num_layers,
        generatation_save_path = f"{generatation_save_dic_path}/{save_file_name}",
        act = act,
        seed=SEED,
        ponder_penalty = args.ponder_penalty,
        num_generated_tokens=num_generated_tokens,
        tokenizer = tokenizer)


if __name__ == "__main__":
    trainer = Trainer(model, train_dataset, eval_dataset, training_config)
    trainer.train()

    # Write args to file
    args_file_name = f'{generatation_save_dic_path}/train_args.txt'
    with open(args_file_name, 'w') as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
