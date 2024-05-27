import os
import math
import json

import logging
from tqdm import tqdm 

import numpy as np
import pandas as pd 

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim.lr_scheduler as lr_scheduler 



import wandb

logger = logging.getLogger(__name__)



class TrainerConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class WarmupThenCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.cosine_annealing_steps = total_steps - warmup_steps
        super(WarmupThenCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            elapsed_steps = self.last_epoch - self.warmup_steps
            return [base_lr * (1 + math.cos(math.pi * elapsed_steps / self.cosine_annealing_steps)) / 2
                    for base_lr in self.base_lrs]

class Trainer:
    def __init__(self, model, trainset, evalset, train_config):
        self.model = model
        self.trainset = trainset
        self.evalset = evalset
        self.config = train_config

        wandb.init(project="UT-runs", name=f'Train_{"UT" if train_config.act else "Vanilla"}_model_\
                   {train_config.max_epoch}_epoch_{train_config.train_batch_size}_batch_{train_config.learning_rate:.0e}_LR\
                    {"_"+str(train_config.ponder_penalty)+"_ponder_penalty" if train_config.act else ""}_{train_config.seed}_seed')

        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)


    def save_checkpoints(self, ckpt_id):
        model = self.model
        ckpt_folder = self.config.ckpt_path
        torch.save(model.state_dict(), f"{ckpt_folder}/{ckpt_id}.pth")
    
    def generate_text(self, model, num_tokens):
        idx = torch.zeros((1,1), dtype=torch.long).to(self.device)
        if self.config.act:
            token_ids, ponder_time, token_ponder_dict = model.generate(idx, num_tokens)
            decoded_token_ponder_dict = {generate_index: (self.config.tokenizer.decode(token_id), ponder_time) 
            for generate_index, (token_id, ponder_time) in token_ponder_dict.items()}
        else:
            token_ids = model.generate(idx, num_tokens)
        text = self.config.tokenizer.decode(token_ids.squeeze())
        return (text, ponder_time, decoded_token_ponder_dict) if self.config.act else text

    def train(self):
        config = self.config
        model = self.model
        optimizer = model.UT_optimizer(config)

        lr_steps = int(len(self.trainset) / config.train_batch_size * config.max_epoch)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, lr_steps)
        scheduler = WarmupThenCosineAnnealingLR(optimizer, total_steps=lr_steps, warmup_steps=int(0.1*lr_steps))
        
        def train_loop(train_dataloader, epoch_idx=1):
            model.train()

            for itr, (x,y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Train'):
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                if config.act:
                    _, loss, (_, n_updates)= model(x, y)
                    if float(config.ponder_penalty) != 0.0:
                        ponder_cost = n_updates.mean() * config.ponder_penalty
                        loss = loss + ponder_cost
                else:
                    _, loss = model(x, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_metrics = {"train/train_loss": loss, "train/train_lr": scheduler.get_last_lr()[0]}
                if config.act:
                    train_metrics.update({"train/train_ponder": n_updates.mean().item()})
                wandb.log(train_metrics)

                if itr%1000 == 0:
                    if config.act:
                        generated_text, ponder_time, token_ponder_dict = self.generate_text(model, num_tokens=config.num_generated_tokens)
                    else:
                        generated_text = self.generate_text(model, num_tokens=config.num_generated_tokens)
                    
                    state_generated_text = {"epoch":epoch_idx,
                                            "model": "UT" if config.act else "Vanilla",
                                            "ponder_penalty" : config.ponder_penalty if config.act else 0,
                                            "ponder_time" : ponder_time if config.act else config.num_layers,
                                            "generated_text": generated_text,
                                            "token_ponder_dict": token_ponder_dict if config.act else {"all_tokens":config.num_layers},
                                            "train_itr": itr}
                    try:
                        if os.path.exists(config.generatation_save_path):
                            with open(config.generatation_save_path, 'r') as file:
                                data = json.load(file)
                                data.append(state_generated_text)
                        else:
                            data = [state_generated_text]

                        with open(config.generatation_save_path, 'w') as file:
                            json.dump(data, file, indent=4) 
                    except IOError as e:
                        print(f"Error writing to JSON file: {e}")

        def eval_loop(eval_dataloader):
            model.eval()
            losses = []
            for _, (x, y) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc='Eval'):
                x = x.to(self.device)
                y = y.to(self.device)

                if config.act:
                    _, loss, (_, n_updates)= model(x, y)
                else:
                    _, loss = model(x, y)

                losses.append(loss.item())
                val_metrics = {"val/val_loss": loss}
                if config.act:
                    val_metrics.update({"val/val_ponder": n_updates.mean().item()})
                wandb.log(val_metrics)

            return float(np.mean(losses))
        

        train_dataloader = DataLoader(
            self.trainset,
            batch_size = config.train_batch_size,
            num_workers = config.num_workers,
            drop_last = True,
        )

        eval_dataloader = DataLoader(
            self.evalset,
            batch_size =  config.eval_batch_size,
            num_workers = config.num_workers,
            drop_last= True
        )

        best_loss = float('inf')
        for epoch in range(config.max_epoch):
            logger.info(f"===============Epoch:{epoch+1}/{config.max_epoch}=============")
            epoch_idx = (epoch+1)
            train_loop(train_dataloader, epoch_idx=epoch_idx)
            eval_loss = eval_loop(eval_dataloader)
            goodModel = eval_loss < best_loss
            if config.ckpt_path is not None and goodModel:
                best_loss = eval_loss
                self.save_checkpoints(f"{config.max_epoch}epoch_best_model_{config.train_batch_size}batch{'_'+str(config.ponder_penalty)+'ponder_penalty' if config.act else ''}_{config.learning_rate:.0e}LR_{config.seed}Seed")

        self.save_checkpoints(f"{config.max_epoch}epoch_last_model_{config.train_batch_size}batch{'_'+str(config.ponder_penalty)+'ponder_penalty' if config.act else ''}_{config.learning_rate:.0e}LR_{config.seed}Seed")
        wandb.finish()