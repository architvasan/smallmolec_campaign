import smallmolec_campaign.training.optimizer as optimizer
from tqdm import tqdm
from torch.optim import Adam
import torch
import wandb
import os
class BERTTrainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        test_dataloader=None, 
        lr= 1e-6,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10000,
        log_freq=10,
        device='cuda',
        project_name='bert_smiles',
        run_name='bert_smiles',
        run_id=None,
        run_dir=None,
        run_notes=None, 
        ):

        self.device = device
        #print(self.device)
        self.model = model.to(device)
        #print(self.model)
        #self.model.to('cpu')
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = optimizer.ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps
            )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        
        mode = "train" if train else "test"

        avg_loss = 0
        for i, batch in tqdm(enumerate(data_loader)):
            #[key.to(self.device) for key in batch]
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(self.device)
            # 1. forward the next_sentence_prediction and masked_lm model
            mask_lm_output = self.model.forward(input_ids, attention_mask)
            #continue
            #import sys
            #sys.exit()
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), labels)
            #print(mask_loss)
            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                mask_loss.backward()
                self.optim_schedule.step_and_update_lr()
            # next sentence prediction accuracy
            avg_loss += mask_loss.item()
            print(f"{mask_loss=}")
            wandb.log({
                    f"{mode}loss": avg_loss/ (i + 1),
                    "step": i
                })


            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": mask_loss.item()
            }
            print(post_fix)
        print(len(data_loader))
        print(
            f"EP{epoch}, {mode}: \
            avg_loss={avg_loss / len(data_loader)}") 
