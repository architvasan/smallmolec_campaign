'''test run'''
import pandas as pd
from smallmolec_campaign.training import smiles_dataloader, trainer, model
import wandb
def run_train(
        data_train,
        data_val,
        vocab_file='../../VocabFiles/vocab_spe.txt',
        spe_file = '../../VocabFiles/SPE_ChEMBL.txt',
        max_len=64,
        batch_size=256,
        mask_prob=0.15,
        epochs=20,
        d_model=256,
        n_layers=21,
        heads=32,
        dropout=0.1,
        device='cuda',
        ):
        wandb.init(project='sst runs', name='bert_smiles')
        # Initialize and train the model
        tokenizer = smiles_dataloader.smilespetok(vocab_file=vocab_file, spe_file=spe_file)
        input_data_train = list(pd.read_csv(data_train)['smiles'])
        input_data_val = list(pd.read_csv(data_val)['smiles'])
        # Create dataloaders for training and validation 
        train_dataloader = smiles_dataloader.MLMDataloader(
                                               texts=input_data_train,
                                               tokenizer=tokenizer, 
                                               batch_size=batch_size,
                                               max_length=max_len,
                                               mask_prob=mask_prob).get_dataloader()

        val_dataloader = smiles_dataloader.MLMDataloader(
                                               texts=input_data_train,
                                               tokenizer=tokenizer, 
                                               batch_size=batch_size,
                                               max_length=max_len,
                                               mask_prob=mask_prob).get_dataloader()

        bert_model = model.BERT(
                       vocab_size=len(tokenizer.vocab),
                       d_model=d_model,
                       n_layers=n_layers,
                       heads=heads,
                       dropout=dropout
                       )

        bert_lm = model.BERTLM(bert_model, len(tokenizer.vocab))
        bert_trainer = trainer.BERTTrainer(bert_lm, train_dataloader, val_dataloader, device='cuda')
        epochs = 20

        for epoch in range(epochs):
          bert_trainer.train(epoch)
          bert_trainer.test(epoch)

if __name__ == '__main__':
        import argparse

        parser = argparse.ArgumentParser(description='Train a BERT model for SMILES data.')

        parser.add_argument('--data_train', type=str, required=True, help='Path to the training data CSV file.')
        parser.add_argument('--data_val', type=str, required=True, help='Path to the validation data CSV file.')
        parser.add_argument('--vocab_file', type=str, default='../../VocabFiles/vocab_spe.txt', help='Path to the vocabulary file.')
        parser.add_argument('--spe_file', type=str, default='../../VocabFiles/SPE_ChEMBL.txt', help='Path to the SPE file.')
        parser.add_argument('--max_len', type=int, default=64, help='Maximum length of the sequences.')
        parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
        parser.add_argument('--mask_prob', type=float, default=0.15, help='Probability of masking tokens.')
        parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
        parser.add_argument('--d_model', type=int, default=256, help='Dimension of the model.')
        parser.add_argument('--n_layers', type=int, default=21, help='Number of layers in the model.')
        parser.add_argument('--heads', type=int, default=32, help='Number of attention heads.')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu").')
        args = parser.parse_args()
        # Call the run_train function with the provided arguments
        run_train(
                data_train=args.data_train,
                data_val=args.data_val,
                vocab_file=args.vocab_file,
                spe_file = args.spe_file,
                max_len=args.max_len,
                batch_size=args.batch_size,
                mask_prob=args.mask_prob,
                epochs=args.epochs,
                d_model=args.d_model,
                n_layers=args.n_layers,
                heads=args.heads,
                dropout=args.dropout,
                device=args.device,
                )