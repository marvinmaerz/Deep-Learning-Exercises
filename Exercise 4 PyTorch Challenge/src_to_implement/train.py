import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker
import numpy as np
import model
import pandas as pd
import os
import datetime
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # TRAINING HYPERPARAMETERS:
    batch_size = 150
    learning_rate = 0.001
    learning_rate_decay = 0.6
    learning_rate_decay_steps = 40
    momentum = 0.9
    use_cuda = True
    early_stopping_patience = 50
    early_stopping_threshold = 0.003     # criterion for best validation loss: val_loss[now] < best_val_loss - threshold, in patience window


    # Load the data from the csv file and perform a train-test-split
    # Locate the csv file in file system and read it:
    csv_path = ''
    for root, _, files in os.walk('.'):
        for name in files:
            if name == 'data.csv':
                csv_path = os.path.join(root, name)
    # Open & split dataframes:
    df = pd.read_csv(csv_path, sep=';')                # pandas dataframe
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)   # split df in two
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    # Create datasets:
    train_ds = ChallengeDataset(train_df, mode="train")
    val_ds = ChallengeDataset(val_df, mode="val")

    # Create dataloaders:
    train_dl = t.utils.data.DataLoader(train_ds, batch_size=batch_size, pin_memory=True)
    val_dl = t.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True)

    # Create instance of ResNet model:
    model = model.ResNet()

    # Define loss, optimization, and setup trainer:
    loss_func = t.nn.BCELoss()
    optimizer = t.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # TODO: try ADAM as optimizer
    lr_annealer = t.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay)
    # lr_annealer = None
    trainer = Trainer(model, loss_func, optimizer, lr_annealer, train_dl, val_dl, use_cuda, early_stopping_patience, early_stopping_threshold)

    # Train the model:
    # trainer.restore_checkpoint(316)
    train_loss, val_loss, epoch, best_epoch, f1_best, acc_best = trainer.fit()

    # Plot the results:
    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss')
    plt.plot(np.arange(len(val_loss)), val_loss, label='val loss')
    plt.title(#"Improvement on checkpoint_316\n" +
              f"learning rate = {learning_rate} (step-annealed), batch size = {batch_size} \n" +
              f"momentum = {momentum}\n" +
              f"best epoch = {best_epoch} : f1 score = {f1_best:.3f}, accuracy = {acc_best:.3f}", loc="left", pad=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.gca().yaxis.set_label_position("right")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'loss curves\\epochs={best_epoch}_lr={learning_rate}_bs={batch_size}_mo={momentum}.png')
    plt.show()