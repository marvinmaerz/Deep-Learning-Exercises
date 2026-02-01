import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
import os
import datetime
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # HYPERPARAMETERS:
    batch_size = 128
    learning_rate = 0.2
    use_cuda = True
    early_stopping_patience = 5
    CPU_CORES = 0                   # increasing seems to halt training altogether


    # load the data from the csv file and perform a train-test-split
    # this can be accomplished using the already imported pandas and sklearn.model_selection modules
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

    # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
    train_dl = t.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=CPU_CORES, pin_memory=True)
    val_dl = t.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=CPU_CORES, pin_memory=True)

    # create an instance of our ResNet model
    model = model.ResNet()

    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    # set up the optimizer (see t.optim)
    # create an object of type Trainer and set its early stopping criterion
    loss_func = t.nn.BCELoss()
    optimizer = t.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    trainer = Trainer(model, loss_func, optimizer, train_dl, val_dl, use_cuda, early_stopping_patience)

    # go, go, go... call fit on trainer
    train_loss, val_loss, epoch = trainer.fit(1)

    # Optional: Export model as onnx
    now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    trainer.save_onnx(f"exported models\\model {now}.onnx")


    # plot the results
    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss')
    plt.plot(np.arange(len(val_loss)), val_loss, label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'loss curves\\losses_{epoch}epochs.png')
    plt.show()