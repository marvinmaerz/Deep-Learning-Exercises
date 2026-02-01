import torch as t
import numpy as np
import tabulate
from sklearn.metrics import f1_score
import threading
import time
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping. If validation loss does not decrease for this amount of epochs, stop training
        # self._model = model
        # self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._stop_training = False             # set via threaded listener for keyboard interrupts

        self._early_stopping_patience = early_stopping_patience

        self._device = t.device(
            t.accelerator.current_accelerator().type
            if cuda and t.cuda.is_available()
            else "cpu")
        print(f"Using {self._device} device")

        self._model = model.to(self._device)
        self._crit = crit.to(self._device)

        # if cuda:
        #     # If the current accelerator is available, we will use it. Otherwise, we use the CPU.
        #     device = t.accelerator.current_accelerator().type if t.accelerator.is_available() else "cpu"
        #     print(f"Using {device} device")
        #     self._model = model.cuda()
        #     self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))


    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])


    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})


    def _listen_for_break(self):
        while True:
            cmd = input()
            if cmd.strip().lower() == "break":
                self._stop_training = True
                break


    def train_step(self, x, y):
        # See: https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data
        # => Function loss_batch in the tutorial on train_dl with optimizer
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called.
        #   This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        loss = self._crit(self._model(x), y)
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()
        return loss.item(), len(x)
        
    
    def val_test_step(self, x, y):
        # See: https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data
        # => There, corresponds to loss_batch without optimizer on valid_dl
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        preds = self._model(x)
        loss = self._crit(preds, y)
        return loss.item(), len(x), preds


    def train_epoch(self):
        # See: https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data
        # => There, inside fit(), corresponds to the loop over the train_dl
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        self._model.train()
        total_loss = 0.0
        total_num = 0
        for xb, yb in self._train_dl:           # Suffix "b" to denote that x and y are batched according to the dataloader settings
            xb = xb.to(self._device)
            yb = yb.to(self._device)

            loss, n = self.train_step(xb, yb)
            total_loss += loss * n
            total_num += n

        train_loss = total_loss / total_num
        # print("Training loss:   ", train_loss)
        return train_loss

    
    def val_test(self):
        pass
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        self._model.eval()
        total_loss = 0.0
        total_num = 0
        predictions = []        # Saved predictions
        targets = []            # Saved ground truths
        with t.no_grad():
            for xb, yb in self._val_test_dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)

                loss, n, preds = self.val_test_step(xb, yb)

                total_loss += loss * n
                total_num += n

                # Convert predictions
                preds_cpu = preds.detach().cpu()
                yb_cpu = yb.detach().cpu()

                # Threshold, since BCE returns probabilities
                preds_labels = (preds_cpu > 0.5).int()

                predictions.append(preds_labels.numpy())
                targets.append(yb_cpu.numpy())

        val_loss = total_loss / total_num
        # TODO: add other metrics (F1 score, accuracy?)
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        f1 = f1_score(targets, predictions, average="macro")        # average="macro" due to multi-label learning
        accuracy = np.mean(predictions == targets)

        return val_loss, f1, accuracy
        
    
    def fit(self, epochs=-1):
        """
        Trains the model on the data and computes training and validation loss,
        while early stopping criterion is not fulfilled.
        Stops training if the validation loss didn't decrease for self._early_stopping_patience many
        epochs.
        :param epochs: Epochs to train the model for.
        :return: Training and validation losses for each epoch and the epoch on which training stopped.
            If early stopping was done, consider subtracting self._early_stopping_patience from the returned epoch
            counter when saving the model with save_checkpoint(epoch) for minimal validation loss.
        """

        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_loss = []
        val_loss = []
        epoch = 0
        best_val_loss = float('inf')
        best_epoch = 0
        remaining_patience = self._early_stopping_patience

        threading.Thread(target=self._listen_for_break, daemon=True).start()

        table = []
        
        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            if epochs != -1 and epoch >= epochs: break
            if self._stop_training:
                print("Manual training stop triggered.\nSaving current model.")
                self.save_checkpoint(epoch)
                break

            start = time.time()

            # Training and validating
            print("* Epoch ", epoch)
            train_loss.append(self.train_epoch())
            v_loss, f1, acc = self.val_test()
            val_loss.append(v_loss)

            # Saving best model
            if val_loss[epoch] < best_val_loss:
                best_val_loss = val_loss[epoch]
                best_epoch = epoch
                print("Saving best current model at epoch ", epoch)
                self.save_checkpoint(epoch)

            # Early stopping checks:
            val_loss_diff = 0.0
            tolerance = 0.0                     # hyperparameter? suggestion: set to either very small negative or positive value to allow more or less strict differences
            if epoch >= 1:
                val_loss_diff = val_loss[epoch] - val_loss[epoch - 1]
            if val_loss_diff > tolerance:       # validation loss didn't decrease, so count down patience
                if remaining_patience == 0:
                    print("Early stopping!")
                    break
                remaining_patience -= 1
            elif val_loss_diff <= tolerance:    # validation loss decreased, reset remaining patience
                remaining_patience = self._early_stopping_patience

            # Printing & collecting metrics:
            # print("Validation loss: ", val_loss[epoch], f" ({val_loss_diff})")
            print("Took {:.3f}".format(time.time() - start), " seconds.")
            table.append([epoch, val_loss[epoch], val_loss_diff, f1, acc])
            table.append([])

            epoch += 1

        print("Restoring best model at epoch ", best_epoch)
        self.restore_checkpoint(best_epoch)

        print(tabulate.tabulate(table, headers=["Epoch", "Validaton Loss", "Difference", "F1 Score", "Accuracy"], tablefmt="github"))

        return train_loss, val_loss, epoch
                    


        
        
