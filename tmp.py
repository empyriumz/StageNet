import numpy as np
import argparse
import os
import random

RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

from utils import utils
from utils import metrics
from model import StageNet

import optuna


def parse_arguments(parser):
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learing rate")


    args = parser.parse_args()
    return args


def define_model(trial):
    input_dim, output_dim = 1, 1
    hidden_dim = 384
    conv_size = trial.suggest_int("conv_size", 1, 15, 3)
    chunk_level = trial.suggest_int("chunk_level", 3, 6, 3)
    dropconnect_rate = 0.5
    dropout_rate = 0.5
    dropres_rate = 0.3

    model = StageNet(
        input_dim,
        hidden_dim,
        conv_size,
        output_dim,
        chunk_level,
        dropconnect_rate,
        dropout_rate,
        dropres_rate,
    )

    return model


def get_data(data_path, batch_size):
    train_data_loader = utils.DataLoader(
        dataset_dir=data_path,
        batch_size=batch_size,
        file_list="demo-list.csv",
        shuffle=True,
    )
    val_data_loader = utils.DataLoader(
        dataset_dir=data_path,
        batch_size=batch_size,
        file_list="demo-val-list.csv",
        shuffle=False,
    )

    return train_data_loader, val_data_loader


def objective(trial):
    model = define_model(trial).to(device)
    train_data_loader, val_data_loader = get_data("data/train", 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """Train phase"""
    print("Start training ... ")

    train_loss = []
    val_loss = []
    batch_loss = []
    # max_auprc = 0

    # file_name = "./saved_weights/" + args.file_name
    for epoch in range(args.epochs):
        cur_batch_loss = []
        model.train()
        for each_batch in range(train_data_loader.steps):
            batch_data = next(train_data_loader)
            batch_interval = batch_data["interval"]
            batch_mask = batch_data["mask"]
            batch_x, batch_y = batch_data["data"]
            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
            batch_interval = torch.tensor(batch_interval, dtype=torch.float32).to(
                device
            )
            batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
            # cut long sequence
            if batch_interval.size()[1] > 400:
                batch_x = batch_x[:, :400, :]
                batch_y = batch_y[:, :400, :]
                batch_interval = batch_interval[:, :400, :]
                batch_mask = batch_mask[:, :400, :]

            optimizer.zero_grad()
            output, _ = model(batch_x, batch_interval, device)
            output = output * batch_mask
            loss = batch_y * torch.log(output + 1e-7) + (1 - batch_y) * torch.log(
                1 - output + 1e-7
            )
            loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
            loss = torch.neg(torch.sum(loss))
            cur_batch_loss.append(loss.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

            if each_batch % 50 == 0:
                print(
                    "epoch %d, Batch %d: Loss = %.4f"
                    % (epoch, each_batch, cur_batch_loss[-1])
                )

        batch_loss.append(cur_batch_loss)
        train_loss.append(np.mean(np.array(cur_batch_loss)))

        print("\n==>Predicting on validation")
        with torch.no_grad():
            model.eval()
            cur_val_loss = []
            valid_true = []
            valid_pred = []
            for each_batch in range(val_data_loader.steps):
                valid_data = next(val_data_loader)
                valid_interval = valid_data["interval"]
                valid_mask = valid_data["mask"]
                valid_x, valid_y = valid_data["data"]
                valid_x = torch.tensor(valid_x, dtype=torch.float32).to(device)
                valid_y = torch.tensor(valid_y, dtype=torch.float32).to(device)
                valid_mask = torch.tensor(valid_mask, dtype=torch.float32).to(device)
                valid_interval = torch.tensor(valid_interval, dtype=torch.float32).to(
                    device
                )
                if valid_interval.size()[1] > 400:
                    valid_x = valid_x[:, :400, :]
                    valid_y = valid_y[:, :400, :]
                    valid_interval = valid_interval[:, :400, :]
                    valid_mask = valid_mask[:, :400, :]

                valid_output, _ = model(valid_x, valid_interval, device)
                valid_output = valid_mask * valid_output
                valid_loss = valid_y * torch.log(valid_output + 1e-7) + (
                    1 - valid_y
                ) * torch.log(1 - valid_output + 1e-7)
                valid_loss = torch.sum(valid_loss, dim=1) / torch.sum(valid_mask, dim=1)
                valid_loss = torch.neg(torch.sum(valid_loss))
                cur_val_loss.append(valid_loss.cpu().detach().numpy())

                for m, t, p in zip(
                    valid_mask.cpu().numpy().flatten(),
                    valid_y.cpu().numpy().flatten(),
                    valid_output.cpu().detach().numpy().flatten(),
                ):
                    if np.equal(m, 1):
                        valid_true.append(t)
                        valid_pred.append(p)

            val_loss.append(np.mean(np.array(cur_val_loss)))
            print("Valid loss = %.4f" % (val_loss[-1]))
            print("\n")
            valid_pred = np.array(valid_pred)
            valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
            ret = metrics.print_metrics_binary(valid_true, valid_pred)
            cur_auprc = ret["auprc"]

            # if cur_auprc > max_auprc:
            #     max_auprc = cur_auprc
            #     state = {
            #         "net": model.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #         "epoch": epoch,
            #     }
            #     torch.save(state, file_name)
            #     print("\n------------ Save best model ------------\n")

            trial.report(cur_auprc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return cur_auprc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    """ Prepare training data"""
    print("Preparing training data ... ")

    """Model structure"""
    print("Constructing model ... ")
    device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")
    print("available device: {}".format(device))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, timeout=1600)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
