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
from utils.preprocessing import OneHotEncoder, Normalizer
from utils import common_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau

import optuna

EPOCHS = 20


def parse_arguments(parser):
    parser.add_argument(
        "--data_path",
        type=str,
        metavar="<data_path>",
        default="./mortality_data",
        help="The path to the MIMIC-III data directory",
    )
    parser.add_argument(
        "--small_part", "-s", type=bool, default=False, help="Use part of training data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )  
    args = parser.parse_args()
    return args


def objective(trial):
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    train_data_loader = common_utils.MortalityDataLoader(
        dataset_dir=os.path.join(args.data_path, "train"),
        listfile=os.path.join(args.data_path, "train-mortality.csv"),
        small_part=args.small_part,
    )

    #pos_weight = train_data_loader.pos_weight
    pos_weight = np.sqrt(train_data_loader.pos_weight)
    val_data_loader = common_utils.MortalityDataLoader(
        dataset_dir=os.path.join(args.data_path, "train"),
        listfile=os.path.join(args.data_path, "val-mortality.csv"),
        small_part=args.small_part,
    )

    encoder = OneHotEncoder(
        store_masks=False,
        impute_strategy="previous",
        start_time="zero",
    )

    encoder_header = encoder.create_header().split(",")
    # select non-categorical channels for normalization
    cont_channels = [i for (i, x) in enumerate(encoder_header) if x.find("->") == -1]
    # cont_channels = cont_channels + list(range(59,76))
    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = "utils/resources/mortality_normalizer_with_interval.pkl"
    normalizer.load_params(normalizer_state)

    train_data_gen = utils.BatchDataGenerator(
        train_data_loader,
        encoder,
        normalizer,
        args.batch_size,
        shuffle=True,
    )
    val_data_gen = utils.BatchDataGenerator(
        val_data_loader,
        encoder,
        normalizer,
        args.batch_size,
        shuffle=False,
    )
    output_dim = 1
    input_dim = 59
    conv_size = trial.suggest_int("conv_size", 9, 15)
    chunk_level =  trial.suggest_categorical("chunk_level", [2, 3, 4, 6, 8, 12])
    hidden_dim = trial.suggest_categorical("hidden_dim", [384, 576])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropconnect_rate = trial.suggest_categorical("dropconnect_rate", [0, 0.5])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0, 0.5])
    dropres_rate = trial.suggest_categorical("dropres_rate", [0, 0.5])
    weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-5, 1e-4, 1e-3])
    model = StageNet(
        input_dim,
        hidden_dim,
        conv_size,
        output_dim,
        chunk_level,
        dropconnect_rate,
        dropout_rate,
        dropres_rate,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.2)
    """Train phase"""
    print("Start training ... ")

    train_loss = []
    val_loss = []
    max_auroc = 0
    for epoch in range(EPOCHS):
        batch_loss = []
        model.train()
        for _ in range(train_data_gen.steps):
            batch_data = next(train_data_gen)
            batch_data = batch_data["data"]
            batch_x = batch_data[0]
            batch_y = batch_data[1]
            # separate input data and time interval
            # because the interval data will be feed into hidden states and
            # input gate as well
            batch_interval = batch_x[:, :, -17:]
            batch_x = batch_x[:, :, :-17]

            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
            batch_interval = torch.tensor(batch_interval, dtype=torch.float32).to(
                device
            )

            # cut long sequence
            if batch_x.size()[1] > 400:
                batch_x = batch_x[:, :400, :]
                batch_y = batch_y[:, :400, :]
                batch_interval = batch_interval[:, :400, :]

            output_step = 1
            optimizer.zero_grad()
            output, _ = model(batch_x, batch_interval, output_step, device)
            output = output.mean(axis=1)
            loss = pos_weight * batch_y * torch.log(output + 1e-7) + (
                1 - batch_y
            ) * torch.log(1 - output + 1e-7)
            loss = torch.neg(torch.sum(loss)) / batch_x.size()[0]
            batch_loss.append(loss.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(np.array(batch_loss))
        print("Epoch: {}, Training loss = {:.6f}".format(epoch, epoch_loss))
        train_loss.append(epoch_loss)

        print("==>Predicting on validation")
        with torch.no_grad():
            model.eval()
            cur_val_loss = []
            valid_true = []
            valid_pred = []
            for _ in range(val_data_gen.steps):
                valid_data = next(val_data_gen)
                valid_data = valid_data["data"]
                valid_x = valid_data[0]
                valid_y = valid_data[1]
                valid_interval = valid_x[:, :, -17:]
                valid_x = valid_x[:, :, :-17]

                valid_x = torch.tensor(valid_x, dtype=torch.float32).to(device)
                valid_y = torch.tensor(valid_y, dtype=torch.float32).to(device)
                valid_interval = torch.tensor(valid_interval, dtype=torch.float32).to(
                    device
                )

                if valid_x.size()[1] > 400:
                    valid_x = valid_x[:, :400, :]
                    valid_y = valid_y[:, :400, :]
                    valid_interval = valid_interval[:, :400, :]

                output_step = 1
                valid_output, _ = model(valid_x, valid_interval, output_step, device)
                valid_output = valid_output.mean(axis=1)
                valid_loss = pos_weight * valid_y * torch.log(valid_output + 1e-7) + (
                    1 - valid_y
                ) * torch.log(1 - valid_output + 1e-7)
                valid_loss = torch.neg(torch.sum(valid_loss)) / valid_x.size()[0]
                cur_val_loss.append(valid_loss.cpu().detach().numpy())

                for t, p in zip(
                    valid_y.cpu().numpy().flatten(),
                    valid_output.cpu().detach().numpy().flatten(),
                ):
                    valid_true.append(t)
                    valid_pred.append(p)
            cur_val_loss = np.mean(np.array(cur_val_loss))           
            scheduler.step(cur_val_loss)
            print("Validation loss = {:.6f}".format(cur_val_loss))
            val_loss.append(cur_val_loss)
            print("\n")
            valid_pred = np.array(valid_pred)
            valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
            ret = metrics.print_metrics_binary(valid_true, valid_pred, verbose=0)
            cur_auroc = ret["auroc"]
            
            if cur_auroc > max_auroc:
                max_auroc = cur_auroc
                print("ROC AUC={:.6f}".format(max_auroc))
                state = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "params": trial.params,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }
                file_name = "./saved_weights/model_trial_no_origin_h_and_mask_{}_{:.4f}".format(
                    trial.number, max_auroc
                )
                torch.save(state, file_name)
                print("  Params: ")
                for key, value in trial.params.items():
                    print(" {}: {}".format(key, value))
            
            trial.report(cur_auroc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return cur_auroc


if __name__ == "__main__":
    """ Prepare training data"""
    print("Preparing training data ... ")

    """Model structure"""
    print("Constructing model ... ")
    device = torch.device("cuda:2" if torch.cuda.is_available() == True else "cpu")
    print("available device: {}".format(device))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

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
        print(" {}: {}".format(key, value))