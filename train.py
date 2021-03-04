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
from utils.preprocessing import OneHotEncoder, Normalizer
from utils import metrics
from utils import common_utils
from model import StageNet


def parse_arguments(parser):
    parser.add_argument(
        "--data_path",
        type=str,
        metavar="<data_path>",
        default="./mortality_data",
        help="The path to the MIMIC-III data directory",
    )
    parser.add_argument(
        "--file_name", type=str, metavar="<data_path>", help="File name to save model"
    )
    parser.add_argument(
        "--small_part", type=int, default=0, help="Use part of training data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learing rate")

    parser.add_argument(
        "--input_dim", type=int, default=59, help="Dimension of visit record data"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=384, help="Dimension of hidden units in RNN"
    )
    parser.add_argument(
        "--output_dim", type=int, default=1, help="Dimension of prediction target"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--dropconnect_rate", type=float, default=0.5, help="Dropout rate in RNN"
    )
    parser.add_argument(
        "--dropres_rate",
        type=float,
        default=0.3,
        help="Dropout rate in residue connection",
    )
    parser.add_argument("--K", type=int, default=10, help="1D-conv filter size")   
    parser.add_argument(
        "--chunk_level", type=int, default=3, help="Value controlling the coarse grain level"
    )
    parser.add_argument("--div", type=int, default=1, help="divide the input time step to get output time step")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    """ Prepare training data"""
    print("Preparing training data ... ")
    train_data_loader = common_utils.MortalityDataLoader(
        dataset_dir=os.path.join(args.data_path, "train"),
        listfile=os.path.join(args.data_path, "train-mortality.csv"),
    )
    val_data_loader = common_utils.MortalityDataLoader(
        dataset_dir=os.path.join(args.data_path, "train"),
        listfile=os.path.join(args.data_path, "val-mortality.csv"),
    )
    encoder = OneHotEncoder(
        store_masks=False,
        impute_strategy="previous",
        start_time="zero",
    )

    encoder_header = encoder.create_header().split(",")
    # select non-categorical channels 
    cont_channels = [
        i for (i, x) in enumerate(encoder_header) if x.find("->") == -1
    ]
    cont_channels = cont_channels + list(range(59, 59+17))
    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = "utils/resources/mortality_normalizer.pkl"
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

    """Model structure"""
    print("Constructing model ... ")
    device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")
    print("available device: {}".format(device))

    if encoder._store_masks:
        input_dim = args.input_dim + 17
    else:
        input_dim = args.input_dim

    model = StageNet(
        input_dim,
        args.hidden_dim,
        args.K,
        args.output_dim,
        args.chunk_level,
        args.dropconnect_rate,
        args.dropout_rate,
        args.dropres_rate,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """Train phase"""
    print("Start training ... ")

    train_loss = []
    val_loss = []
    max_auroc = 0

    file_name = "./saved_weights/" + args.file_name
    for epoch in range(args.epochs):
        batch_loss = []
        model.train()
        for each_batch in range(train_data_gen.steps):
            batch_data = next(train_data_gen)
            batch_data = batch_data["data"]
            batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)
            batch_y = batch_data[1]
            batch_y = batch_y[:, 0, :]
            # cut long sequence
            if batch_x.size()[1] > 400:
                batch_x = batch_x[:, :400, :]
                batch_y = batch_y[:, :400, :]
                
            batch_x = batch_x[:, :, :-17]
            batch_interval = batch_x[:, :, -17:]
            
            output_step = batch_x.size()[1] // args.div
            optimizer.zero_grad()
            output, _ = model(batch_x, batch_interval, output_step, device)
            output = output.mean(axis=1)
            loss = batch_y * torch.log(output + 1e-7) + (1 - batch_y) * torch.log(
                1 - output + 1e-7
            )
            loss = torch.neg(torch.sum(loss)) / batch_x.size()[0]
            batch_loss.append(loss.cpu().detach().numpy())
            
            loss.backward()
            optimizer.step()
        
        epoch_loss = np.mean(np.array(batch_loss))
        print("Epoch: {}, Training loss = {:.6f}".format(epoch, epoch_loss))
        train_loss.append(epoch_loss)

        print("\n==>Predicting on validation")
        with torch.no_grad():
            model.eval()
            cur_val_loss = []
            valid_true = []
            valid_pred = []
            for each_batch in range(val_data_gen.steps):
                valid_data = next(val_data_gen)
                valid_data = valid_data["data"]
                valid_x = torch.tensor(valid_data[0][0], dtype=torch.float32).to(device)
                valid_y = torch.tensor(valid_data[1], dtype=torch.float32).to(device)
                tmp = torch.zeros(valid_x.size(0), 17, dtype=torch.float32).to(device)
                valid_interval = torch.zeros(
                    (valid_x.size(0), valid_x.size(1), 17), dtype=torch.float32
                ).to(device)

                for i in range(valid_x.size(1)):
                    cur_ind = valid_x[:, i, -17:]
                    tmp += (cur_ind == 0).float()
                    valid_interval[:, i, :] = cur_ind * tmp
                    tmp[cur_ind == 1] = 0

                if valid_x.size()[1] > 400:
                    valid_x = valid_x[:, :400, :]
                    valid_y = valid_y[:, :400, :]
                    valid_interval = valid_interval[:, :400, :]
                
                valid_y = valid_y[:, 0, :]
                output_step = valid_x.size()[1] // args.div
                valid_output, _ = model(valid_x, valid_interval, output_step, device)
                valid_output = valid_output.mean(axis=1)
                valid_loss = valid_y * torch.log(valid_output + 1e-7) + (
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

            val_loss = np.mean(np.array(cur_val_loss))
            print("Validation loss = {:.6f}".format(val_loss))
            print("\n")
            valid_pred = np.array(valid_pred)
            valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
            ret = metrics.print_metrics_binary(valid_true, valid_pred)
            cur_auroc = ret["auroc"]
            if cur_auroc > max_auroc:
                max_auroc = cur_auroc
                state = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state, file_name)
                print("\n------------ Save the best model ------------\n")
