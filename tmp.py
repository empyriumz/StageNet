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
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils
from model import StageNet


def parse_arguments(parser):
    parser.add_argument(
        "--data_path",
        type=str,
        metavar="<data_path>",
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
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
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
    parser.add_argument("--K", type=int, default=10, help="Value of hyper-parameter K")
    parser.add_argument(
        "--chunk_level", type=int, default=3, help="Value of hyper-parameter K"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    """ Prepare training data"""
    print("Preparing training data ... ")
    train_data_loader = common_utils.DeepSupervisionDataLoader(
        dataset_dir=os.path.join(args.data_path, "train"),
        listfile=os.path.join(args.data_path, "demo-list.csv"),
        small_part=args.small_part,
    )
    val_data_loader = common_utils.DeepSupervisionDataLoader(
        dataset_dir=os.path.join(args.data_path, "train"),
        listfile=os.path.join(args.data_path, "demo-val-list.csv"),
        small_part=args.small_part,
    )
    discretizer = Discretizer(
        timestep=1.0,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
    )

    discretizer_header = discretizer.create_header().split(",")
    cont_channels = [
        i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
    ]

    normalizer = Normalizer(fields=cont_channels)
    # where does this comes from?
    normalizer_state = "decomp_normalizer"
    normalizer_state = os.path.join(args.data_path, normalizer_state)
    normalizer.load_params(normalizer_state)

    train_data_gen = utils.BatchGenDeepSupervision(
        train_data_loader,
        discretizer,
        normalizer,
        args.batch_size,
        shuffle=True,
        return_names=True,
    )
    val_data_gen = utils.BatchGenDeepSupervision(
        val_data_loader,
        discretizer,
        normalizer,
        args.batch_size,
        shuffle=False,
        return_names=True,
    )

    """Model structure"""
    print("Constructing model ... ")
    device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")
    print("available device: {}".format(device))
    
    if discretizer._store_masks:
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
    batch_loss = []
    max_auprc = 0

    file_name = "./saved_weights/" + args.file_name
    for epoch in range(args.epochs):
        cur_batch_loss = []
        model.train()
        for each_batch in range(train_data_gen.steps):
            batch_data = next(train_data_gen)
            batch_data = batch_data["data"]

            batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
            batch_mask = (
                torch.tensor(batch_data[0][1], dtype=torch.float32)
                .unsqueeze(-1)
                .to(device)
            )
            batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)
            tmp = torch.zeros(batch_x.size(0), 17, dtype=torch.float32).to(device)
            batch_interval = torch.zeros(
                (batch_x.size(0), batch_x.size(1), 17), dtype=torch.float32
            ).to(device)

            for i in range(batch_x.size(1)):
                # go over time direction
                # cur_ind represents the mask part in the data
                cur_ind = batch_x[:, i, -17:]
                # identify the empty data spot and accumulate
                tmp += (cur_ind == 0).float()
                # keeps track of the the interval from last non-zero data
                batch_interval[:, i, :] = cur_ind * tmp
                # so those with non-zero data at the moment, set the timer to zero
                tmp[cur_ind == 1] = 0
            
            # cut long sequence
            if batch_mask.size()[1] > 400:
                batch_x = batch_x[:, :400, :]
                batch_mask = batch_mask[:, :400, :]
                batch_y = batch_y[:, :400, :]
                batch_interval = batch_interval[:, :400, :]

            optimizer.zero_grad()
            cur_output, _ = model(batch_x, batch_interval, device)
            masked_output = cur_output * batch_mask
            loss = batch_y * torch.log(masked_output + 1e-7) + (
                1 - batch_y
            ) * torch.log(1 - masked_output + 1e-7)
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
            for each_batch in range(val_data_gen.steps):
                valid_data = next(val_data_gen)
                valid_data = valid_data["data"]
                valid_x = torch.tensor(valid_data[0][0], dtype=torch.float32).to(device)
                valid_mask = (
                    torch.tensor(valid_data[0][1], dtype=torch.float32)
                    .unsqueeze(-1)
                    .to(device)
                )
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

                if valid_mask.size()[1] > 400:
                    valid_x = valid_x[:, :400, :]
                    valid_mask = valid_mask[:, :400, :]
                    valid_y = valid_y[:, :400, :]
                    valid_interval = valid_interval[:, :400, :]

                valid_output, valid_dis = model(valid_x, valid_interval, device)
                masked_valid_output = valid_output * valid_mask

                valid_loss = valid_y * torch.log(masked_valid_output + 1e-7) + (
                    1 - valid_y
                ) * torch.log(1 - masked_valid_output + 1e-7)
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
            if cur_auprc > max_auprc:
                max_auprc = cur_auprc
                state = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state, file_name)
                print("\n------------ Save best model ------------\n")
