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

from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        "--small_part", "-s", type=bool, default=False, help="Use part of training data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.0015, help="Learing rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Value controlling the coarse grain level"
    )
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
    parser.add_argument("--conv_size", type=int, default=13, help="1D-conv filter size")   
    parser.add_argument(
        "--chunk_level", type=int, default=6, help="Value controlling the coarse grain level"
    )
    parser.add_argument(
        "--store_masks", type=bool, default=False, help="including mask as input"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    """ Prepare training data"""
    print("Preparing training data ... ")
    train_data_loader = common_utils.MortalityDataLoader(
        dataset_dir=os.path.join(args.data_path, "train"),
        listfile=os.path.join(args.data_path, "train-mortality.csv"),
        small_part=args.small_part,
    )
    pos_weight = torch.sqrt(torch.tensor(train_data_loader.pos_weight, dtype=torch.float32))
    
    val_data_loader = common_utils.MortalityDataLoader(
        dataset_dir=os.path.join(args.data_path, "train"),
        listfile=os.path.join(args.data_path, "val-mortality.csv"),
        small_part=args.small_part,
    )
    encoder = OneHotEncoder(
        store_masks=args.store_masks,
        impute_strategy="previous",
        start_time="zero",
    )

    encoder_header = encoder.create_header().split(",")
    # select non-categorical channels 
    cont_channels = [
        i for (i, x) in enumerate(encoder_header) if x.find("->") == -1
    ]
    #cont_channels = cont_channels + [59]
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

    """Model structure"""
    print("Constructing model ... ")
    device = torch.device("cuda:1" if torch.cuda.is_available() == True else "cpu")
    print("available device: {}".format(device))

    if encoder._store_masks:
        input_dim = args.input_dim + 17
    else:
        input_dim = args.input_dim
    model_params = vars(args)  
    model_params["input_dim"] = input_dim
    model = StageNet(
        model_params["input_dim"],
        model_params["hidden_dim"],
        model_params["conv_size"],
        model_params["output_dim"],
        model_params["chunk_level"],
        model_params["dropconnect_rate"],
        model_params["dropout_rate"],
        model_params["dropres_rate"],
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
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
            batch_x = batch_data[0]
            batch_y = batch_data[1]
            # separate input data and time interval
            # because the interval data will be feed into hidden states and 
            # input gate as well
            batch_interval = batch_x[:, :, -17:]  
            batch_x = batch_x[:, :, :-17]
                 
            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
            batch_interval = torch.tensor(batch_interval, dtype=torch.float32).to(device)
            
            # cut long sequence
            if batch_x.size()[1] > 400:
                batch_x = batch_x[:, :400, :]
                batch_y = batch_y[:, :400, :]
                batch_interval = batch_interval[:, :400, :]
                      
            output_step = 1
            optimizer.zero_grad()
            output, _ = model(batch_x, batch_interval, output_step, device)
            output = output.mean(axis=1)
            loss = pos_weight * batch_y * torch.log(output + 1e-7) + (1 - batch_y) * torch.log(
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
                valid_x = valid_data[0]
                valid_y = valid_data[1]
                valid_interval = valid_x[:, :, -17:]  
                valid_x = valid_x[:, :, :-17]
                    
                valid_x = torch.tensor(valid_x, dtype=torch.float32).to(device)
                valid_y = torch.tensor(valid_y, dtype=torch.float32).to(device)
                valid_interval = torch.tensor(valid_interval, dtype=torch.float32).to(device)
                
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
            ret = metrics.print_metrics_binary(valid_true, valid_pred)
            cur_auroc = ret["auroc"]
            if cur_auroc > max_auroc:
                max_auroc = cur_auroc
                state = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "params": model_params,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }
                torch.save(state, file_name)
                print("\n------------ Save the best model ------------\n")
    end_time = time.time()
    print("total used time = {}".format(end_time - start_time))
