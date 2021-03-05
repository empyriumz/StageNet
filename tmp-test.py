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
        default="./mortality_data",
        metavar="<data_path>",
        help="The path to the MIMIC-III data directory",
    )
    parser.add_argument(
        "--file_name", type=str, metavar="<data_path>", help="File name to load the trainded model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
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
    parser.add_argument("--K", type=int, default=10, help="1D-conv filter size")   
    parser.add_argument(
        "--chunk_level", type=int, default=3, help="Value controlling the coarse grain level"
    )
    parser.add_argument("--div", type=int, default=2, help="divide the input time step to get output time step")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    """ Prepare test data"""
    test_data_loader = common_utils.MortalityDataLoader(
        dataset_dir=os.path.join(args.data_path, "test"),
        listfile=os.path.join(args.data_path, "test-mortality.csv"),
    )
    encoder = OneHotEncoder(
        store_masks=False,
        impute_strategy="previous",
        start_time="zero",
    )

    encoder_header = encoder.create_header().split(",")
    cont_channels = [
        i for (i, x) in enumerate(encoder_header) if x.find("->") == -1
    ]

    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = "utils/resources/mortality_normalizer_with_interval.pkl"
    normalizer.load_params(normalizer_state)

    test_data_gen = utils.BatchDataGenerator(
        test_data_loader,
        encoder,
        normalizer,
        args.batch_size,
        shuffle=False,
    )

    """Model structure"""
    print("Loading model ... ")
    device = torch.device("cuda:2" if torch.cuda.is_available() == True else "cpu")
    print("available device: {}".format(device))

    if encoder._store_masks:
        input_dim = args.input_dim + 17
    else:
        input_dim = args.input_dim

    file_name = 'saved_weights/model_1d_trial_1_0.8455'
    checkpoint = torch.load(file_name) 
    model = StageNet(
        input_dim,
        args.hidden_dim,
        checkpoint['params']['conv_size'],
        args.output_dim,
        checkpoint['params']['chunk_level'],
        args.dropconnect_rate,
        args.dropout_rate,
        args.dropres_rate,
    ).to(device)
    saved_epoch = checkpoint['epoch']
    print("last saved model is epoch {}".format(saved_epoch))
    model.load_state_dict(checkpoint['net'])
    print("\n==>Predicting on test dataset")
    
    with torch.no_grad():
        model.eval()
        test_loss = []
        test_true = []
        test_pred = []
        for each_batch in range(test_data_gen.steps):
            test_data = next(test_data_gen)
            test_data = test_data["data"]
            test_x = test_data[0]
            test_y = test_data[1]
            test_interval = test_x[:, :, -17:]  
            test_x = test_x[:, :, :-17]
            
            test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
            test_y = torch.tensor(test_y, dtype=torch.float32).to(device)
            test_interval = torch.tensor(test_interval, dtype=torch.float32).to(device)
                      
            if test_x.size()[1] > 400:
                test_x = test_x[:, :400, :]
                test_y = test_y[:, :400, :]
                test_interval = test_interval[:, :400, :]
            
            #output_step = test_x.size()[1] // args.div
            output_step = 1
            test_output, _ = model(test_x, test_interval, output_step, device)
            test_output = test_output.mean(axis=1)
            loss = test_y * torch.log(test_output + 1e-7) + (
                1 - test_y
            ) * torch.log(1 - test_output + 1e-7)
            loss = torch.neg(torch.sum(loss)) / test_x.size()[0]
            test_loss.append(loss.cpu().detach().numpy())

            for t, p in zip(
                test_y.cpu().numpy().flatten(),
                test_output.cpu().detach().numpy().flatten(),
            ):
                test_true.append(t)
                test_pred.append(p)

        test_loss = np.mean(np.array(test_loss))
        print("Test loss = {:.6f}".format(test_loss))
        print("\n")
        test_pred = np.array(test_pred)
        test_pred = np.stack([1 - test_pred, test_pred], axis=1)
        metrics.print_metrics_binary(test_true, test_pred)