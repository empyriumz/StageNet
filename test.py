import os
import argparse
import torch
import numpy as np
from train import import_module
import json
import pickle
from utils import metrics

def main(test_config):
    # load model architecture
    model_path = test_config["trained_model_path"]
    model_config = torch.load(model_path)["config"]   
    model = import_module("model", model_config)(**model_config["model"]["args"])
    #model.summary()
    
    # setup data_loader instances
    test_data_loader = import_module("test_data_loader", test_config)(**test_config["test_data_loader"]["args"])
    # load state dict
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        test_loss = []
        test_true = []
        test_pred = []
        for _ in range(test_data_loader.steps):
            test_data = next(test_data_loader)
            test_interval = test_data["interval"]
            test_mask = test_data["mask"]
            test_x, test_y = test_data["data"]
            test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
            test_y = torch.tensor(test_y, dtype=torch.float32).to(device)
            test_mask = torch.tensor(test_mask, dtype=torch.float32).to(device)
            test_interval = torch.tensor(test_interval, dtype=torch.float32).to(
                device
            )
            if test_interval.size()[1] > 400:
                test_x = test_x[:, :400, :]
                test_y = test_y[:, :400, :]
                test_interval = test_interval[:, :400, :]
                test_mask = test_mask[:, :400, :]

            test_output, _ = model(test_x, test_interval, device)
            test_output = test_mask * test_output
            loss = test_y * torch.log(test_output + 1e-7) + (
                1 - test_y
            ) * torch.log(1 - test_output + 1e-7)
            loss = torch.sum(loss, dim=1) / torch.sum(test_mask, dim=1)
            loss = torch.neg(torch.sum(loss))
            test_loss.append(loss.cpu().detach().numpy())

            for m, t, p in zip(
                test_mask.cpu().numpy().flatten(),
                test_y.cpu().numpy().flatten(),
                test_output.cpu().detach().numpy().flatten(),
            ):
                if np.equal(m, 1):
                    test_true.append(t)
                    test_pred.append(p)
    test_pred = np.array(test_pred)
    test_pred = np.stack([1 - test_pred, test_pred], axis=1)
    result = metrics.print_metrics_binary(test_true, test_pred)
    test_result = {"test_"+key: value for key, value in result.items()}
    log = {
        "val_loss": np.mean(np.array(test_loss) / len(test_data_loader)),
    }
    log = {**log, **test_result}
    print(log)
    save_dir = os.path.join(os.path.abspath(os.path.join(model_path, "..")))
        
    with open(os.path.join(save_dir, "test-results.pkl"), "wb") as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--test_config",
        default=None,
        type=str,
        help="test dataloader config",
    )

    args = parser.parse_args()
    assert args.test_config != None, "need data and model to test"
    test_config = json.load(open(args.test_config))
   
    main(test_config)