import os
import json
import argparse
import torch
from utils.common_utils import DataLoader, Logger

def get_instance(module, name, config, *args):
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])

def import_module(name, config):
    return getattr(
        __import__("{}.{}".format(name, config[name]["module_name"])),
        config[name]["type"],
    )

def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    train_data_loader = get_instance(DataLoader, "train_data_loader", config)
    val_data_loader = get_instance(DataLoader, "val_data_loader", config)
    
    # build model architecture
    model = import_module("model", config)(**config["model"]["args"])
    # model = get_instance(module_arch, 'arch', config)
    print(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, "optimizer", config, trainable_params)
    lr_scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", config, optimizer
        )
    Trainer = import_module("trainer", config)
    trainer = Trainer(
        model,
        loss,
        metrics,
        optimizer,
        resume=resume,
        config=config,
        data_loader=train_data_loader,
        valid_data_loader=val_data_loader,
        lr_scheduler=lr_scheduler,
        train_logger=train_logger,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAPER Trainer")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args = parser.parse_args()
    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config["trainer"]["save_dir"], config["name"])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)["config"]
    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.json', for example."
        )

    main(config, args.resume)
