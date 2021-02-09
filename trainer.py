import os
import json
import logging
import datetime
import torch
import numpy as np
from utils.visualization import WriterTensorboardX
from utils import metrics

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
class Trainer:
    """
    Base class for all trainers
    """

    def __init__(
        self,
        model,
        optimizer,
        resume,
        config,
        train_data_loader=None,
        val_data_loader=None,
        lr_scheduler=None,
        train_logger=None,
    ):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # setup GPU device if available, move model into configured device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() == True else "cpu"
        )
        self.model = model.to(self.device)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.train_logger = train_logger

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.verbosity = cfg_trainer["verbosity"]
        self.monitor = cfg_trainer.get("monitor", "off")
        self.log_step = cfg_trainer.get(
            "log_step", int(np.sqrt(self.train_data_loader.batch_size))
        )
        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = np.inf if self.mnt_mode == "min" else -np.inf
            self.early_stop = cfg_trainer.get("early_stop", np.inf)

        self.start_epoch = 1

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.checkpoint_dir = os.path.join(
            cfg_trainer["save_dir"], config["name"], start_time
        )
        ensure_dir(self.checkpoint_dir)
        # setup visualization writer instance
        writer_dir = os.path.join(cfg_trainer["log_dir"], config["name"], start_time)
        self.writer = WriterTensorboardX(
            writer_dir, self.logger, cfg_trainer["tensorboardX"]
        )

        # Save configuration file into checkpoint directory:
        config_save_path = os.path.join(self.checkpoint_dir, "config.json")
        
        with open(config_save_path, "w") as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            # save logged informations into log dict
            log = self._train_epoch(epoch)
            # print logged informations to the screen
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            not_improved_count = 0
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min" and log[self.mnt_metric] < self.mnt_best
                    ) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] > self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. Training stops.".format(
                            self.early_stop
                        )
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """

        self.model.train()
        batch_loss = []
        train_true = []
        train_pred = []
        for _ in range(self.train_data_loader.steps):
            batch_data = next(self.train_data_loader)
            batch_interval = batch_data["interval"]
            batch_mask = batch_data["mask"]
            batch_x, batch_y = batch_data["data"]
            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(self.device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(self.device)
            batch_interval = torch.tensor(batch_interval, dtype=torch.float32).to(
                self.device
            )
            batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(self.device)
            # cut long sequence
            if batch_interval.size()[1] > 400:
                batch_x = batch_x[:, :400, :]
                batch_y = batch_y[:, :400, :]
                batch_interval = batch_interval[:, :400, :]
                batch_mask = batch_mask[:, :400, :]

            self.optimizer.zero_grad()
            output, _ = self.model(batch_x, batch_interval, self.device)
            output = output * batch_mask
            loss = batch_y * torch.log(output + 1e-7) + (1 - batch_y) * torch.log(
                1 - output + 1e-7
            )
            loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
            loss = torch.neg(torch.sum(loss)) / self.train_data_loader.batch_size
            batch_loss.append(loss.cpu().detach().numpy())

            loss.backward()
            self.optimizer.step()
            for m, t, p in zip(
                    batch_mask.cpu().numpy().flatten(),
                    batch_y.cpu().numpy().flatten(),
                    output.cpu().detach().numpy().flatten(),
                ):
                    if np.equal(m, 1):
                        train_true.append(t)
                        train_pred.append(p)
        train_pred = np.array(train_pred)
        train_pred = np.stack([1 - train_pred, train_pred], axis=1)
        # print("Trainning loss for epoch {} = {.4f}".format(epoch, batch_loss))
        # print("\n")
        result = metrics.print_metrics_binary(train_true, train_pred)
        train_result = {"train_"+key: value for key, value in result.items()}
        log = {
            "Epoch": epoch,
            "train_loss": np.mean(np.array(batch_loss)),
        }
        log = {**log, **train_result}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        val_log = self._valid_epoch(epoch)
        log = {**log, **val_log}
        
        return log
    
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        self.model.eval()
        with torch.no_grad():
            val_loss = []
            valid_true = []
            valid_pred = []
            for _ in range(self.val_data_loader.steps):
                valid_data = next(self.val_data_loader)
                valid_interval = valid_data["interval"]
                valid_mask = valid_data["mask"]
                valid_x, valid_y = valid_data["data"]
                valid_x = torch.tensor(valid_x, dtype=torch.float32).to(self.device)
                valid_y = torch.tensor(valid_y, dtype=torch.float32).to(self.device)
                valid_mask = torch.tensor(valid_mask, dtype=torch.float32).to(self.device)
                valid_interval = torch.tensor(valid_interval, dtype=torch.float32).to(
                    self.device
                )
                if valid_interval.size()[1] > 400:
                    valid_x = valid_x[:, :400, :]
                    valid_y = valid_y[:, :400, :]
                    valid_interval = valid_interval[:, :400, :]
                    valid_mask = valid_mask[:, :400, :]

                valid_output, _ = self.model(valid_x, valid_interval, self.device)
                valid_output = valid_mask * valid_output
                valid_loss = valid_y * torch.log(valid_output + 1e-7) + (
                    1 - valid_y
                ) * torch.log(1 - valid_output + 1e-7)
                valid_loss = torch.sum(valid_loss, dim=1) / torch.sum(valid_mask, dim=1)
                valid_loss = torch.neg(torch.sum(valid_loss))/ self.val_data_loader.batch_size
                val_loss.append(valid_loss.cpu().detach().numpy())

                for m, t, p in zip(
                    valid_mask.cpu().numpy().flatten(),
                    valid_y.cpu().numpy().flatten(),
                    valid_output.cpu().detach().numpy().flatten(),
                ):
                    if np.equal(m, 1):
                        valid_true.append(t)
                        valid_pred.append(p)
        valid_pred = np.array(valid_pred)
        valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
        result = metrics.print_metrics_binary(valid_true, valid_pred)
        val_result = {"val_"+key: value for key, value in result.items()}
        log = {
            "val_loss": np.mean(np.array(val_loss)),
        }
        log = {**log, **val_result}
        
        return log
     
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "logger": self.train_logger,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = os.path.join(
            self.checkpoint_dir, "checkpoint-epoch{}.pth".format(epoch)
        )
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_name = "model_best_epoch_{}.pth".format(epoch)
            best_path = os.path.join(self.checkpoint_dir, best_name)
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format(best_name))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"]["type"]
            != self.config["optimizer"]["type"]
        ):
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                + "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.train_logger = checkpoint["logger"]
        self.logger.info(
            "Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch)
        )
