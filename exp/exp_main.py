"""
Main experiment class
Class to execute basic time series forecasting experiments
Inherits from Exp_Basic and implements concrete experiment processing
"""

import importlib
from tqdm import tqdm
import os
import time
import warnings
from pathlib import Path
import numpy as np

import torch
from torch.optim import lr_scheduler
import torch.distributed as dist

from exp.exp_basic import Exp_Basic
from util.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, load_model_compile
from util.metrics import metric, update_metrics, calculate_metrics
from util.metrics_collector import TimestepMetricsCollector, PredictionCollector, calculate_metrics as calc_metrics, get_memory_usage, get_gpu_memory_usage


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        # Initialize metrics collection
        self.metrics_collector = None
        self.enable_detailed_metrics = getattr(args, 'enable_detailed_metrics', False)

        # Initialize prediction result collection
        self.prediction_collector = None
        self.save_prediction = getattr(args, 'save_prediction', False)

    def _unfreeze(self, model):
        pass

    @property
    def _model(self):
        if self.args.local_rank >= 0:
            return self.model.module
        return self.model


    def vali(self, vali_data, vali_loader, criterion):
        """
        Method to execute validation processing

        Args:
            vali_data: Validation dataset
            vali_loader: Validation dataloader
            criterion: Loss function

        Returns:
            float: Average validation loss
        """
        self.phase = 'valid'  # Set phase to validation
        total_loss = []
        self.model.eval()  # Set model to evaluation mode
        # Disable gradient computation to reduce memory usage
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                # Get predictions with forward propagation
                outputs = self.forward(batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Use first element if tuple

                # Get ground truth
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)

                # Calculate loss
                loss = criterion(outputs, true)
                total_loss.append(loss.item())

        # Calculate average loss
        total_loss = np.average(total_loss)

        # Synchronize loss across all processes in distributed learning
        if self.args.local_rank != -1:
            total_loss = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item()

        return total_loss

    def train(self):
        """
        Method to execute model training

        Args:

        Returns:
            tuple: (model, train_data, train_loader, vali_data, vali_loader)
        """
        # =====================================
        # 1. Prepare data
        # =====================================
        self.phase = 'train'
        train_data, train_loader = self._get_data(flag='train', setting='offline_learn')
        if self.args.valid:
            vali_data, vali_loader = self._get_data(flag='valid', setting='offline_learn')

        # =====================================
        # 2. Set checkpoint path
        # =====================================
        path = os.path.join(self.args.savepath_itr, 'checkpoints')

        # =====================================
        # 3. Initial training setup
        # =====================================
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()  # Select optimizer
        criterion = self._select_criterion()    # Select loss function

        # Configure automatic mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        # =====================================
        # 4. Configure learning rate scheduler
        # =====================================
        if self.args.lradj == 'TST':
            # OneCycleLR scheduler for TST
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )
        elif self.args.model == 'GPT4TS':
            # CosineAnnealingLR scheduler for GPT4TS
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model_optim, T_max=self.args.tmax, eta_min=1e-8
            )
        else:
            scheduler = None

        # =====================================
        # 5. Epoch loop
        # =====================================
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # Set sampler for distributed learning
            if self.args.local_rank != -1:
                train_loader.sampler.set_epoch(epoch)
                if hasattr(self, 'online_phases') and 'valid' not in self.online_phases:
                    vali_loader.sampler.set_epoch(epoch)

            self.model.train()  # Set model to training mode
            epoch_time = time.time()

            # =====================================
            # 6. Batch loop (one epoch of training)
            # =====================================
            for i, batch in enumerate(train_loader):
                iter_count += 1

                # Model update (forward + backward + parameter update)
                loss, _ = self._update(batch, criterion, model_optim, scaler)
                train_loss.append(loss.item())

                # Update learning rate per step for TST scheduler
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # =====================================
            # 7. End of epoch processing
            # =====================================
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss), end=' ')

            if self.args.valid:
                if epoch >= self.args.begin_valid_epoch:
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                    print("Vali Loss: {:.7f}".format(vali_loss))
                    early_stopping(vali_loss, self, path)
                else:
                    print("epoch < begin_valid_epoch")
            else:
                # For train_only, use training loss for early stopping
                early_stopping(train_loss, self, path)
            self.phase = 'train'

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # Check early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Adjust learning rate (for non-TST cases)
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # =====================================
        # 8. Post-training processing
        # =====================================
        if self.args.train_epochs > 0:
            print('Best Valid MSE:', -early_stopping.best_score)
            # Load best checkpoint
            self.load_state_dict(early_stopping.best_checkpoint, strict=not (hasattr(self.args, 'freeze') and self.args.freeze))

            # Save checkpoint
            if path and self.args.local_rank <= 0:
                if not os.path.exists(path):
                    os.makedirs(path)
                print('Save checkpoint to', path)
                torch.save(self.state_dict(local_rank=self.args.local_rank), path + '/' + 'checkpoint.pth')

        return

    def test(self, phase='test', savename=''):
        """
        Method to execute test processing

        Args:

        Returns:
            tuple: (MSE, MAE, test_data, test_loader)
        """
        self.phase = phase
        self.savename = savename

        # Initialize metrics collection
        if self.enable_detailed_metrics:
            metrics_dir = Path(self.args.savepath_itr) / self.savename / "metrics"
            self.metrics_collector = TimestepMetricsCollector(metrics_dir, use_gpu=self.args.use_gpu, device=self.device)
            self.metrics_collector.start_collection()

        # Initialize prediction result collection
        if self.save_prediction:
            predictions_dir = Path(self.args.savepath_itr) / self.savename / "predictions"
            self.prediction_collector = PredictionCollector(predictions_dir)

        # Prepare test data
        test_data, test_loader = self._get_data(flag=self.phase, setting='offline_test')

        # =====================================
        # Execute test
        # =====================================
        # Initialize statistics
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}

        test_loader = tqdm(test_loader, mininterval=10)

        self.model.eval()  # Set model to evaluation mode
        # Disable gradient computation to reduce memory usage
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                timestep_start_time = time.time()
                # Get predictions with forward propagation
                outputs = self.forward(batch)

                # Get ground truth
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)

                # Collect prediction results
                if self.prediction_collector:
                    self.prediction_collector.add_prediction(i, batch[0], batch[1], outputs)
                # Update metrics
                update_metrics(outputs, true, statistics)

                # Metrics collection per timestep
                if self.enable_detailed_metrics:
                    timestep_time = time.time() - timestep_start_time
                    memory_usage_percent, memory_usage_str = get_memory_usage()
                    gpu_memory_allocated_mb, gpu_memory_max_allocated_mb = get_gpu_memory_usage(self.device)

                    # Calculate MSE/MAE at current timestep
                    current_metrics = calc_metrics(outputs, batch[self.label_position].to(self.device))

                    if self.metrics_collector:
                        self.metrics_collector.add_timestep_metrics(
                            timestep=i,
                            mse=current_metrics['mse'],
                            mae=current_metrics['mae'],
                            prediction_time=timestep_time,
                            memory_usage_str=memory_usage_str,
                            memory_usage_percent=memory_usage_percent,
                            gpu_memory_allocated_mb=gpu_memory_allocated_mb,
                            gpu_memory_max_allocated_mb=gpu_memory_max_allocated_mb
                        )

        # End metrics collection
        if self.enable_detailed_metrics and self.metrics_collector:
            self.metrics_collector.stop_collection()
            self.metrics_collector.save_metrics()

        # Save prediction results
        if self.prediction_collector:
            print(f'save prediction is to heavy, skip')
            # self.prediction_collector.save_predictions()

        # Calculate final metrics
        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        print(self.phase, 'mse:{}, mae:{}'.format(mse, mae))

        return mse, mae
