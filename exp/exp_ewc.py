"""
Online learning experiment class
Class to execute online time series forecasting experiments
Inherits from Exp_Main and implements online learning-specific processing
"""

import copy
from tqdm import tqdm
import time
import warnings
from pathlib import Path
import numpy as np
import json

import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader

from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp.exp_main import Exp_Main
from util.metrics import metric, update_metrics, calculate_metrics
from util.tools import test_params_flop
from util.metrics_collector import TimestepMetricsCollector, PredictionCollector, calculate_metrics as calc_metrics, get_memory_usage, get_gpu_memory_usage
from util.buffer import Buffer



warnings.filterwarnings('ignore')

# =============================
# Base class for online learning
# =============================
class Exp_EWC(Exp_Main):
    """
    Base class for online time series forecasting experiments
    Inherits from Exp_Main and implements online learning-specific processing
    train() is inherited from Exp_Main; mainly implements online learning in vali and test

    Main features:
    - Data acquisition in online learning phase (test, online)
    - Online learning with/without information leakage
    - Sequential model updates
    - Performance evaluation of online learning
    """
    def __init__(self, args):
        """
        Initialization
        - Configure online learning phase
        - Configure data acquisition for sequential learning
        """
        super().__init__(args)
        # Configure data acquisition for sequential learning
        self.wrap_data_kwargs.update(gap=self.args.pred_len)

        # FAR hyperparameters (obtained from args)
        self.ewc_lambda = args.ewc_lambda
        if self.ewc_lambda > 0:
            # Initialize dictionaries to store Fisher information matrix and optimal parameters of previous task
            self.fisher_matrix = {}
            self.optimal_params = {}
            print(f"EWC is enabled with lambda = {self.ewc_lambda}")

        self.is_initialized = False
        if args.checkpoints:
            # When train is performed offline (checkpoint exists)
            # 1. self.model becomes optimal parameters of previous task
            # 2. Calculate Fisher information matrix using training dataset
            # 3. Perform online learning on valid, test; EWC is applied at this time
            self.is_checkpoint = True
        else:
            # When train is performed online (checkpoint doesn't exist)
            # 1. Perform online learning on model in train, adding data to buffer at that time
            # 2. Calculate Fisher matrix and optimal parameters
            # 3. Perform online learning on valid, test; EWC is applied at this time
            self.is_checkpoint = False
            self.buffer_size = getattr(args, 'buffer_size', 64)      # Number of data to use for Fisher calculation
            self.mini_batch = getattr(args, 'mini_batch', 4)
            self.buffer = Buffer(self.buffer_size, self.device)
            self.count = 0

    def _initialize_ewc_from_loader(self, train_loader, criterion):
        """Calculate Fisher matrix and optimal parameters from data loader"""
        if self.is_initialized: return
        print("\nCalculating EWC components from training data loader...")
        self._calculate_fisher_matrix(train_loader, criterion)
        print("EWC components have been initialized.\n")
        self.is_initialized = True

    def _initialize_ewc_from_buffer(self, criterion):
        """Calculate Fisher matrix and optimal parameters from buffer"""
        if self.is_initialized: return

        if self.buffer.is_empty():
            print("Warning: Buffer is empty. Cannot initialize EWC.")
            return

        print("Initializing EWC from buffer...")
        buff = self.buffer.get_data(self.buffer_size)
        x_data, y_data, x_mark_data, y_mark_data = buff[0], buff[1], buff[2], buff[3]
        buffer_dataset = TensorDataset(x_data, y_data, x_mark_data, y_mark_data)
        buffer_loader = DataLoader(buffer_dataset, batch_size=self.mini_batch)

        self._calculate_fisher_matrix(buffer_loader, criterion)
        print("EWC components have been initialized.\n")
        self.is_initialized = True

    def _calculate_fisher_matrix(self, data_loader, criterion):
        """Common logic for Fisher matrix calculation"""
        self.model.eval()
        # Deep copy to completely separate as past parameters
        self.optimal_params = {name: p.clone().detach() for name, p in self.model.named_parameters() if p.requires_grad}
        self.fisher_matrix = {name: torch.zeros_like(p) for name, p in self.optimal_params.items()}

        for batch in data_loader:
            self.model.zero_grad()
            outputs = self.forward(batch)
            loss = self.train_loss(criterion, batch, outputs)
            loss.backward()
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None and name in self.fisher_matrix:
                    self.fisher_matrix[name] += p.grad.data.clone() ** 2

        num_samples = len(data_loader.dataset)
        for name in self.fisher_matrix:
            self.fisher_matrix[name] /= num_samples

    def _add_ewc_regularization_grad(self):
        """
        Calculate gradient of EWC regularization term and add to existing gradient.
        Assumes self.fisher_matrix and self.optimal_params have been calculated.
        This method should be called immediately after loss.backward() in training loop.
        """
        # Do nothing if EWC is disabled or parameters for EWC are not saved
        if self.ewc_lambda <= 0 or not self.optimal_params or not self.fisher_matrix:
            return

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                # Process only if gradient exists and important past parameters are saved
                if p.grad is not None and name in self.optimal_params:
                    # Calculate EWC gradient: λ * Fisher * (current parameters - past optimal parameters)
                    # Add calculated EWC gradient to gradient calculated from loss function
                    regularization_grad = p.grad + self.ewc_lambda * self.fisher_matrix[name] * (p.data - self.optimal_params[name])
                    regularization_grad = regularization_grad.to(dtype=p.grad.dtype)
                    p.grad.copy_(regularization_grad)

    def _update_train(self, batch, criterion, optimizer, scaler=None):
        """
        Add data to buffer during training
        """
        loss, outputs = self._update(batch, criterion, optimizer, scaler)
        idx = self.count + torch.arange(batch[1].size(0)).to(self.device)
        self.count += batch[1].size(0)
        self.buffer.add_data(*batch[:4], idx)  # Add data and index to buffer
        return loss, outputs

    def _update(self, batch, criterion, optimizer, scaler=None):
        """
        Model update (backpropagation and parameter update)

        Args:
            batch: Input batch data
            criterion: Loss function
            optimizer: Optimizer
            scaler: Scaler for automatic mixed precision

        Returns:
            tuple: (loss, outputs)
        """

        if not isinstance(optimizer, tuple):
            optimizer = (optimizer, )  # Convert to tuple
        for optim in optimizer:
            optim.zero_grad()  # Clear gradients
        outputs = self.forward(batch)  # Forward propagation
        loss = self.train_loss(criterion, batch, outputs)  # Loss calculation
        loss.backward()

        if self.is_initialized:
            self._add_ewc_regularization_grad() # EWC regularization

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # Optimizer updates all parameters
        # EWC target parameters are updated with modified gradients, others with original gradients
        for optim in optimizer:
            optim.step()

        return loss, outputs


    def online(self, phase='test', savename='', show_progress=False):
        """
        Main loop for online learning
        - Branch processing based on information leakage
        - Make predictions while sequentially updating model
        - Calculate performance metrics (MSE, MAE)
        - online_data: Dataset for online learning (auto-generated if omitted)
        - target_variate: Target variable for evaluation (optional)
        - phase: 'test' or 'valid' or 'online', etc.
        - show_progress: Progress display with tqdm
        Return value: (mse, mae, online_data, [predictions])
        """
        self.phase = phase
        self.savename = savename

        if not self.is_initialized and self.ewc_lambda > 0 and self.phase != 'train':
            if self.enable_detailed_metrics:
                metrics_dir = Path(self.args.savepath_itr) / "initialize" / "metrics"
                self.metrics_collector = TimestepMetricsCollector(metrics_dir, use_gpu=self.args.use_gpu, device=self.device)
                self.metrics_collector.start_collection()

            # Assume first offline learning is complete and calculate Fisher matrix
            print("First time entering online phase. Calculating initial Fisher matrix...")
            timestep_start_time = time.time()
            if self.is_checkpoint: # If checkpoint exists (train was performed offline)
                train_data, train_loader = self._get_data(flag='train', setting='offline_learn')
                self._initialize_ewc_from_loader(train_loader, self._select_criterion())
            else: # If checkpoint doesn't exist (train was performed online)
                self._initialize_ewc_from_buffer(self._select_criterion())

            # Metrics collection during initialization
            if self.enable_detailed_metrics:
                timestep_time = time.time() - timestep_start_time
                memory_usage_percent, memory_usage_str = get_memory_usage()
                gpu_memory_allocated_mb, gpu_memory_max_allocated_mb = get_gpu_memory_usage(self.device)
                if self.metrics_collector:
                    self.metrics_collector.add_timestep_metrics(
                        timestep=-1,
                        mse=0,
                        mae=0,
                        prediction_time=timestep_time,
                        memory_usage_str=memory_usage_str,
                        memory_usage_percent=memory_usage_percent,
                        gpu_memory_allocated_mb=gpu_memory_allocated_mb,
                        gpu_memory_max_allocated_mb=gpu_memory_max_allocated_mb
                    )
            if self.enable_detailed_metrics and self.metrics_collector:
                self.metrics_collector.stop_collection()
                self.metrics_collector.save_metrics()

        # Initialize metrics collection
        if self.enable_detailed_metrics:
            metrics_dir = Path(self.args.savepath_itr) / self.savename / "metrics"
            self.metrics_collector = TimestepMetricsCollector(metrics_dir, use_gpu=self.args.use_gpu, device=self.device)
            self.metrics_collector.start_collection()

        # Initialize prediction result collection
        if self.save_prediction:
            predictions_dir = Path(self.args.savepath_itr) / self.savename / "predictions"
            self.prediction_collector = PredictionCollector(predictions_dir)


        # Auto-generate dataset if unspecified
        online_data = get_dataset(self.args, self.phase, 'online', self.device, wrap_class=[Dataset_Recent], **self.wrap_data_kwargs)
        online_loader = get_dataloader(online_data, self.args, setting='online')


        # Dictionary for accumulating performance metrics
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}

        # Prepare optimizer, loss function, and AMP scaler
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        # Progress display (display progress bar with tqdm)
        if show_progress:
            online_loader = tqdm(online_loader, mininterval=10)

        # Main loop for online learning
        for i, (recent_batch, current_batch) in enumerate(online_loader):
            timestep_start_time = time.time()

            self.model.train()  # Set model to training mode
            # Sequentially update model online (update parameters with recent_batch)
            if self.phase == 'train':
                self._update_train(recent_batch, criterion, model_optim, scaler)
            else:
                self._update(recent_batch, criterion, model_optim, scaler)

            self.model.eval()  # Set model to inference mode
            with torch.no_grad():
                # Make predictions with current_batch
                outputs = self.forward(current_batch)

                # Collect prediction results
                if self.prediction_collector:
                    self.prediction_collector.add_prediction(i, current_batch[0], current_batch[1], outputs)
                # Update performance metrics (MSE, MAE, etc.)
                update_metrics(outputs, current_batch[self.label_position].to(self.device), statistics)

            # Metrics collection per timestep
            if self.enable_detailed_metrics:
                timestep_time = time.time() - timestep_start_time
                memory_usage_percent, memory_usage_str = get_memory_usage()
                gpu_memory_allocated_mb, gpu_memory_max_allocated_mb = get_gpu_memory_usage(self.device)

                # Calculate MSE/MAE at current timestep
                current_metrics = calc_metrics(outputs, current_batch[self.label_position].to(self.device))

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

        # Aggregate performance metrics for all samples
        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        print(self.phase, 'mse:{}, mae:{}'.format(mse, mae))
        return mse, mae
