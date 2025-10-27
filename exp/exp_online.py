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

from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp.exp_main import Exp_Main
from util.metrics import metric, update_metrics, calculate_metrics
from util.tools import test_params_flop
from util.metrics_collector import TimestepMetricsCollector, PredictionCollector, calculate_metrics as calc_metrics, get_memory_usage, get_gpu_memory_usage



warnings.filterwarnings('ignore')

transformers = ['Autoformer', 'Transformer', 'Informer']

# =============================
# Base class for online learning
# =============================
class Exp_Online(Exp_Main):
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

    def _update_online(self, batch, criterion, optimizer, scaler=None, current_batch=None):
        return self._update(batch, criterion, optimizer, scaler)

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
            self._update_online(recent_batch, criterion, model_optim, scaler, current_batch)
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

