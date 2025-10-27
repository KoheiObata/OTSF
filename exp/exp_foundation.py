import torch
import os
import warnings
from tqdm import tqdm
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from exp import Exp_Online
from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from util.metrics import metric, update_metrics, calculate_metrics
from util.metrics_collector import TimestepMetricsCollector, PredictionCollector, calculate_metrics as calc_metrics, get_memory_usage, get_gpu_memory_usage

from chronos import BaseChronosPipeline

warnings.filterwarnings('ignore')

class Exp_Chronos(Exp_Online):
    def __init__(self, args):

        self.args = args
        self.label_position = 1  # Label position (index in batch)
        self.device = self._acquire_device()  # Get device (GPU/CPU)
        # self.device = 'cpu'
        print('self.device', self.device)
        self.wrap_data_kwargs = {}  # Additional arguments for data wrapper

        # Initialize model optimizer (dummy)
        self.model_optim = type('DummyOptimizer', (), {
            'param_groups': [{'lr': args.online_learning_rate}]
        })()

        # Initialize metrics collection
        self.metrics_collector = None
        self.enable_detailed_metrics = getattr(args, 'enable_detailed_metrics', False)

        # Initialize prediction result collection
        self.prediction_collector = None
        self.save_prediction = getattr(args, 'save_prediction', False)

        self.model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
            device_map=self.device,  # use "cpu" for CPU inference
            torch_dtype=torch.bfloat16,
        )

        # Configure data acquisition for sequential learning
        self.wrap_data_kwargs.update(gap=self.args.pred_len)


    def online(self, phase='test', savename='', show_progress=False):
        """
        Online prediction on test data
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

        # Progress display (display progress bar with tqdm)
        if show_progress:
            online_loader = tqdm(online_loader, mininterval=10)

        for i, (recent_batch, current_batch) in enumerate(online_loader):
            timestep_start_time = time.time()

            # Make predictions with current_batch
            # print('current_batch exisist in device:', current_batch[0].device)
            # print('mode exisist in device:', next(self.model.model.parameters()).device)
            batch_x, batch_y, batch_x_mark, batch_y_mark, index = current_batch
            batch_x = batch_x.transpose(1, 2)  # [B, D, L]
            batch_x = batch_x.reshape(-1, self.args.seq_len)  # [B*D, L]
            quantiles, outputs = self.model.predict_quantiles(
                context=batch_x,
                prediction_length=self.args.pred_len,
                quantile_levels=[],
            )
            outputs = outputs.reshape(-1, self.args.enc_in, self.args.pred_len)  # [B*D, H, L]
            outputs = outputs.transpose(1, 2)  # [B, L, H]
            outputs = outputs.to(self.device)
            # print('outputs exisist in device:', outputs.device)

            # Collect prediction results
            if self.prediction_collector:
                self.prediction_collector.add_prediction(index, current_batch[0], current_batch[1], outputs)
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
                    # Convert timestep to integer if it's a tensor
                    timestep_int = index.item() if torch.is_tensor(index) else index
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


    def test(self, phase='test', savename=''):

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
        self.args.batch_size = 1
        test_data, test_loader = self._get_data(flag=self.phase, setting='offline_test')

        # =====================================
        # Execute test
        # =====================================
        # Initialize statistics
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}

        test_loader = tqdm(test_loader, mininterval=10)

        for i, current_batch in enumerate(test_loader):
            timestep_start_time = time.time()

            batch_x, batch_y, batch_x_mark, batch_y_mark, index = current_batch
            batch_x = batch_x.transpose(1, 2)  # [B, D, L]
            batch_x = batch_x.reshape(-1, self.args.seq_len)  # [B*D, L]
            quantiles, outputs = self.model.predict_quantiles(
                context=torch.tensor(batch_x),
                prediction_length=self.args.pred_len,
                quantile_levels=[],
            )
            outputs = outputs.reshape(-1, self.args.enc_in, self.args.pred_len)  # [B*D, H, L]
            outputs = outputs.transpose(1, 2)  # [B, L, H]
            outputs = outputs.to(self.device)

            # Collect prediction results
            if self.prediction_collector:
                self.prediction_collector.add_prediction(index, current_batch[0], current_batch[1], outputs)
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
                    # Convert timestep to integer if it's a tensor
                    # timestep_int = index.item() if torch.is_tensor(index) else index
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

