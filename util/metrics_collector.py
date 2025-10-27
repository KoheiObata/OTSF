#!/usr/bin/env python3
"""
Metrics collection utility
"""

import json
import time
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import queue
import json
import time

class TimestepMetricsCollector:
    """Metrics collection class per timestep"""

    def __init__(self, save_dir: Optional[Path] = None, use_gpu: bool = False, device=None):
        self.save_dir = save_dir
        self.use_gpu = use_gpu
        self.device = device
        self.timestep_metrics = []
        self.metrics_queue = queue.Queue()
        self.running = False
        self.collection_thread = None

    def start_collection(self):
        """Start metrics collection"""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_system_metrics)
        self.collection_thread.daemon = True
        self.collection_thread.start()

    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()

    def _collect_system_metrics(self):
        """Collect system metrics in background"""
        while self.running:
            try:
                timestamp = time.time()

                # System memory information
                system_memory = psutil.virtual_memory()
                system_memory_percent = system_memory.percent
                system_memory_used_mb = system_memory.used / (1024 * 1024)
                system_memory_available_mb = system_memory.available / (1024 * 1024)

                # CPU usage
                cpu_usage = psutil.cpu_percent()

                # GPU memory information (if GPU usage is enabled and available)
                gpu_memory_info = {}
                if self.use_gpu and torch.cuda.is_available():
                    # Use that device if device is specified
                    if self.device is not None:
                        if isinstance(self.device, str) and self.device.startswith('cuda:'):
                            device_idx = int(self.device.split(':')[1])
                        elif isinstance(self.device, torch.device) and self.device.type == 'cuda':
                            device_idx = self.device.index
                        else:
                            device_idx = 0

                        with torch.cuda.device(device_idx):
                            gpu_memory_allocated = torch.cuda.memory_allocated()
                            gpu_memory_max_allocated = torch.cuda.max_memory_allocated()
                            gpu_memory_reserved = torch.cuda.memory_reserved()
                            gpu_memory_total = torch.cuda.get_device_properties(device_idx).total_memory
                    else:
                        # Use current device if device is not specified
                        gpu_memory_allocated = torch.cuda.memory_allocated()
                        gpu_memory_max_allocated = torch.cuda.max_memory_allocated()
                        gpu_memory_reserved = torch.cuda.memory_reserved()
                        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory

                    gpu_memory_info = {
                        'gpu_memory_allocated_mb': gpu_memory_allocated / (1024 * 1024),
                        'gpu_memory_max_allocated_mb': gpu_memory_max_allocated / (1024 * 1024),
                        'gpu_memory_reserved_mb': gpu_memory_reserved / (1024 * 1024),
                        'gpu_memory_total_mb': gpu_memory_total / (1024 * 1024),
                        'gpu_memory_allocated_percent': (gpu_memory_allocated / gpu_memory_total) * 100,
                        'gpu_memory_reserved_percent': (gpu_memory_reserved / gpu_memory_total) * 100
                    }

                self.metrics_queue.put({
                    'timestamp': timestamp,
                    'system_memory_percent': system_memory_percent,
                    'system_memory_used_mb': system_memory_used_mb,
                    'system_memory_available_mb': system_memory_available_mb,
                    'cpu_usage': cpu_usage,
                    **gpu_memory_info
                })

                time.sleep(1)  # Collect at 1 second intervals
            except Exception as e:
                print(f"System metrics collection error: {e}")
                break

    def add_timestep_metrics(self, timestep: int, mse: float, mae: float,
                           prediction_time: float, memory_usage_str: str, memory_usage_percent: float,
                           gpu_memory_allocated_mb: float = 0.0, gpu_memory_max_allocated_mb: float = 0.0):
        """Add metrics for one timestep"""
        # Debug: Detect and convert tensors
        def convert_tensor_to_float(value):
            if torch.is_tensor(value):
                return value.item() if value.numel() == 1 else value.detach().cpu().numpy().tolist()
            return value

        # Check each value and convert tensors
        mse = convert_tensor_to_float(mse)
        mae = convert_tensor_to_float(mae)
        prediction_time = convert_tensor_to_float(prediction_time)
        memory_usage_percent = convert_tensor_to_float(memory_usage_percent)
        gpu_memory_allocated_mb = convert_tensor_to_float(gpu_memory_allocated_mb)
        gpu_memory_max_allocated_mb = convert_tensor_to_float(gpu_memory_max_allocated_mb)

        self.timestep_metrics.append({
            'timestep': timestep,
            'mse': mse,
            'mae': mae,
            'prediction_time': prediction_time,
            'memory_usage_str': memory_usage_str,
            'memory_usage_percent': memory_usage_percent,
            'gpu_memory_allocated_mb': gpu_memory_allocated_mb,
            'gpu_memory_max_allocated_mb': gpu_memory_max_allocated_mb,
            'timestamp': time.time()
        })

    def get_system_metrics(self):
        """Get collected system metrics"""
        metrics_list = []
        while not self.metrics_queue.empty():
            metrics_list.append(self.metrics_queue.get())
        return metrics_list

    def save_metrics(self, save_dir: Optional[Path] = None):
        """Save metrics to file"""
        if save_dir is None:
            save_dir = self.save_dir

        if save_dir is None:
            return

        save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics per timestep
        if self.timestep_metrics:
            timestep_file = save_dir / "timestep_metrics.json"

            # Debug: Detect tensors
            def check_for_tensors(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        if torch.is_tensor(value):
                            print(f"ERROR: Found tensor at {current_path}: {type(value)}")
                            return True
                        elif isinstance(value, (dict, list)):
                            if check_for_tensors(value, current_path):
                                return True
                elif isinstance(obj, list):
                    for i, value in enumerate(obj):
                        current_path = f"{path}[{i}]"
                        if torch.is_tensor(value):
                            print(f"ERROR: Found tensor at {current_path}: {type(value)}")
                            return True
                        elif isinstance(value, (dict, list)):
                            if check_for_tensors(value, current_path):
                                return True
                return False

            # Check for tensors
            if check_for_tensors(self.timestep_metrics):
                print("ERROR: Found tensors in timestep_metrics, cannot serialize to JSON")
                return

            try:
                with open(timestep_file, 'w') as f:
                    json.dump(self.timestep_metrics, f, indent=2)
            except TypeError as e:
                print(f"ERROR: JSON serialization failed: {e}")
                print(f"ERROR: timestep_metrics type: {type(self.timestep_metrics)}")
                print(f"ERROR: timestep_metrics length: {len(self.timestep_metrics)}")
                if self.timestep_metrics:
                    print(f"ERROR: First item type: {type(self.timestep_metrics[0])}")
                    print(f"ERROR: First item keys: {list(self.timestep_metrics[0].keys())}")
                    for key, value in self.timestep_metrics[0].items():
                        print(f"ERROR: {key}: {type(value)} = {value}")
                raise

        # System metrics
        system_metrics = self.get_system_metrics()
        if system_metrics:
            system_file = save_dir / "system_metrics.json"
            with open(system_file, 'w') as f:
                json.dump(system_metrics, f, indent=2)

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics"""
        if not self.timestep_metrics:
            return {}

        mse_values = [m['mse'] for m in self.timestep_metrics]
        mae_values = [m['mae'] for m in self.timestep_metrics]
        time_values = [m['prediction_time'] for m in self.timestep_metrics]
        memory_percent_values = [m['memory_usage_percent'] for m in self.timestep_metrics]
        memory_str_values = [m['memory_usage_str'] for m in self.timestep_metrics]

                # GPU memory information (if GPU usage is enabled)
        gpu_memory_summary = {}
        if self.use_gpu and torch.cuda.is_available():
            gpu_allocated_values = [m.get('gpu_memory_allocated_mb', 0) for m in self.timestep_metrics]
            gpu_max_allocated_values = [m.get('gpu_memory_max_allocated_mb', 0) for m in self.timestep_metrics]

            gpu_memory_summary = {
                'max_allocated_mb': np.max(gpu_allocated_values),
                'avg_allocated_mb': np.mean(gpu_allocated_values),
                'peak_allocated_mb': np.max(gpu_max_allocated_values),
                'min_allocated_mb': np.min(gpu_allocated_values)
            }

        # System metrics summary
        system_metrics_summary = {}
        system_metrics = self.get_system_metrics()
        if system_metrics:
            system_memory_percent = [m.get('system_memory_percent', 0) for m in system_metrics]
            system_memory_used = [m.get('system_memory_used_mb', 0) for m in system_metrics]
            cpu_usage = [m.get('cpu_usage', 0) for m in system_metrics]

            system_metrics_summary = {
                'system_memory': {
                    'max_percent': np.max(system_memory_percent),
                    'avg_percent': np.mean(system_memory_percent),
                    'max_used_mb': np.max(system_memory_used),
                    'avg_used_mb': np.mean(system_memory_used)
                },
                'cpu_usage': {
                    'max_percent': np.max(cpu_usage),
                    'avg_percent': np.mean(cpu_usage)
                }
            }

            # GPU metrics (if GPU usage is enabled and available)
            if self.use_gpu and torch.cuda.is_available():
                gpu_allocated = [m.get('gpu_memory_allocated_mb', 0) for m in system_metrics]
                gpu_max_allocated = [m.get('gpu_memory_max_allocated_mb', 0) for m in system_metrics]
                gpu_reserved = [m.get('gpu_memory_reserved_mb', 0) for m in system_metrics]
                gpu_allocated_percent = [m.get('gpu_memory_allocated_percent', 0) for m in system_metrics]

                system_metrics_summary['gpu_memory'] = {
                    'max_allocated_mb': np.max(gpu_allocated),
                    'avg_allocated_mb': np.mean(gpu_allocated),
                    'max_reserved_mb': np.max(gpu_reserved),
                    'avg_reserved_mb': np.mean(gpu_reserved),
                    'max_allocated_percent': np.max(gpu_allocated_percent),
                    'avg_allocated_percent': np.mean(gpu_allocated_percent),
                    'peak_allocated_mb': np.max(gpu_max_allocated) if gpu_max_allocated else 0
                }

        return {
            'mse': {
                'mean': np.mean(mse_values),
                'std': np.std(mse_values),
                'min': np.min(mse_values),
                'max': np.max(mse_values)
            },
            'mae': {
                'mean': np.mean(mae_values),
                'std': np.std(mae_values),
                'min': np.min(mae_values),
                'max': np.max(mae_values)
            },
            'total_time': np.sum(time_values),
            'memory_usage': {
                'max_percent': np.max(memory_percent_values),
                'avg_percent': np.mean(memory_percent_values),
                'max_str': max(memory_str_values, key=lambda x: float(x.replace('MB', ''))),
                'avg_str': f"{np.mean([float(x.replace('MB', '')) for x in memory_str_values]):.1f}MB"
            },
            'gpu_memory': gpu_memory_summary,
            'system_metrics': system_metrics_summary,
            'num_timesteps': len(self.timestep_metrics)
        }


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Calculate metrics from predictions and targets"""
    # If predictions is a tuple, use first element
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Use first element

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # MSE
    mse = np.mean((predictions - targets) ** 2)

    # MAE
    mae = np.mean(np.abs(predictions - targets))

    return {
        'mse': float(mse),
        'mae': float(mae)
    }

def get_memory_usage() -> tuple[float, str]:
    """Get current memory usage and amount"""
    memory = psutil.virtual_memory()
    usage_percent = memory.percent
    usage_mb = memory.used / (1024 * 1024)  # Convert to MB
    usage_str = f"{usage_mb:.1f}MB"
    return usage_percent, usage_str

def get_gpu_memory_usage(device=None, reset_peak=True) -> tuple[float, float]:
    """Get GPU memory usage (if available)"""
    if device == 'cpu':
        return 0.0, 0.0
    if torch.cuda.is_available():
        try:
            # Use current device if device is not specified
            if device is None:
                device = torch.cuda.current_device()
            elif isinstance(device, str) and device.startswith('cuda:'):
                device = int(device.split(':')[1])
            elif isinstance(device, torch.device) and device.type == 'cuda':
                device = device.index

            # Switch to specified device and get memory usage
            with torch.cuda.device(device):
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                if reset_peak:
                    torch.cuda.reset_peak_memory_stats(device)

            return allocated_mb, max_allocated_mb
        except Exception as e:
            print(f"GPU memory retrieval error: {e}")
            return 0.0, 0.0
    else:
        print("GPU is not available")
        return 0.0, 0.0


def get_model_complexity_info(model) -> dict:
    """Get model computational complexity information"""
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params
    }


def get_gpu_utilization() -> dict:
    """Get GPU utilization information (if available)"""
    if torch.cuda.is_available():
        try:
            # Get GPU usage (using pynvml library)
            gpu_utilization = 0.0
            memory_bandwidth_utilization = 0.0
            gpu_temperature = 0.0

            try:
                # If pynvml library is available
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = utilization.gpu

                # GPU temperature
                gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # Memory bandwidth utilization (simplified version)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_bandwidth_utilization = (memory_info.used / memory_info.total) * 100

            except ImportError:
                # Use nvidia-smi command if pynvml is not available
                try:
                    import subprocess
                    # GPU utilization
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_utilization = float(result.stdout.strip())

                    # GPU temperature
                    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_temperature = float(result.stdout.strip())
                except:
                    pass

            # Calculate GPU memory usage (obtained from PyTorch)
            memory_allocated = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_utilization = (memory_allocated / memory_total) * 100

            return {
                'gpu_utilization_percent': gpu_utilization,
                'memory_utilization_percent': memory_utilization,
                'memory_bandwidth_utilization_percent': memory_bandwidth_utilization,
                'gpu_temperature_celsius': gpu_temperature
            }
        except Exception as e:
            print(f"GPU utilization retrieval error: {e}")
            return {
                'gpu_utilization_percent': 0.0,
                'memory_utilization_percent': 0.0,
                'memory_bandwidth_utilization_percent': 0.0,
                'gpu_temperature_celsius': 0.0
            }
    return {
        'gpu_utilization_percent': 0.0,
        'memory_utilization_percent': 0.0,
        'memory_bandwidth_utilization_percent': 0.0,
        'gpu_temperature_celsius': 0.0
    }


def calculate_flops(model, input_shape) -> int:
    """Calculate FLOPs of model (simplified version)"""
    try:
        # Create input tensor
        input_tensor = torch.randn(input_shape)

        # FLOPs calculation (simplified version)
        flops = 0
        for module in model.modules():
            if hasattr(module, 'weight'):
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    # Fully connected layer
                    flops += module.in_features * module.out_features
                elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    # Convolution layer
                    if hasattr(module, 'kernel_size'):
                        kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                        flops += module.in_channels * module.out_channels * kernel_size * kernel_size

        return flops
    except Exception as e:
        print(f"FLOPs calculation error: {e}")
        return 0


class PredictionCollector:
    """Prediction result collection class"""

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir
        self.predictions = []
        self.inputs = []
        self.targets = []
        self.timesteps = []

    def add_prediction(self, timestep: int, input_data: torch.Tensor, target_data: torch.Tensor, prediction: torch.Tensor):
        """Add prediction result, input data, and target data"""
        # If prediction is a tuple, use first element
        if isinstance(prediction, tuple):
            prediction = prediction[0]  # Use first element

        # Move tensor to CPU and convert to numpy array
        if isinstance(prediction, torch.Tensor):
            prediction_np = prediction.detach().cpu().numpy()
        else:
            prediction_np = prediction

        if isinstance(input_data, torch.Tensor):
            input_np = input_data.detach().cpu().numpy()
        else:
            input_np = input_data

        if isinstance(target_data, torch.Tensor):
            target_np = target_data.detach().cpu().numpy()
        else:
            target_np = target_data

        # If batch size is greater than 1, add each sample individually
        if len(prediction_np.shape) > 0 and prediction_np.shape[0] > 1:
            for i in range(prediction_np.shape[0]):
                self.predictions.append(prediction_np[i:i+1])
                self.inputs.append(input_np[i:i+1])
                self.targets.append(target_np[i:i+1])
                self.timesteps.append(timestep + i)
        else:
            # If batch size is 1, add as is
            self.predictions.append(prediction_np)
            self.inputs.append(input_np)
            self.targets.append(target_np)
            self.timesteps.append(timestep)

    def save_predictions(self, save_dir: Optional[Path] = None):
        """Save prediction results, input data, and target data to file"""
        if save_dir is None:
            save_dir = self.save_dir

        if save_dir is None or not self.predictions:
            return

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save as list to support different batch sizes
        predictions_file = save_dir / "predictions.npz"

        # Check shape of each array and save as list if not uniform
        try:
            # Check for uniform shape
            pred_shapes = [pred.shape for pred in self.predictions]
            input_shapes = [inp.shape for inp in self.inputs]
            target_shapes = [targ.shape for targ in self.targets]

            # Check if all shapes are the same
            uniform_pred = len(set(str(shape) for shape in pred_shapes)) == 1
            uniform_input = len(set(str(shape) for shape in input_shapes)) == 1
            uniform_target = len(set(str(shape) for shape in target_shapes)) == 1

            if uniform_pred and uniform_input and uniform_target:
                # If uniform shape, save as numpy array as usual
                np.savez(
                    predictions_file,
                    predictions=np.array(self.predictions),
                    inputs=np.array(self.inputs),
                    targets=np.array(self.targets),
                    timesteps=np.array(self.timesteps)
                )
            else:
                # If different shapes, save as list
                np.savez(
                    predictions_file,
                    predictions=self.predictions,
                    inputs=self.inputs,
                    targets=self.targets,
                    timesteps=np.array(self.timesteps),
                    pred_shapes=pred_shapes,
                    input_shapes=input_shapes,
                    target_shapes=target_shapes
                )

        except Exception as e:
            # If error occurs, save as list
            print(f"Warning: Failed to save as uniform array, saving as list: {e}")
            np.savez(
                predictions_file,
                predictions=self.predictions,
                inputs=self.inputs,
                targets=self.targets,
                timesteps=np.array(self.timesteps)
            )

        # Save metadata as JSON
        metadata = {
            'num_predictions': len(self.predictions),
            'prediction_shape': self.predictions[0].shape if self.predictions else None,
            'input_shape': self.inputs[0].shape if self.inputs else None,
            'target_shape': self.targets[0].shape if self.targets else None,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        metadata_file = save_dir / "predictions_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Predictions, inputs, and targets saved to: {predictions_file}")
        print(f"Metadata saved to: {metadata_file}")

    def get_predictions(self) -> List[np.ndarray]:
        """Get collected prediction results"""
        return self.predictions

    def get_inputs(self) -> List[np.ndarray]:
        """Get collected input data"""
        return self.inputs

    def get_targets(self) -> List[np.ndarray]:
        """Get collected target data"""
        return self.targets

    def get_timesteps(self) -> List[int]:
        """Get collected timesteps"""
        return self.timesteps

    def clear(self):
        """Clear collected data"""
        self.predictions.clear()
        self.inputs.clear()
        self.targets.clear()
        self.timesteps.clear()

    def get_total_samples(self) -> int:
        """Get number of collected samples"""
        return len(self.predictions)

    def get_batch_size_info(self) -> dict:
        """Get batch size information"""
        if not self.predictions:
            return {}

        batch_sizes = []
        for pred in self.predictions:
            if len(pred.shape) > 0:
                batch_sizes.append(pred.shape[0])
            else:
                batch_sizes.append(1)

        return {
            'total_batches': len(batch_sizes),
            'unique_batch_sizes': list(set(batch_sizes)),
            'batch_size_distribution': {size: batch_sizes.count(size) for size in set(batch_sizes)}
        }


def load_predictions(predictions_file: Path) -> dict:
    """Load prediction result file"""
    try:
        data = np.load(predictions_file, allow_pickle=True)

        # If saved as list
        if 'predictions' in data and isinstance(data['predictions'], np.ndarray) and data['predictions'].dtype == object:
            return {
                'predictions': data['predictions'].tolist(),
                'inputs': data['inputs'].tolist(),
                'targets': data['targets'].tolist(),
                'timesteps': data['timesteps'],
                'pred_shapes': data.get('pred_shapes', None),
                'input_shapes': data.get('input_shapes', None),
                'target_shapes': data.get('target_shapes', None)
            }
        else:
            # If saved as regular numpy array
            return {
                'predictions': data['predictions'],
                'inputs': data['inputs'],
                'targets': data['targets'],
                'timesteps': data['timesteps']
            }
    except Exception as e:
        print(f"Prediction result loading error: {e}")
        return {}