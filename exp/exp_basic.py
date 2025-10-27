"""
Basic experiment class
Base class providing common functionality for time series forecasting experiments
"""

import os
import warnings
import importlib
import numpy as np
import typing
import torch
from torch import optim
import torch.nn as nn
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP


import settings
import models.normalization #RevIn
from data_provider.data_factory import get_dataset, get_dataloader
from util.tools import remove_state_key_prefix


class Exp_Basic(object):
    """Basic experiment class - provides common functionality for time series forecasting experiments"""

    def __init__(self, args):
        """
        Initialization

        Args:
            args: Experiment configuration argument object
        """
        self.args = args
        self.label_position = 1  # Label position (index in batch)
        self.device = self._acquire_device()  # Get device (GPU/CPU)
        self.wrap_data_kwargs = {}  # Additional arguments for data wrapper
        self.model_optim = None  # Model optimizer
        model = self._build_model()  # Build model
        if model is not None:
            print('move model to device : ', self.device)
            self.model = model.to(self.device)  # Move model to device
            self.model_optim = self._select_optimizer()  # Select optimizer

    def _acquire_device(self):
        """
        Get device (GPU/CPU)

        Returns:
            torch.device: Device to use
        """
        if self.args.use_gpu:
            # When using GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            # When using CPU
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self, model=None, framework_class=None):
        """
        Method to build model
        Concrete implementation of abstract method defined in Exp_Basic

        Args:
            model: Existing model (create new if None)
            framework_class: Framework class to wrap model (optional)

        Returns:
            torch.nn.Module: Built model
        """
        # =====================================
        # 1. Create basic model
        # =====================================
        if model is None:
            if self.args.model.endswith('_Ensemble'):
                # For Ensemble model: Create ensemble model combining multiple models
                # Example: 'PatchTST_Ensemble' -> models.PatchTST.Model_Ensemble
                base_model_name = self.args.model[:-len('_Ensemble')]
                model = importlib.import_module(f'models.{base_model_name}').Model_Ensemble(self.args).float()
            else:
                # For normal model: Create single model
                # Example: 'PatchTST' -> models.PatchTST.Model
                # Dynamically import module
                model = importlib.import_module(f'models.{self.args.model}').Model(self.args).float()

        # =====================================
        # 2. Add normalization layer
        # =====================================
        if self.args.normalization and self.args.online_method != 'OneNet' and self.args.model != 'FSNet_Ensemble':
            # If normalization is needed and not OneNet or FSNet_Ensemble
            # Wrap model with ForecastModel and add normalization processing like RevIN
            model = models.normalization.ForecastModel(
                model,
                num_features=self.args.enc_in,  # Number of input features
                seq_len=self.args.seq_len,      # Sequence length
                process_method=self.args.normalization  # Normalization method (e.g., 'revin')
            )

        # =====================================
        # 3. Load from checkpoint
        # =====================================
        if hasattr(self.args, 'load_path'):
            # freeze=False when updating model (default setting)
            if not self.args.freeze:
                # Recreate optimizer if parameters are not frozen
                self.model_optim = self._select_optimizer(model=model.to(self.device))
            print('Build Model: Load checkpoints from', self.args.load_path)
            # Load model parameters from checkpoint
            model = self.load_checkpoint(self.args.load_path, model)
            if self.model_optim is not None:
                print('Learning rate of model_optim is', self.model_optim.param_groups[0]['lr'])

            if self.args.freeze:
                # Freeze parameters (disable gradient computation)
                model.requires_grad_(False)

        # =====================================
        # 4. Wrap with framework class
        # =====================================
        # Wrap model from outside
        model_params = sum([param.nelement() for param in model.parameters()])
        # framework_class is specified for Proceed and OneNet
        if framework_class is not None:
            if isinstance(framework_class, list):
                # Apply sequentially if there are multiple framework classes
                for cls in framework_class:
                    model = cls(model, self.args)
            else:
                # Apply single framework class
                model = framework_class(model, self.args)

            # Record change in number of parameters
            new_model_params = sum([param.nelement() for param in model.parameters()])
            print(f'Number of Params: {model_params} -> {new_model_params} (+{new_model_params - model_params})')
            self.model_params = model_params

            # Add new parameters to optimizer
            if self.model_optim is not None:
                param_set = set()
                for group in self.model_optim.param_groups:
                    param_set.update(set(group['params']))
                # Search for new parameters not included in existing optimizer
                new_params = list(filter(lambda p: p not in param_set and p.requires_grad, model.parameters()))
                if len(new_params) > 0:
                    # Add new parameter group to optimizer
                    self.model_optim.add_param_group({'params': new_params})

        # =====================================
        # 5. Configure distributed learning
        # =====================================
        if self.args.use_multi_gpu and self.args.use_gpu:
            # Multi-GPU (DataParallel) configuration
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        elif self.args.local_rank != -1:
            # Distributed learning (DistributedDataParallel) configuration
            model = model.to(self.device)
            model = DDP(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=self.args.find_unused_parameters
            )

        # =====================================
        # 6. PyTorch 2.0 compilation (optional)
        # =====================================
        if torch.__version__ >= '2' and self.args.compile:
            print('Compile the model by Pytorch 2.0')
            model = torch.compile(model)

        return model

    def _get_data(self, flag, setting, **kwargs):
        """
        Get dataset and dataloader

        Args:
            flag: Data flag ('train', 'valid', 'test', etc.)
            **kwargs: Additional arguments

        Returns:
            tuple: (dataset, dataloader)
        """
        data_set = get_dataset(self.args, flag, setting, self.device, wrap_class=[], **kwargs)
        data_loader = get_dataloader(data_set, self.args, setting)
        return data_set, data_loader

    def _select_optimizer(self, filter_frozen=True, return_self=True, model=None):
        """
        Select optimizer

        Args:
            filter_frozen: Whether to filter frozen parameters
            return_self: Whether to return own optimizer
            model: Target model (self.model if None)

        Returns:
            torch.optim.Optimizer: Selected optimizer
        """
        print('select_optimizer')

        if return_self and hasattr(self, 'model_optim') and self.model_optim is not None:
            print('reutrn original optimizer')
            return self.model_optim
        else:
            print('create new optimizer')
            # Instantiate new optimizer
            params = self.model.parameters() if model is None else model.parameters()
            if filter_frozen:
                params = filter(lambda p: p.requires_grad, params)  # Only parameters that require gradient computation
            if not hasattr(self.args, 'optim'):
                self.args.optim = 'Adam'  # Default is Adam
            model_optim = getattr(optim, self.args.optim)(params, lr=self.args.learning_rate)
            if return_self:
                self.model_optim = model_optim
            return model_optim

    def _select_criterion(self):
        """
        Select loss function

        Returns:
            nn.Module: Selected loss function (default is MSE)
        """
        criterion = nn.MSELoss()
        return criterion

    def _process_batch(self, batch):
        """
        Preprocess batch data

        Args:
            batch: Input batch data

        Returns:
            Processed batch data
        """
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
        if self.args.model in settings.need_x_y_mark:

            # decoder input
            dec_inp = torch.zeros_like(batch_x[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1)

            # inp = [batch_x, batch_x_mark, dec_inp, batch_y_mark] + batch[4:]
            inp = [batch_x, batch_x_mark, dec_inp, batch_y_mark]
        elif self.args.model in settings.need_x_mark or hasattr(self.args, 'online_method') and self.args.online_method == 'OneNet':
            # Make batch=[batch_x, batch_x_mark]
            # batch = batch[:3] + batch[4:] # Remove batch_y_mark
            # inp = [batch_x] + batch[2:] # Remove batch_y
            inp = [batch_x, batch_x_mark]
        else:
            # Make batch=[batch_x]
            # batch = batch[:2] + batch[4:]
            # inp = [batch_x] + batch[2:]
            inp = [batch_x]
        return inp

    def forward(self, batch):
        """
        Forward propagation

        Args:
            batch: Input batch data

        Returns:
            Model output
        """
        if not self.args.pin_gpu:
            # If GPU pinning is disabled, move tensors to device
            batch = [batch[i].to(self.device) if isinstance(batch[i], torch.Tensor) and i != self.label_position
                     else batch[i] for i in range(len(batch))]
        inp = self._process_batch(batch)  # Preprocess batch
        if self.args.use_amp:
            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(*inp)
        else:
            outputs = self.model(*inp)
        return outputs

    def train_loss(self, criterion, batch, outputs):
        """
        Calculate training loss

        Args:
            criterion: Loss function
            batch: Input batch data
            outputs: Model output

        Returns:
            torch.Tensor: Calculated loss
        """
        batch_y = batch[1]  # Label data
        if not self.args.pin_gpu:
            batch_y = batch_y.to(self.device)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Use first element if tuple
        # print("outputs.shape:",outputs.shape,"batch_y.shape:",batch_y.shape)
        loss = criterion(outputs, batch_y)
        return loss

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

        if self.args.use_amp:
            # Backpropagation with automatic mixed precision
            scaler.scale(loss).backward()
            # 1. Unscale gradients first
            for optim in optimizer:
                scaler.unscale_(optim)
            # 2. Clip unscaled gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # 3. Update parameters
            for optim in optimizer:
                scaler.step(optim)
            scaler.update()
        else:
            # Normal backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            for optim in optimizer:
                optim.step()
        return loss, outputs

    def state_dict(self, destination: typing.OrderedDict[str, torch.Tensor]=None,
                   prefix='', keep_vars=False, local_rank=-1) -> typing.OrderedDict[str, torch.Tensor]:
        """
        Return state dictionary of model and optimizer

        Args:
            destination: Output dictionary
            prefix: Prefix for keys
            keep_vars: Whether to keep variables
            local_rank: Local rank in distributed learning

        Returns:
            OrderedDict: State dictionary of model and optimizer
        """
        if hasattr(self.args, 'save_opt') and self.args.save_opt:
            # If also saving optimizer state
            if destination is None:
                destination = OrderedDict()
                destination._metadata = OrderedDict()
            destination['model'] = self.model.state_dict() if local_rank == -1 else self.model.module.state_dict()
            if hasattr(self.args, 'freeze') and self.args.freeze:
                # Exclude frozen parameters
                for k, v in self.model.named_parameters() if local_rank == -1 else self.model.module.named_parameters():
                    if not v.requires_grad:
                        destination['model'].pop(k)
            destination['model_optim'] = self.model_optim.state_dict()
            return destination
        else:
            # Save only model state
            return self.model.state_dict() if local_rank == -1 else self.model.module.state_dict()

    def load_state_dict(self, state_dict: typing.OrderedDict[str, torch.Tensor], model=None, strict=False) -> nn.Module:
        """
        Copy parameters from state dictionary to model and optimizer

        Args:
            state_dict: Dictionary containing parameters and buffers
            model: Target model (self.model if None)
            strict: Whether to perform strict loading

        Returns:
            nn.Module: Loaded model
        """
        if model is None:
            model = self.model

        if 'model_optim' not in state_dict:
            # If optimizer state is not included
            model.load_state_dict(remove_state_key_prefix(state_dict, model), strict=strict)
        else:
            # If optimizer state is also included
            for k, v in state_dict.items():
                if k == 'model':
                    print('load model state_dict')
                    model.load_state_dict(remove_state_key_prefix(v, model), strict=strict)
                elif hasattr(self, k) and getattr(self, k) is not None:
                    print('load optimizer state_dict')
                    if isinstance(getattr(self, k), optim.Optimizer):
                        # Load optimizer state
                        assert len(getattr(self, k).param_groups) == len(v['param_groups'])
                        try:
                            getattr(self, k).load_state_dict(v)
                        except ValueError:
                            warnings.warn(f'{k} has different state dict from the checkpoint. '
                                          f'Trying to save all states of frozen parameters...')
                            assert k == 'model_optim'
                            self.model_optim = self._select_optimizer(filter_frozen=False, return_self=False, model=model)
                            self.model_optim.load_state_dict(v)
                            self.remove_frozen_param_from_optim(self.model_optim)
                    else:
                        getattr(self, k).load_state_dict(v, strict=strict)

        return model

    def remove_frozen_param_from_optim(self, model_optim):
        """
        Remove frozen parameters from optimizer

        Args:
            model_optim: Target optimizer
        """
        new_index = []
        for i, p in enumerate(model_optim.param_groups[0]['params']):
            if p.requires_grad:
                new_index.append(i)
        model_optim.param_groups[0]['params'] = [model_optim.param_groups[0]['params'][i] for i in new_index]
        delete_ps = []
        for p in model_optim.state:
            if not p.requires_grad:
                delete_ps.append(p)
        for p in delete_ps:
            model_optim.state.pop(p)

    def load_checkpoint(self, load_path=None, model=None, strict=False):
        """
        Load model from checkpoint file

        Args:
            load_path: Path to checkpoint file
            model: Target model
            strict: Whether to perform strict loading

        Returns:
            nn.Module: Loaded model
        """
        # Load to CPU first, then move to appropriate device
        checkpoint = torch.load(load_path, map_location='cpu')
        return self.load_state_dict(checkpoint, model, strict=strict)

    def vali(self):
        """Validation processing (to be implemented in subclass)"""
        pass

    def train(self):
        """Training processing (to be implemented in subclass)"""
        pass

    def test(self):
        """Test processing (to be implemented in subclass)"""
        pass
