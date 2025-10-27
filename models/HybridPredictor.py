import torch
import torch.nn as nn
import numpy as np
import time
from layers.RevIN import RevIN

# Base class (assuming it exists or can be a simple placeholder)
class RLSLinearPredictor:
    def __init__(self, seq_len, pred_len, num_variates, lambda_forget, device, update_batch='single'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_variates = num_variates
        self.lambda_forget = lambda_forget
        self.device = device
        self.update_batch = update_batch
        self.count = 0
        self.rev_mean = []
        self.rev_stdev = []
        print(f"RLSLinearPredictor initialized with seq_len: {seq_len}, pred_len: {pred_len}, num_variates: {num_variates}, lambda_forget: {lambda_forget}, device: {device}, update_batch: {update_batch}")

class RLSLinearPredictorShare(RLSLinearPredictor):
    """
    RLS-based linear autoregressive predictor (weight sharing version)

    Shares weight W [pred_len, seq_len] across variates.
    Updates are performed by aggregating information (X_t, x_{d,t+k,true}) from D variates at each time t+k.
    k_step can take different values for each batch.
    """
    def __init__(self, seq_len: int, pred_len: int, num_variates: int, lambda_forget: float = 0.95, device: str = 'cpu', update_batch='single', revin: bool = False):
        super().__init__(seq_len, pred_len, num_variates, lambda_forget, device, update_batch)

        # self.weights and self.covariance are set in initialize_with_data,
        # but do temporary initialization in constructor (or set to None)
        # Unified as float32 (to match input data)
        self.weights = torch.zeros(self.pred_len, self.seq_len, device=device, dtype=torch.float32)

        # Initialize ReVIN functionality
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(num_features=num_variates, eps=1e-5, affine=False, subtract_last=False)

        update_method = self.choose_update_method()
        print(f"update_method: {update_method}")
        if update_method == 'block':
            self.covariance = torch.eye(self.seq_len, device=device, dtype=torch.float32) * 1000.0
        elif update_method == 'information':
            # Hold information matrix P_inv instead of covariance matrix P
            # P_inv: [seq_len, seq_len] (L x L)
            # If initial value of P_0 is large (e.g., 1000 * I), P_inv_0 is small (1/1000 * I)
            self.inv_covariance = torch.eye(self.seq_len, device=device, dtype=torch.float32) * (1.0 / 1000.0)

            # Also hold P, calculated from inv_covariance
            self.covariance = torch.inverse(self.inv_covariance)

            # Intermediate term I_vec for weight calculation in information matrix format
            # Defined as I_t = lambda * I_{t-1} + V_t^T Y_t in RLS literature.
            # Since W_t = P_t @ I_t, this I_vec also has shape [L, H]
            self.information_vector = torch.zeros(self.seq_len, self.pred_len, device=device, dtype=torch.float32)

    def choose_update_method(self):
        # Computational complexity is O((D*H)^2) for _block, O(L^3) for _information
        if self.seq_len >= self.pred_len*self.num_variates:
            update_method = 'block'
        else:
            update_method = 'information'

        st = time.time()
        self._update_information_pred_len_()
        time_info = time.time() - st
        print(f"update_information_pred_len_ time: {time_info}")

        st = time.time()
        self._update_block_pred_len_()
        time_block = time.time() - st
        print(f"update_block_pred_len_ time: {time_block}")

        if time_info < time_block:
            update_method = 'information'
        else:
            update_method = 'block'


        if update_method == 'block':
            self.update = self._update_block_pred_len
            self.initialize_with_data = self._initialize_with_data_block
        elif update_method == 'information':
            self.update = self._update_information_pred_len
            self.initialize_with_data = self._initialize_with_data_information

        return update_method


    def forward(self, X_t: torch.Tensor, revin_X = None) -> torch.Tensor:
        """
        Prediction phase: predict future Y_t from input X_t.

        Args:
            X_t (torch.Tensor): Input window at time t.
                                 shape: [batch_size, num_variates, seq_len] (B x D x L)

        Returns:
            torch.Tensor: Predicted future Y_t.
                          shape: [batch_size, num_variates, pred_len] (B x D x H)
        """
        # Unify data type (convert to float32)
        # X_t = X_t.to(dtype=torch.float32)
        if self.revin:
            if revin_X is not None:
                revin_X_norm = revin_X.transpose(1, 2)
                revin_X_norm = self.revin_layer(revin_X_norm, mode='norm')
                revin_X_norm = revin_X_norm.transpose(1, 2)
                X_t_norm = X_t
            else:
                X_t_norm = X_t.transpose(1, 2)
                X_t_norm = self.revin_layer(X_t_norm, mode='norm')
                # Restore normalized data to original shape: [B, L, D] -> [B, D, L]
                X_t_norm = X_t_norm.transpose(1, 2)
        else:
            X_t_norm = X_t

        Y_t_pred = torch.einsum('hl,bdl->bdh', self.weights, X_t_norm)

        if self.revin:
            Y_t_pred_norm = Y_t_pred.transpose(1, 2)
            Y_t_pred_norm = self.revin_layer(Y_t_pred_norm, mode='denorm')
            Y_t_pred_norm = Y_t_pred_norm.transpose(1, 2)
        else:
            Y_t_pred_norm = Y_t_pred
        return Y_t_pred_norm



    def _initialize_with_data_information(self, X_init: torch.Tensor, Y_init: torch.Tensor, epsilon: float = 1e-6, revin_X = None):
        """
        Initialize self.weights and self.inv_covariance using small amount of initial training data.
        This is based on batch least squares method (information matrix format).
        """
        # Unify data type
        # X_init = X_init.to(dtype=torch.float32)
        # Y_init = Y_init.to(dtype=torch.float32)
        if self.revin:
            if revin_X is not None:
                revin_X_norm = revin_X.transpose(1, 2)
                revin_X_norm = self.revin_layer(revin_X_norm, mode='norm')
                revin_X_norm = revin_X_norm.transpose(1, 2)
                X_init_norm = X_init
            else:
                X_init_norm = X_init.transpose(1, 2)
                X_init_norm = self.revin_layer(X_init_norm, mode='norm')
                # Restore normalized data to original shape: [B, L, D] -> [B, D, L]
                X_init_norm = X_init_norm.transpose(1, 2)

            Y_init_norm = Y_init.transpose(1, 2)
            Y_init_norm = self.revin_layer(Y_init_norm, mode='norm_only')
            Y_init_norm = Y_init_norm.transpose(1, 2)
        else:
            X_init_norm = X_init
            Y_init_norm = Y_init

        m_eff = X_init_norm.shape[0] * self.num_variates
        X_flat = X_init_norm.reshape(m_eff, self.seq_len) # [M_eff, L]
        Y_flat = Y_init_norm.reshape(m_eff, self.pred_len) # [M_eff, H] (H is output dimension here)

        # 1. Initialize P_inv: (X_flat^T @ X_flat) + epsilon * I
        # XT_X: [L, L]
        XT_X = torch.matmul(X_flat.T, X_flat)
        self.inv_covariance = XT_X + epsilon * torch.eye(self.seq_len, device=self.device, dtype=torch.float32)

        # 2. Initialize information_vector: (X_flat^T @ Y_flat)
        # XT_Y: [L, H]
        self.information_vector = torch.matmul(X_flat.T, Y_flat)

        # 3. Initial calculation of covariance (P): (P_inv)^-1
        try:
            self.covariance = torch.inverse(self.inv_covariance)
        except RuntimeError:
            warnings.warn(f"Initial inverse of information matrix failed. Resetting P_inv to identity.")
            self.inv_covariance = torch.eye(self.seq_len, device=self.device, dtype=torch.float32) # Reset
            self.covariance = torch.inverse(self.inv_covariance)

        # 4. Initial calculation of weights (W): P @ I
        self.weights = torch.matmul(self.information_vector.T, self.covariance) # [H,L] -> Correct for W=IP form.


    def _initialize_with_data_block(self, X_init: torch.Tensor, Y_init: torch.Tensor, epsilon: float = 1e-6, revin_X = None):
        """
        Initialize self.weights and self.covariance using small amount of initial training data.
        This is based on batch least squares method.

        Args:
            X_init (torch.Tensor): Input data for initial learning. shape: [B, D, L]
            Y_init (torch.Tensor): True output data for initial learning. shape: [B, D, H]
            epsilon (float): Regularization term for numerical stability (added to diagonal elements).
        """
        # Unify data type (convert to float32)
        # X_init = X_init.to(dtype=torch.float32)
        # Y_init = Y_init.to(dtype=torch.float32)

        # X_init: [B, D, L]
        # Y_init: [B, D, H]

        if self.revin:
            if revin_X is not None:
                revin_X_norm = revin_X.transpose(1, 2)
                revin_X_norm = self.revin_layer(revin_X_norm, mode='norm')
                revin_X_norm = revin_X_norm.transpose(1, 2)
                X_init_norm = X_init
            else:
                X_init_norm = X_init.transpose(1, 2)
                X_init_norm = self.revin_layer(X_init_norm, mode='norm')
                # Restore normalized data to original shape: [B, L, D] -> [B, D, L]
                X_init_norm = X_init_norm.transpose(1, 2)

            Y_init_norm = Y_init.transpose(1, 2)
            Y_init_norm = self.revin_layer(Y_init_norm, mode='norm_only')
            Y_init_norm = Y_init_norm.transpose(1, 2)
        else:
            X_init_norm = X_init
            Y_init_norm = Y_init

        # 1. Reshape input to format suitable for batch least squares
        # Treat each (batch item, variate) as one sample
        # M_eff = B * D
        m_eff = X_init_norm.shape[0] * self.num_variates
        X_flat = X_init_norm.reshape(m_eff, self.seq_len) # [M_eff, L]
        Y_flat = Y_init_norm.reshape(m_eff, self.pred_len) # [M_eff, H] (H is output dimension here)

        # 2. Calculate X_flat^T @ X_flat
        # XT_X: [L, L]
        XT_X = torch.matmul(X_flat.T, X_flat)

        # 3. Calculate X_flat^T @ Y_flat
        # XT_Y: [L, H]
        XT_Y = torch.matmul(X_flat.T, Y_flat)

        # 4. Calculate (X_flat^T @ X_flat)^-1 and numerical stabilization
        # XT_X_inv: [L, L]
        # Add regularization term to ensure stability of inverse matrix calculation
        XT_X_inv = torch.inverse(XT_X + epsilon * torch.eye(self.seq_len, device=self.device, dtype=torch.float32))

        # 5. Calculate weight matrix W (W_shared_T = XT_X_inv @ XT_Y)
        # W_shared_T: [L, H]
        W_shared_T = torch.matmul(XT_X_inv, XT_Y)

        # 6. Set self.weights (transpose since W_shared is [H, L])
        self.weights = W_shared_T.T # [H, L]

        # 7. Set self.covariance
        # P in RLS is usually considered as inverse of information matrix of input features
        # P_0 = (X^T X)^-1
        self.covariance = XT_X_inv # [L, L]




    def _update_block_pred_len(self, X_t: torch.Tensor, Y_t_true: torch.Tensor, revin_X = None):
        """
        Strict block RLS update logic.
        This method is called when true values Y_t_true for H steps ahead are fully available.

        Args:
            X_t (torch.Tensor): Input window at time t.
                                 shape: [batch_size, num_variates, seq_len] (B x D x L)
            Y_t_true (torch.Tensor): True values at time t.
                                     shape: [batch_size, num_variates, pred_len] (B x D x H)
        """

        batch_size = X_t.shape[0]
        num_variates = self.num_variates
        seq_len = self.seq_len
        pred_len = self.pred_len # H

        m_eff = batch_size * num_variates # Effective number of samples (B*D)


        if self.revin:
            if revin_X is not None:
                revin_X_norm = revin_X.transpose(1, 2)
                revin_X_norm = self.revin_layer(revin_X_norm, mode='norm')
                revin_X_norm = revin_X_norm.transpose(1, 2)
                X_t_norm = X_t
            else:
                X_t_norm = X_t.transpose(1, 2)
                X_t_norm = self.revin_layer(X_t_norm, mode='norm')
                # Restore normalized data to original shape: [B, L, D] -> [B, D, L]
                X_t_norm = X_t_norm.transpose(1, 2)

            Y_t_true_norm = Y_t_true.transpose(1, 2)
            Y_t_true_norm = self.revin_layer(Y_t_true_norm, mode='norm_only')
            Y_t_true_norm = Y_t_true_norm.transpose(1, 2)
        else:
            X_t_norm = X_t
            Y_t_true_norm = Y_t_true

        # V_combined: [M_eff, L] (input matrix)
        V_combined = X_t_norm.reshape(m_eff, seq_len)
        # Y_combined: [M_eff, H] (true output matrix)
        Y_combined = Y_t_true_norm.reshape(m_eff, pred_len)

        # --- 1. Calculate intermediate term Q: [M_eff, M_eff] ---
        # Q = lambda * I_M_eff + V_combined @ P_old @ V_combined.T
        # P_old is self.covariance [L, L]
        # V_combined @ P_old: [M_eff, L] @ [L, L] -> [M_eff, L]
        # (V_combined @ P_old) @ V_combined.T: [M_eff, L] @ [L, M_eff] -> [M_eff, M_eff]
        Q_matrix = self.lambda_forget * torch.eye(m_eff, device=self.device, dtype=torch.float32) + \
                   torch.matmul(torch.matmul(V_combined, self.covariance), V_combined.T)

        # --- 2. Calculate gain matrix K_Block: [L, M_eff] ---
        # K_Block = P_old @ V_combined.T @ Q_matrix^-1
        # P_old @ V_combined.T: [L, L] @ [L, M_eff] -> [L, M_eff]
        # Q_matrix^-1: [M_eff, M_eff]

        # Add small regularization before inverse matrix calculation for numerical stability
        epsilon_inverse_q = 1e-9 # For Q_matrix inverse calculation
        try:
            Q_inv = torch.inverse(Q_matrix + epsilon_inverse_q * torch.eye(m_eff, device=self.device, dtype=torch.float32))
        except RuntimeError:
            warnings.warn(f"Inverse of Q_matrix failed in Block RLS at step {self.count}. Returning without update.")
            # If failed, skip update for this step and model maintains previous state
            return

        K_Block = torch.matmul(torch.matmul(self.covariance, V_combined.T), Q_inv)

        # --- 3. Update weights (self.weights) ---
        # W_new = W_old + K_Block @ (Y_combined - V_combined @ W_old.T).T
        # W_old is self.weights [H, L]
        # (Y_combined - V_combined @ W_old.T): [M_eff, H] - ([M_eff, L] @ [L, H]) = [M_eff, H]
        # Transpose of this error matrix: [H, M_eff]

        # K_Block @ (transpose of error matrix): [L, M_eff] @ [H, M_eff].T is invalid.
        # It should be K_Block @ (Y_combined - V_combined @ self.weights.T).T
        # or, Y_combined is [M_eff, H], V_combined @ self.weights.T is [M_eff, L] @ [L, H] = [M_eff, H]
        # Error = Y_combined - (V_combined @ self.weights.T) -> [M_eff, H]
        # Update_term = K_Block @ Error.T -> [L, M_eff] @ [H, M_eff].T => [L, M_eff] @ [M_eff, H] => [L, H]
        # self.weights.T += Update_term
        # self.weights += Update_term.T


        error_matrix = Y_combined - torch.matmul(V_combined, self.weights.T) # [M_eff, H]
        update_term_W = torch.matmul(error_matrix.T, K_Block.T) # [H, L]

        self.weights = self.weights + update_term_W # [H, L]

        # --- 4. Update covariance matrix (self.covariance) ---
        # P_new = (P_old - K_Block @ V_combined @ P_old) / lambda_forget
        # K_Block @ V_combined: [L, M_eff] @ [M_eff, L] -> [L, L]
        update_term_P = torch.matmul(K_Block, V_combined) # [L, L]
        self.covariance = (self.covariance - torch.matmul(update_term_P, self.covariance)) / self.lambda_forget

        # Force symmetry of P (correct asymmetry due to floating point errors)
        self.covariance = (self.covariance + self.covariance.T) / 2

        # # Check positive definiteness of P and reset if there's a problem
        # epsilon_chol_reg = 1e-6
        # try:
        #     _ = torch.linalg.cholesky(self.covariance + epsilon_chol_reg * torch.eye(seq_len, device=self.device, dtype=torch.float32))
        # except RuntimeError:
        #     warnings.warn(f"ERROR: P_matrix became non-positive definite in Block RLS at step {self.count}. Resetting P.")
        #     self.covariance = torch.eye(seq_len, device=self.device, dtype=torch.float32) * 1000.0


    def _update_information_pred_len(self, X_t: torch.Tensor, Y_t_true: torch.Tensor, revin_X = None):
        """
        Correct batch RLS (information matrix format) update logic.
        This method is called when true values Y_t_true for H steps ahead are fully available.

        Args:
            X_t (torch.Tensor): Input window at time t.
                                 shape: [batch_size, num_variates, seq_len] (B x D x L)
            Y_t_true (torch.Tensor): True values at time t.
                                     shape: [batch_size, num_variates, pred_len] (B x D x H)
        """

        batch_size = X_t.shape[0]
        num_variates = self.num_variates
        seq_len = self.seq_len
        pred_len = self.pred_len # H

        m_eff = batch_size * num_variates

        if self.revin:
            if revin_X is not None:
                revin_X_norm = revin_X.transpose(1, 2)
                revin_X_norm = self.revin_layer(revin_X_norm, mode='norm')
                revin_X_norm = revin_X_norm.transpose(1, 2)
                X_t_norm = X_t
            else:
                X_t_norm = X_t.transpose(1, 2)
                X_t_norm = self.revin_layer(X_t_norm, mode='norm')
                # Restore normalized data to original shape: [B, L, D] -> [B, D, L]
                X_t_norm = X_t_norm.transpose(1, 2)

            Y_t_true_norm = Y_t_true.transpose(1, 2)
            Y_t_true_norm = self.revin_layer(Y_t_true_norm, mode='norm_only')
            Y_t_true_norm = Y_t_true_norm.transpose(1, 2)
        else:
            X_t_norm = X_t
            Y_t_true_norm = Y_t_true

        # V_combined: [M_eff, L] (input matrix for RLS update)
        V_combined = X_t_norm.reshape(m_eff, seq_len)
        # Y_combined: [M_eff, H] (true output matrix for RLS update)
        Y_combined = Y_t_true_norm.reshape(m_eff, pred_len)

        # --- 1. Update information matrix (inv_covariance) ---
        # P_new_inv = lambda * P_old_inv + V_combined.T @ V_combined
        # V_combined.T @ V_combined is [L, M_eff] @ [M_eff, L] -> [L, L]
        self.inv_covariance = self.lambda_forget * self.inv_covariance + torch.matmul(V_combined.T, V_combined)

        # --- 2. Update information vector (information_vector) ---
        # I_new = lambda * I_old + V_combined.T @ Y_combined
        # V_combined.T @ Y_combined is [L, M_eff] @ [M_eff, H] -> [L, H]
        self.information_vector = self.lambda_forget * self.information_vector + torch.matmul(V_combined.T, Y_combined)

        # --- 3. Calculate covariance matrix (P) ---
        # P = P_inv^-1
        epsilon_inv = 1e-9 # Appropriate small value
        try:
            self.covariance = torch.inverse(self.inv_covariance + epsilon_inv * torch.eye(seq_len, device=self.device, dtype=torch.float32))
        except RuntimeError:
            warnings.warn(f"Inverse of information matrix failed in Info Form RLS at step {self.count if hasattr(self, 'count') else 'N/A'}. Resetting P_inv.")
            self.inv_covariance = torch.eye(seq_len, device=self.device, dtype=torch.float32) / 1000.0
            self.covariance = torch.inverse(self.inv_covariance)

        # Force symmetry of P
        self.covariance = (self.covariance + self.covariance.T) / 2

        # --- 4. Update weights (self.weights) ---
        # W_new = I_new^T @ P_new
        self.weights = torch.matmul(self.information_vector.T, self.covariance)

        self.count += 1



    # Method to measure computation time
    def _update_information_pred_len_(self):

        batch_size = 1 if self.update_batch == 'single' else self.pred_len
        num_variates = self.num_variates
        seq_len = self.seq_len
        pred_len = self.pred_len # H
        lambda_forget = 1
        m_eff = batch_size * num_variates

        weights = torch.zeros(pred_len, seq_len)
        inv_covariance = torch.eye(seq_len) * (1.0 / 1000.0)
        covariance = torch.inverse(inv_covariance)
        information_vector = torch.zeros(seq_len, pred_len)


        V_combined = torch.zeros(m_eff, seq_len)
        Y_combined = torch.zeros(m_eff, pred_len)

        inv_covariance = lambda_forget * inv_covariance + torch.matmul(V_combined.T, V_combined)

        information_vector = lambda_forget * information_vector + torch.matmul(V_combined.T, Y_combined)

        epsilon_inv = 1e-9 # Appropriate small value
        covariance = torch.inverse(inv_covariance + epsilon_inv * torch.eye(seq_len))

        covariance = (covariance + covariance.T) / 2

        weights = torch.matmul(information_vector.T, covariance)


    def _update_block_pred_len_(self):

        batch_size = 1 if self.update_batch == 'single' else self.pred_len
        num_variates = self.num_variates
        seq_len = self.seq_len
        pred_len = self.pred_len # H
        lambda_forget = 1
        m_eff = batch_size * num_variates

        weights = torch.zeros(pred_len, seq_len)
        covariance = torch.eye(seq_len) * 1000.0

        V_combined = torch.zeros(m_eff, seq_len)
        Y_combined = torch.zeros(m_eff, pred_len)

        Q_matrix = lambda_forget * torch.eye(m_eff) + torch.matmul(torch.matmul(V_combined, covariance), V_combined.T)

        epsilon_inverse_q = 1e-9
        Q_inv = torch.inverse(Q_matrix + epsilon_inverse_q * torch.eye(m_eff))
        K_Block = torch.matmul(torch.matmul(covariance, V_combined.T), Q_inv)

        error_matrix = Y_combined - torch.matmul(V_combined, weights.T) # [M_eff, H]
        update_term_W = torch.matmul(error_matrix.T, K_Block.T) # [H, L]

        weights = weights + update_term_W # [H, L]

        update_term_P = torch.matmul(K_Block, V_combined) # [L, L]
        covariance = (covariance - torch.matmul(update_term_P, covariance)) / lambda_forget
        covariance = (covariance + covariance.T) / 2


