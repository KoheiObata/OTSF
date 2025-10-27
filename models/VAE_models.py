import torch.nn as nn
import torch


class ConvEncoder(nn.Module):
    """
    Generic ConvEncoder - supports arbitrary window size and feature count
    input: [batch, window, feature]
    """
    def __init__(self, output_dim, window_size=100, feature_dim=1):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        self.window_size = window_size
        self.feature_dim = feature_dim

        # More efficient architecture to handle large feature_dim
        if feature_dim > 500:
            # Efficient architecture for Traffic dataset (feature_dim > 500)
            self.conv1 = nn.Conv2d(1, 8, kernel_size=(9, min(50, feature_dim)), stride=(2, 2), padding=(0, 0))
            self.bn1 = nn.BatchNorm2d(8)

            self.conv2 = nn.Conv2d(8, 16, kernel_size=(7, min(50, feature_dim//2)), stride=(2, 2), padding=(0, 0))
            self.bn2 = nn.BatchNorm2d(16)

            self.conv3 = nn.Conv2d(16, 32, kernel_size=(5, min(50, feature_dim//4)), stride=(2, 1), padding=(0, 0))
            self.bn3 = nn.BatchNorm2d(32)

            self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0))
            self.bn4 = nn.BatchNorm2d(64)
        elif feature_dim > 100:
            # Ultra-lightweight architecture for ECL dataset (100 < feature_dim <= 500)
            # Use only 1D convolution to maximize CUDA compatibility
            self.conv1 = nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=0)
            self.bn1 = nn.BatchNorm1d(16)

            self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=0)
            self.bn2 = nn.BatchNorm1d(32)

            self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=0)
            self.bn3 = nn.BatchNorm1d(64)

            self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=0)
            self.bn4 = nn.BatchNorm1d(128)
        else:
            # Normal architecture
            self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, min(3, feature_dim)), stride=(2, 1), padding=(0, 0))
            self.bn1 = nn.BatchNorm2d(16)

            self.conv2 = nn.Conv2d(16, 32, kernel_size=(7, min(3, feature_dim)), stride=(2, 1), padding=(0, 0))
            self.bn2 = nn.BatchNorm2d(32)

            self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, min(3, feature_dim)), stride=(2, 1), padding=(0, 0))
            self.bn3 = nn.BatchNorm2d(64)

            self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0))
            self.bn4 = nn.BatchNorm2d(128)

        # Activation function
        self.act = nn.ReLU(inplace=True)

        # Calculate output size dynamically
        self._calculate_output_size()

        # Final output layer
        if feature_dim > 500:
            self.conv_z = nn.Conv2d(64, output_dim, kernel_size=(self.final_h, self.final_w), stride=(1, 1))
        elif feature_dim > 100:
            self.conv_z = nn.Conv1d(128, output_dim, kernel_size=self.final_h, stride=1)
        else:
            self.conv_z = nn.Conv2d(128, output_dim, kernel_size=(self.final_h, self.final_w), stride=(1, 1))

    def _calculate_output_size(self):
        """Calculate actual output size after convolution using dummy data"""
        with torch.no_grad():
            # Create dummy data: [1, 1, window_size, feature_dim]
            dummy_input = torch.zeros(1, 1, self.window_size, self.feature_dim)

            # Create temporary convolution layers (same configuration as actual ConvEncoder)
            if self.feature_dim > 500:
                # Efficient architecture for Traffic dataset (feature_dim > 500)
                temp_conv1 = nn.Conv2d(1, 8, kernel_size=(9, min(50, self.feature_dim)), stride=(2, 2), padding=(0, 0))
                temp_conv2 = nn.Conv2d(8, 16, kernel_size=(7, min(50, self.feature_dim//2)), stride=(2, 2), padding=(0, 0))
                temp_conv3 = nn.Conv2d(16, 32, kernel_size=(5, min(50, self.feature_dim//4)), stride=(2, 1), padding=(0, 0))
                temp_conv4 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0))
            elif self.feature_dim > 100:
                # 1D convolution calculation for ECL dataset
                # Create dummy data: [1, 1, window_size * feature_dim]
                dummy_input = torch.zeros(1, 1, self.window_size * self.feature_dim)

                # Create temporary 1D convolution layers
                temp_conv1 = nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=0)
                temp_conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=0)
                temp_conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=0)
                temp_conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=0)

                # Apply 1D convolution layers
                x = torch.relu(temp_conv1(dummy_input))
                x = torch.relu(temp_conv2(x))
                x = torch.relu(temp_conv3(x))
                x = torch.relu(temp_conv4(x))

                # Get final size (1D)
                self.final_h = x.size(2)
                self.final_w = 1  # Width is 1 for 1D convolution
                return
            else:
                # Normal architecture
                temp_conv1 = nn.Conv2d(1, 16, kernel_size=(9, min(3, self.feature_dim)), stride=(2, 1), padding=(0, 0))
                temp_conv2 = nn.Conv2d(16, 32, kernel_size=(7, min(3, self.feature_dim)), stride=(2, 1), padding=(0, 0))
                temp_conv3 = nn.Conv2d(32, 64, kernel_size=(5, min(3, self.feature_dim)), stride=(2, 1), padding=(0, 0))
                temp_conv4 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0))

            # Apply convolution layers
            x = torch.relu(temp_conv1(dummy_input))
            x = torch.relu(temp_conv2(x))
            x = torch.relu(temp_conv3(x))
            x = torch.relu(temp_conv4(x))

            # Get final size
            self.final_h = x.size(2)
            self.final_w = x.size(3)

    def forward(self, x):
        # Get input size dynamically
        batch_size = x.size(0)

        # 1D convolution processing for ECL dataset
        if self.feature_dim > 100:
            # Unify input format: [batch, window, feature] -> [batch, 1, window*feature]
            if len(x.shape) == 3:
                x = x.view(batch_size, 1, x.size(1) * x.size(2))
            elif len(x.shape) == 4:
                x = x.view(batch_size, 1, x.size(1) * x.size(2))

            # Debug: Check input for NaN
            if torch.isnan(x).any():
                print(f"NaN in ConvEncoder input: {x.shape}")

            # Apply 1D convolution layers
            x = self.act(self.bn1(self.conv1(x)))
            if torch.isnan(x).any():
                print(f"NaN after conv1: {x.shape}")

            x = self.act(self.bn2(self.conv2(x)))
            if torch.isnan(x).any():
                print(f"NaN after conv2: {x.shape}")

            x = self.act(self.bn3(self.conv3(x)))
            if torch.isnan(x).any():
                print(f"NaN after conv3: {x.shape}")

            x = self.act(self.bn4(self.conv4(x)))
            if torch.isnan(x).any():
                print(f"NaN after conv4: {x.shape}")

            # Final output (for 1D convolution)
            z = self.conv_z(x).view(batch_size, self.output_dim)
            if torch.isnan(z).any():
                print(f"NaN in ConvEncoder output: {z.shape}")

            return z
        else:
            # Normal 2D convolution processing
            # Unify input format: [batch, window, feature] -> [batch, 1, window, feature]
            if len(x.shape) == 3:
                x = x.view(batch_size, 1, x.size(1), x.size(2))
            elif len(x.shape) == 4:
                if x.size(1) == 1:
                    pass  # Already in correct format
                else:
                    x = x.view(batch_size, 1, x.size(1), x.size(2))

            # Debug: Check input for NaN
            if torch.isnan(x).any():
                print(f"NaN in ConvEncoder input: {x.shape}")

            # Apply convolution layers
            x = self.act(self.bn1(self.conv1(x)))
            if torch.isnan(x).any():
                print(f"NaN after conv1: {x.shape}")

            x = self.act(self.bn2(self.conv2(x)))
            if torch.isnan(x).any():
                print(f"NaN after conv2: {x.shape}")

            x = self.act(self.bn3(self.conv3(x)))
            if torch.isnan(x).any():
                print(f"NaN after conv3: {x.shape}")

            x = self.act(self.bn4(self.conv4(x)))
            if torch.isnan(x).any():
                print(f"NaN after conv4: {x.shape}")

            # Final output
            z = self.conv_z(x).view(batch_size, self.output_dim)
            if torch.isnan(z).any():
                print(f"NaN in ConvEncoder output: {z.shape}")

            return z


class ConvDecoder(nn.Module):
    """
    Generic ConvDecoder - supports arbitrary window size and feature count
    output: [batch, window, feature]
    """
    def __init__(self, input_dim, window_size=100, feature_dim=1):
        super(ConvDecoder, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.feature_dim = feature_dim

        # Determine intermediate size with same calculation as encoder
        self._calculate_intermediate_size()

        # More efficient architecture to handle large feature_dim
        if feature_dim > 500:
            # Efficient architecture for Traffic dataset (feature_dim > 500)
            self.fc = nn.Linear(input_dim, 64 * self.intermediate_h * self.intermediate_w)

            self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0), output_padding=(1, 0))
            self.bn1 = nn.BatchNorm2d(32)

            self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=(5, min(50, feature_dim//4)), stride=(2, 1), padding=(0, 0), output_padding=(1, 0))
            self.bn2 = nn.BatchNorm2d(16)

            self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=(7, min(50, feature_dim//2)), stride=(2, 2), padding=(0, 0), output_padding=(1, 1))
            self.bn3 = nn.BatchNorm2d(8)

            self.deconv4 = nn.ConvTranspose2d(8, 1, kernel_size=(9, min(50, feature_dim)), stride=(2, 2), padding=(0, 0), output_padding=(1, 1))
        elif feature_dim > 100:
            # Intermediate architecture for ECL dataset (100 < feature_dim <= 500)
            # Use safer deconvolution layers
            self.fc = nn.Linear(input_dim, 128 * self.intermediate_h * self.intermediate_w)

            self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0), output_padding=(1, 0))
            self.bn1 = nn.BatchNorm2d(64)

            self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(5, 1), stride=(2, 1), padding=(0, 0), output_padding=(1, 0))
            self.bn2 = nn.BatchNorm2d(32)

            self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=(7, 1), stride=(2, 1), padding=(0, 0), output_padding=(1, 0))
            self.bn3 = nn.BatchNorm2d(16)

            # Safer final layer (smaller kernel size)
            self.deconv4 = nn.ConvTranspose2d(16, 1, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0), output_padding=(1, 0))
        else:
            # Normal architecture
            self.fc = nn.Linear(input_dim, 128 * self.intermediate_h * self.intermediate_w)

            self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0), output_padding=(1, 0))
            self.bn1 = nn.BatchNorm2d(64)

            self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(5, min(3, feature_dim)), stride=(2, 1), padding=(0, 0), output_padding=(1, 0))
            self.bn2 = nn.BatchNorm2d(32)

            self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=(7, min(3, feature_dim)), stride=(2, 1), padding=(0, 0), output_padding=(1, 0))
            self.bn3 = nn.BatchNorm2d(16)

            self.deconv4 = nn.ConvTranspose2d(16, 1, kernel_size=(9, min(3, feature_dim)), stride=(2, 1), padding=(0, 0), output_padding=(1, 0))

        # Activation function
        self.act = nn.ReLU(inplace=True)

        # Adaptive pooling for size adjustment
        self.adaptive_pool = nn.AdaptiveAvgPool2d((window_size, feature_dim))

    def _calculate_intermediate_size(self):
        """Calculate actual intermediate size using dummy data"""
        with torch.no_grad():
            # Create dummy data: [1, 1, window_size, feature_dim]
            dummy_input = torch.zeros(1, 1, self.window_size, self.feature_dim)

            # Temporarily create same convolution layers as encoder
            if self.feature_dim > 500:
                temp_conv1 = nn.Conv2d(1, 8, kernel_size=(9, min(50, self.feature_dim)), stride=(2, 2), padding=(0, 0))
                temp_conv2 = nn.Conv2d(8, 16, kernel_size=(7, min(50, self.feature_dim//2)), stride=(2, 2), padding=(0, 0))
                temp_conv3 = nn.Conv2d(16, 32, kernel_size=(5, min(50, self.feature_dim//4)), stride=(2, 1), padding=(0, 0))
                temp_conv4 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0))
            elif self.feature_dim > 100:
                # Use more conservative kernel size and stride
                temp_conv1 = nn.Conv2d(1, 16, kernel_size=(9, min(8, self.feature_dim//40)), stride=(2, 1), padding=(0, 0))
                temp_conv2 = nn.Conv2d(16, 32, kernel_size=(7, min(8, self.feature_dim//40)), stride=(2, 1), padding=(0, 0))
                temp_conv3 = nn.Conv2d(32, 64, kernel_size=(5, min(8, self.feature_dim//40)), stride=(2, 1), padding=(0, 0))
                temp_conv4 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0))
            else:
                temp_conv1 = nn.Conv2d(1, 16, kernel_size=(9, min(3, self.feature_dim)), stride=(2, 1), padding=(0, 0))
                temp_conv2 = nn.Conv2d(16, 32, kernel_size=(7, min(3, self.feature_dim)), stride=(2, 1), padding=(0, 0))
                temp_conv3 = nn.Conv2d(32, 64, kernel_size=(5, min(3, self.feature_dim)), stride=(2, 1), padding=(0, 0))
                temp_conv4 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0))

            # Apply convolution layers
            x = torch.relu(temp_conv1(dummy_input))
            x = torch.relu(temp_conv2(x))
            x = torch.relu(temp_conv3(x))
            x = torch.relu(temp_conv4(x))

            # Get intermediate size
            self.intermediate_h = x.size(2)
            self.intermediate_w = x.size(3)

    def forward(self, z):
        batch_size = z.size(0)

        # Debug: Check input for NaN
        if torch.isnan(z).any():
            print(f"NaN in ConvDecoder input: {z.shape}")

        # Expand features with linear layer
        x = self.fc(z)
        if torch.isnan(x).any():
            print(f"NaN after fc: {x.shape}")

        # Adjust number of channels according to feature_dim
        if self.feature_dim > 500:
            x = x.view(batch_size, 64, self.intermediate_h, self.intermediate_w)
        elif self.feature_dim > 100:
            x = x.view(batch_size, 128, self.intermediate_h, self.intermediate_w)
        else:
            x = x.view(batch_size, 128, self.intermediate_h, self.intermediate_w)

        if torch.isnan(x).any():
            print(f"NaN after view: {x.shape}")

        # Apply deconvolution layers
        x = self.act(self.bn1(self.deconv1(x)))
        if torch.isnan(x).any():
            print(f"NaN after deconv1: {x.shape}")

        x = self.act(self.bn2(self.deconv2(x)))
        if torch.isnan(x).any():
            print(f"NaN after deconv2: {x.shape}")

        x = self.act(self.bn3(self.deconv3(x)))
        if torch.isnan(x).any():
            print(f"NaN after deconv3: {x.shape}")

        # Execute safer deconvolution
        try:
            x = self.deconv4(x)
            if torch.isnan(x).any():
                print(f"NaN after deconv4: {x.shape}")
        except Exception as e:
            print(f"Error in deconv4: {e}")
            # Fallback: Retry with smaller kernel size
            try:
                fallback_deconv = nn.ConvTranspose2d(16, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                fallback_deconv = fallback_deconv.to(x.device)
                x = fallback_deconv(x)
                print("Used fallback deconvolution")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                # Final fallback: Return zero tensor
                x = torch.zeros(x.size(0), 1, self.window_size, self.feature_dim, device=x.device)

        # Adjust to exact size with adaptive pooling
        try:
            x = self.adaptive_pool(x)
            if torch.isnan(x).any():
                print(f"NaN after adaptive_pool: {x.shape}")
        except Exception as e:
            print(f"Error in adaptive_pool: {e}")
            # Fallback: Resize
            x = torch.nn.functional.interpolate(x, size=(self.window_size, self.feature_dim), mode='bilinear', align_corners=False)

        return x


# Keep existing encoders/decoders as is
class ConvEncoder_60(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_60, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 3), stride=(2,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 3), stride=(2,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), stride=(3,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (2, 1), stride=(1,1))
        self.bn4 = nn.BatchNorm2d(128)

        self.conv_z = nn.Conv2d(128, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # print("x.shape before view:", x.shape)
        x = x.view(-1, 1, 60, 7)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)


        return z



class ConvDecoder_60(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_60, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (2, 7), stride=(1, 1), padding=0)  # 2 x 9
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (4, 1), stride=(6,1), padding=(1,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (4, 1), stride=(2,1), padding=(1,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (4, 1), stride=(2,1), padding=(1,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 1, (4, 1), stride=(2,1), padding=(3,0))  # 64 x 9

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        mu_img = self.conv5(x)
        return mu_img

class ConvEncoder_336(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_336, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 3), stride=(2,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 3), stride=(2,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), stride=(2,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (2, 1), stride=(2,1))
        self.bn4 = nn.BatchNorm2d(128)

        self.conv_z = nn.Conv2d(128, output_dim, (19,1))  # Output: (batch, output_dim, 1, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        x = x.view(-1, 1, 336, 7)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))

        z = self.conv_z(x).view(x.size(0), self.output_dim)


        return z
class ConvDecoder_336(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_336, self).__init__()
        # Initial expansion (height 1→4, width 1→7)
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (4,7), stride=1, padding=0)

        # Height expansion (4→16→64→256→512)
        self.conv2 = nn.ConvTranspose2d(512, 256, (4,1), stride=(4,1), padding=0)
        self.conv3 = nn.ConvTranspose2d(256, 128, (4,1), stride=(2,1), padding=0)
        self.conv4 = nn.ConvTranspose2d(128, 64, (4,1), stride=(5,1), padding=0)
        self.conv5 = nn.ConvTranspose2d(64, 1, (2,1), stride=(2,1), padding=(1,0))

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64)
        ])
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)

        # Layer 1: 1 → 4
        x = self.act(self.bn_layers[0](self.conv1(x)))  # [32,512,4,7]

        # Layer 2: 4 → 16
        x = self.act(self.bn_layers[1](self.conv2(x)))  # [32,256,16,7]

        # Layer 3: 16 → 64
        x = self.act(self.bn_layers[2](self.conv3(x)))  # [32,128,64,7]

        # Layer 4: 64 → 256
        x = self.act(self.bn_layers[3](self.conv4(x)))  # [32,64,256,7]

        # Layer 5: 256 → 512
        x = self.conv5(x)  # [32,1,512,7]

        return x


class ConvEncoder_512(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_512, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 3), stride=(2,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 3), stride=(2,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), stride=(2,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (2, 1), stride=(2,1))
        self.bn4 = nn.BatchNorm2d(128)

        self.conv_z = nn.Conv2d(128, output_dim, (30,1))  # Output: (batch, output_dim, 1, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        x = x.view(-1, 1, 512, 7)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))

        x = self.act(self.bn3(self.conv3(x)))

        x = self.act(self.bn4(self.conv4(x)))

        z = self.conv_z(x).view(x.size(0), self.output_dim)


        return z
class ConvDecoder_512(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_512, self).__init__()
        # Initial expansion (height 1→4, width 1→7)
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (4,7), stride=1, padding=0)

        # Height expansion (4→16→64→256→512)
        self.conv2 = nn.ConvTranspose2d(512, 256, (4,1), stride=(4,1), padding=0)
        self.conv3 = nn.ConvTranspose2d(256, 128, (4,1), stride=(4,1), padding=0)
        self.conv4 = nn.ConvTranspose2d(128, 64, (4,1), stride=(4,1), padding=0)
        self.conv5 = nn.ConvTranspose2d(64, 1, (2,1), stride=(2,1), padding=(0,0))

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64)
        ])
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)

        # Layer 1: 1 → 4
        x = self.act(self.bn_layers[0](self.conv1(x)))  # [32,512,4,7]

        # Layer 2: 4 → 16
        x = self.act(self.bn_layers[1](self.conv2(x)))  # [32,256,16,7]

        # Layer 3: 16 → 64
        x = self.act(self.bn_layers[2](self.conv3(x)))  # [32,128,64,7]

        # Layer 4: 64 → 256
        x = self.act(self.bn_layers[3](self.conv4(x)))  # [32,64,256,7]

        # Layer 5: 256 → 512
        x = self.conv5(x)  # [32,1,512,7]

        return x





class ConvEncoder_Exchange(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_Exchange, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 3), stride=(2,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 3), stride=(2,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), stride=(3,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (2, 2), stride=(1,1))
        self.bn4 = nn.BatchNorm2d(128)

        self.conv_z = nn.Conv2d(128, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 512, 8)

        x = self.act(self.bn1(self.conv1(x)))

        x = self.act(self.bn2(self.conv2(x)))

        x = self.act(self.bn3(self.conv3(x)))

        x = self.act(self.bn4(self.conv4(x)))

        z = self.conv_z(x).view(x.size(0), self.output_dim)


        return z



class ConvDecoder_Exchange(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_Exchange, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (2, 8), stride=(1, 1), padding=0)  # 2 x 9
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (4, 1), stride=(6,1), padding=(1,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (4, 1), stride=(2,1), padding=(1,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (4, 1), stride=(2,1), padding=(1,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 1, (4, 1), stride=(2,1), padding=(3,0))  # 64 x 9

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        mu_img = self.conv5(x)
        return mu_img

################# IDAA ####################

class vae_idaa(nn.Module):
    def __init__(self, z_dim, dataset):
        super(vae_idaa,self).__init__()

        if dataset == 'ucihar':
            self.encoder = ConvEncoder(z_dim)
            self.decoder = ConvDecoder(z_dim)
        elif dataset == 'shar':
            self.encoder = ConvEncoder_shar(z_dim)
            self.decoder = ConvDecoder_shar(z_dim)
        elif dataset == 'usc' or dataset == 'hhar':
            self.encoder = ConvEncoder_usc(z_dim)
            self.decoder = ConvDecoder_usc(z_dim)
        elif dataset == 'ieee_small' or dataset == 'ieee_big' or dataset == 'dalia':
            self.encoder = ConvEncoder_ieeesmall(z_dim)
            self.decoder = ConvDecoder_ieeesmall(z_dim)
        elif dataset == 'ecg':
            self.encoder = ConvEncoder_ecg(z_dim)
            self.decoder = ConvDecoder_ecg(z_dim)

        self.zdim = z_dim
        self.bn = nn.BatchNorm2d(1)
        self.fc11 = nn.Linear(z_dim, z_dim)
        self.fc12 = nn.Linear(z_dim, z_dim)
        self.fc21 = nn.Linear(z_dim, z_dim)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.zdim)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.zdim)
        h3 = self.decoder(z)
        return h3

    def forward(self, x, decode=False):
        if decode:
            z_projected = self.fc21(x)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            gx = torch.squeeze(gx)
            return gx
        else:
            _, mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_projected = self.fc21(z)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            gx = torch.squeeze(gx,1)
        return z, gx, mu, logvar


################# IDAA ####################

class view_learner(nn.Module):
    def __init__(self, dataset):
        super(view_learner,self).__init__()

        self.conv = nn.Conv2d(1, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: input tensor of shape (batch_size, channels, height, width)
        out = self.conv(x)
        out = self.relu(out)
        return out

################# SHAR ####################


class ConvEncoder_shar(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_shar, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 2), stride=(2,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 2), stride=(2,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 1), stride=(3,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (5, 1), stride=(2,1))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 512, (3, 1), stride=(1,1))
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 151, 3)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z


class ConvDecoder_shar(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_shar, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (3, 3), stride=(1, 1), padding=0)  # 2 x 9
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (3, 1), stride=(4,1), padding=(1,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (3, 1), stride=(2,1), padding=(1,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (5, 1), stride=(2,1), padding=(1,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, (7, 1), stride=(2,1), padding=(1,0))  # 64 x 9
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, (9, 1), stride=(2,1), padding=(1,0)) # 128 x 9

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        mu_img = self.conv_final(x)
        return mu_img


#################################################


################# ECL ####################

class ConvEncoder_ECL(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_ECL, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 2), stride=(2,5))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 3), stride=(2,5))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), stride=(2,5))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (2, 2), stride=(2,3))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_z = nn.Conv2d(128, output_dim, (19,1))

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 336, 321)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))

        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z



class ConvDecoder_ECL(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_ECL, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (4, 321), stride=(1, 1), padding=0)  # 2 x 9
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (4, 1), stride=(4,1), padding=(0,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (4, 1), stride=(2,1), padding=(0,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (4, 1), stride=(5,1), padding=(0,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 1, (2, 1), stride=(2,1), padding=(1,0))  # 64 x 9
         # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        mu_img = self.conv5(x)
        return mu_img
################# WTH ####################

class ConvEncoder_WTH(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_WTH, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 2), stride=(3,1))

        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 3), stride=(2,2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), stride=(2,2))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (2, 2), stride=(2,1))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_z = nn.Conv2d(128, output_dim, (19,1))

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 336, 12)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))

        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z



class ConvDecoder_WTH(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_WTH, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (4, 12), stride=(1, 1), padding=0)  # 2 x 9
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (4, 1), stride=(4,1), padding=(0,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (4, 1), stride=(2,1), padding=(0,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (4, 1), stride=(5,1), padding=(0,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 1, (2, 1), stride=(2,1), padding=(1,0))  # 64 x 9
         # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        mu_img = self.conv5(x)
        return mu_img



class ConvEncoder_Weather(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_Weather, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 2), stride=(3,1))

        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 3), stride=(2,2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), stride=(2,2))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (2, 4), stride=(2,1))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_z = nn.Conv2d(128, output_dim, (19,1))

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        x = x.view(-1, 1, 512, 21)
        x = self.act(self.bn1(self.conv1(x)))

        x = self.act(self.bn2(self.conv2(x)))

        x = self.act(self.bn3(self.conv3(x)))

        x = self.act(self.bn4(self.conv4(x)))

        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z



class ConvDecoder_Weather(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_Weather, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (4, 21), stride=(1, 1), padding=0)  # 2 x 9
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (4, 1), stride=(4,1), padding=(0,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (4, 1), stride=(4,1), padding=(0,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (4, 1), stride=(4,1), padding=(0,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 1, (2, 1), stride=(2,1), padding=(0,0))  # 64 x 9
         # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        mu_img = self.conv5(x)
        return mu_img

class ConvEncoder_Weather_336(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_Weather_336, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 2), stride=(3,1))

        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 3), stride=(2,2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), stride=(2,2))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (2, 4), stride=(2,1))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_z = nn.Conv2d(128, output_dim, (19,1))

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 336, 21)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z

class ConvDecoder_Weather_336(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_Weather_336, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (4, 21), stride=(1, 1), padding=0)  # 2 x 9
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (4, 1), stride=(4,1), padding=(0,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (4, 1), stride=(2,1), padding=(0,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (4, 1), stride=(5,1), padding=(0,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 1, (2, 1), stride=(2,1), padding=(1,0))  # 64 x 9
         # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        mu_img = self.conv5(x)
        return mu_img
#################################################

################# traffic ####################

class ConvEncoder_traffic(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_traffic, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 2), stride=(2,5))

        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 3), stride=(2,5))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 3), stride=(2,5))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (2, 3), stride=(2,5))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_z = nn.Conv2d(128, output_dim, (19,1))

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        x = x.view(-1, 1, 336, 862)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z



class ConvDecoder_traffic(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_traffic, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (4, 862), stride=(1, 1), padding=0)  # 2 x 9
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (4, 1), stride=(4,1), padding=(0,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (4, 1), stride=(2,1), padding=(0,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (4, 1), stride=(5,1), padding=(0,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 1, (2, 1), stride=(2,1), padding=(1,0))  # 64 x 9
         # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        mu_img = self.conv5(x)
        return mu_img
#################################################

class ConvEncoder_usc(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_usc, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (9, 2), stride=(2,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (7, 2), stride=(2,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 2), stride=(2,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (5, 2), stride=(2,1))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 512, (2, 2), stride=(1,1))
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 100, 6)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z


class ConvDecoder_usc(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_usc, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (2, 6), stride=(1, 1), padding=0)  # 2 x 6
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (6, 1), stride=(2,1), padding=(1,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (4, 1), stride=(2,1), padding=(1,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (5, 1), stride=(2,1), padding=(1,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, (4, 1), stride=(2,1), padding=(1,0))  # 64 x 9
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, (4, 1), stride=(2,1), padding=(1,0)) # 128 x 9

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        mu_img = self.conv_final(x)
        return mu_img


#################################################


class ConvEncoder_ieeesmall(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_ieeesmall, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (13, 1), stride=(2,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (9, 1), stride=(2,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (9, 1), stride=(2,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (7, 1), stride=(2,1))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 512, (5, 1), stride=(2,1))
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 200, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z


class ConvDecoder_ieeesmall(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_ieeesmall, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (6, 1), stride=(1, 1), padding=0)  # 2 x 6
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (4, 1), stride=(2,1), padding=(1,0))  # 8 x 9
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (5, 1), stride=(2,1), padding=(1,0))  # 16 x 9
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (4, 1), stride=(2,1), padding=(1,0))  # 32 x 9
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, (4, 1), stride=(2,1), padding=(1,0))  # 64 x 9
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, (4, 1), stride=(2,1), padding=(1,0)) # 128 x 9

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)  # Batch, Latent, 1, 1
        x = self.act(self.bn1(self.conv1(x)))   # Batch, 512, 6, 1
        x = self.act(self.bn2(self.conv2(x)))   # Batch, 128, 12, 1
        x = self.act(self.bn3(self.conv3(x)))   # Batch, 64, 25, 1
        x = self.act(self.bn4(self.conv4(x)))   # Batch, 32, 50, 1
        x = self.act(self.bn5(self.conv5(x)))   # Batch, 32, 100, 1
        mu_img = self.conv_final(x)             # Batch, 32, 200, 1
        return mu_img



#################################################


class ConvEncoder_ecg(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder_ecg, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, (12, 2), stride=(3,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (10, 2), stride=(3,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (8, 2), stride=(3,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (8, 1), stride=(3,1))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 512, (7, 1), stride=(3,1))
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, 1000, 4)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        z = self.conv_z(x).view(x.size(0), self.output_dim)
        return z


class ConvDecoder_ecg(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder_ecg, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, (4, 4), stride=(1, 1), padding=0)  # 4 x 4
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 128, (5, 1), stride=(3,1), padding=(1,0))  # 12 x 4
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, (5, 1), stride=(3,1), padding=(1,0))  # 36 x 4
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (6, 1), stride=(3,1), padding=(1,0))  # 109 x 4
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, (9, 1), stride=(3,1), padding=(1,0))  # 331 x 4
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, (12, 1), stride=(3,1), padding=(1,0)) # 1000 x 4

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        mu_img = self.conv_final(x)
        return mu_img
