import torch
from exp import Exp_Online
from util.buffer import Buffer



class Exp_ER(Exp_Online):
    """
    Online learning class implementing Experience Replay
    - Store past data in buffer to prevent forgetting
    - buffer_size determines number of past data, mini_batch determines number of data retrieved from buffer
    - ER_alpha determines weight of loss from past data in buffer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.count = 0
        self.alpha = self.args.ER_alpha

    def train_loss(self, criterion, batch, outputs):
        """
        Loss calculation including experience replay
        - Add loss from past data in buffer to normal loss
        """
        loss = super().train_loss(criterion, batch, outputs)
        if not self.buffer.is_empty():
            buff = self.buffer.get_data(self.args.mini_batch)  # Get mini_batch samples from buffer
            # buff[0] is input data, buff[1] is label data, buff[2] is input time features, buff[3] is output time features
            out = self.forward(buff[:-1])   # buff[:-1] is "data excluding last attribute (index)"
            if isinstance(outputs, (tuple, list)):
                out = out[0]
            loss += self.alpha * criterion(out, buff[1])  # Add buffer data loss scaled by 0.2
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None, current_batch=None):
        """
        Add data to buffer during online update
        """
        loss, outputs = self._update(batch, criterion, optimizer, scaler)
        idx = self.count + torch.arange(batch[1].size(0)).to(self.device)
        self.count += batch[1].size(0)
        self.buffer.add_data(*batch[:4], idx)  # Add data and index to buffer
        return loss, outputs


class Exp_DERpp(Exp_Online):
    """
    Online learning class implementing Dark Experience Replay++ (DER++)
    - Extends ER to achieve more effective experience replay
    - Saves and utilizes prediction outputs in buffer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.count = 0
        self.alpha = self.args.ER_alpha
        self.beta = self.args.ER_beta

    def train_loss(self, criterion, batch, outputs):
        """
        Loss calculation for DER++
        - Add loss from prediction outputs in buffer to normal loss
        """
        loss = super().train_loss(criterion, batch, outputs)
        if not self.buffer.is_empty():
            buff = self.buffer.get_data(self.args.mini_batch)
            out = self.forward(buff[:-1])
            if isinstance(outputs, (tuple, list)):
                out = out[0]
            loss += self.alpha * criterion(buff[1], out)  # Add loss between prediction output and label data
            loss += self.beta * criterion(buff[-1], out)  # Add loss between current and past model prediction outputs
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None, current_batch=None):
        """
        Add prediction outputs to buffer during online update
        """
        loss, outputs = self._update(batch, criterion, optimizer, scaler)
        self.count += batch[1].size(0)
        if isinstance(outputs, (tuple, list)):
            self.buffer.add_data(*(batch + [outputs[0]]))  # Add prediction output to buffer
        else:
            self.buffer.add_data(*(batch + [outputs]))
        return loss, outputs
