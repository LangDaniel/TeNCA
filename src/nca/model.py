import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCA2D(nn.Module):
    def __init__(
        self,
        n_steps,
        channel_n,
        fire_rate,
        hidden_size,
        propagate_time,
        device,
        input_channels,
        n_time_points,
        activation=False,
        init_method='standard',
        kernel_size=3,             
        padding=1,
        ):
        super(NCA2D, self).__init__()

        assert init_method in ['standard', 'xavier']
        assert activation in [False, 'sigmoid']

        self.n_steps = n_steps
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size, stride=1,
                            padding=padding, groups=channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size, stride=1,
                            padding=padding, groups=channel_n, padding_mode="reflect")

        fc0in = channel_n*3
        if propagate_time:
            fc0in += 1
        self.fc0 = nn.Linear(fc0in, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        if not activation:
            with torch.no_grad():
                self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.n_time_points = n_time_points
        self.propagate_time = propagate_time

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.activation = activation

        self.device = device
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x,z1,z2), 1)
        return y

    def update(self, x, step=None):
        x = x.transpose(1, -1) # channels first

        dx = self.perceive(x)
        dx = dx.transpose(1, -1) # change channels back

        if step != None: # add info of step number to first fc layer
            step = torch.ones((*dx.shape[:-1], 1), dtype=dx.dtype) * step
            step = step.to(self.device)
            dx = torch.cat([dx, step], dim=-1)

        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        # add output activation
        if self.activation == 'sigmoid':
            dx = torch.sigmoid(dx)

        if self.fire_rate < 1:
            stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>self.fire_rate
            stochastic = stochastic.float().to(self.device)
            dx = dx * stochastic

        x = x+dx.transpose(1, -1)
        x = x.transpose(1, -1)

        return x

    def forward(self, inputs, acquire_at_step, replace_by_targets=False, *args, **kwargs):
        # zero pad if less time points in acquire_at than self.n_time_points
        x = inputs
        #data = torch.zeros((x.shape[0], self.n_time_points, *x.shape[1:])).to(self.device)
        data = torch.zeros((*acquire_at_step.shape[:2], *x.shape[1:])).to(self.device)

        for step in range(self.n_steps):
            if self.propagate_time:
                progress = step/self.n_steps
            else:
                progress = None

            x = self.update(x, progress)

            if (step > 0) and (step in acquire_at_step):
                # if step in acquire_at_step: add current frame/image to output
                at = torch.where(acquire_at_step == step)
                for b, t in zip(*at): # for batch, time-point in ...
                    data[b, t, ...] = x[b].clone()
                    # if replace and replace_by_target > rand(0, 1): replace current frame by ground truth frame
                    if replace_by_targets > np.random.random(): #np.random.random():
                        x[b] = kwargs['targets'][b, t, ..., :self.input_channels].clone()
        data.requires_grad_()
        return data
