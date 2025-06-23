import torch
from monai.losses.ssim_loss import SSIMLoss
from torchmetrics.image.fid import FrechetInceptionDistance

class MSELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        if reduction:
            assert reduction in ['mean']
        self.reduction = reduction
        super().__init__()
    def forward(self, inputs, targets, *args, **kwargs):
        # exclude nan values, used for padding, from loss
        inp = inputs[~torch.isnan(targets)]
        tar = targets[~torch.isnan(targets)]
        mse = ((inp - tar)**2 )
        if self.reduction == 'mean':
            return mse.mean()
        return mse

class MAELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        if reduction:
            assert reduction in ['mean']
        self.reduction = reduction
        super().__init__()
    def forward(self, inputs, targets, *args, **kwargs):
        # exclude nan values, used for padding, from loss
        inp = inputs[~torch.isnan(targets)]
        tar = targets[~torch.isnan(targets)]
        mae = torch.abs(inp - tar)
        if self.reduction == 'mean':
            return mae.mean()
        return mae

class MaskedSSIMLoss(torch.nn.Module):
    def __init__(self, spatial_dims=2, data_range=1., reduction='mean'):
        super().__init__()
        self.ssim = SSIMLoss(
            spatial_dims=spatial_dims,
            data_range=data_range,
            reduction=reduction
        )

    def forward(self, inputs, targets, mask):
        count = torch.sum((mask == True).type(torch.int))
        ssim_score = 0
        for b, t in zip(*torch.where(mask == True)):
            ssim_score  += self.ssim(
                inputs[b, t, ...].unsqueeze(0).transpose(1, -1),
                targets[b, t, ...].unsqueeze(0).transpose(1, -1),
            )
        # return mean ssim_score of batch
        return ssim_score / count

class MaskedFID(torch.nn.Module):
    def __init__(self, feature_dim, input_img_size, device, normalize=True):
        super().__init__()
        self.fid = FrechetInceptionDistance(
            feature=feature_dim,
            normalize=normalize,
            input_img_size=input_img_size,
        ).to(device)

    def forward(self, inputs, targets, mask):
        #count = torch.sum((mask == True).type(torch.int))
        for b, t in zip(*torch.where(mask == True)):
            self.fid.update(
                inputs[b, t, ...].unsqueeze(0).repeat(1, 1, 1, 3).transpose(-1, 1),
                real=False,
            )
            self.fid.update(
                targets[b, t, ...].unsqueeze(0).repeat(1, 1, 1, 3).transpose(-1, 1),
                real=True,
            )
        # compute fid of the batch for all sequences
        fid_score = self.fid.compute()
        self.fid.reset()
        return fid_score

class SSIMMAELoss(torch.nn.Module):
    def __init__(self, spatial_dims=2, data_range=1., reduction='mean', weights={'mae': 1., 'ssim': 1.}):
        super().__init__()
        self.ssim = MaskedSSIMLoss(spatial_dims, data_range, reduction)
        self.mae = MAELoss(reduction)
        self.weights = weights

    def forward(self, inputs, targets, mask):
        ssim = self.ssim(inputs, targets, mask)
        mae = self.mae(inputs, targets)
        return self.weights['ssim']*ssim + self.weights['mae']*mae
