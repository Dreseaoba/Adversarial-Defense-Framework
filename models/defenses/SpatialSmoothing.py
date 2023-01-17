"""
This module implements the local spatial smoothing defence in `SpatialSmoothing`.
Paper link: https://arxiv.org/abs/1704.01155
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage.filters import median_filter

from art.utils import CLIP_VALUES_TYPE
# from art.defences.preprocessor.preprocessor import Preprocessor



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torchvision.transforms import ToTensor


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

def numpy_to_tensor(a:np.ndarray)->torch.Tensor:
    to_sensor = ToTensor()
    tensor_list = [to_sensor(a[i]).unsqueeze(0) for i in range(a.shape[0])]
    return torch.cat(tensor_list, dim=0)


class SpatialSmoothingTorch():

    def __init__(
        self,
        window_size: int = 3,
        clip_values: Optional[CLIP_VALUES_TYPE] = None,
        device = None,
        return_numpy = False
        ) -> None:
        self.clip_values = clip_values
        self.window_size = window_size
        self._check_params()
        if device is None:
            self.device = torch.device('cuda')
        else:
            self.device = device
        self.filter = MedianPool2d(window_size, same=True)
        self.return_numpy = return_numpy

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply local spatial smoothing to sample `x`.

        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Smoothed sample.
        """
        x_ndim = x.ndim
        if x_ndim not in [4, 5]:
            raise ValueError(
                "Unrecognized input dimension. Spatial smoothing can only be applied to image and video data."
            )

        # get channel index
        x = x.to(self.device)
        result = self.filter(x)

        if self.return_numpy:
            result = result.cpu().numpy()

        # if self.clip_values is not None:
        #     np.clip(result, self.clip_values[0], self.clip_values[1], out=result)

        return result

    def _check_params(self) -> None:
        if not (isinstance(self.window_size, int) and self.window_size > 0):
            raise ValueError("Sliding window size must be a positive integer.")

        if self.clip_values is not None and len(self.clip_values) != 2:
            raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")

        if self.clip_values is not None and np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid 'clip_values': min >= max.")
    


class SpatialSmoothing():
    """
    Implement the local spatial smoothing defence approach.

    | Paper link: https://arxiv.org/abs/1704.01155

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["window_size", "channels_first", "clip_values"]

    def __init__(
        self,
        window_size: int = 3,
        channels_first: bool = False,
        clip_values: Optional[CLIP_VALUES_TYPE] = None,
    ) -> None:
        """
        Create an instance of local spatial smoothing.

        :param channels_first: Set channels first or last.
        :param window_size: The size of the sliding window.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        """
        self.channels_first = channels_first
        self.window_size = window_size
        self.clip_values = clip_values
        self._check_params()

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply local spatial smoothing to sample `x`.

        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Smoothed sample.
        """
        x_ndim = x.ndim
        if x_ndim not in [4, 5]:
            raise ValueError(
                "Unrecognized input dimension. Spatial smoothing can only be applied to image and video data."
            )

        # get channel index
        channel_index = 1 if self.channels_first else x_ndim - 1

        filter_size = [self.window_size] * x_ndim
        # set filter_size at batch and channel indices to 1
        filter_size[0] = 1
        filter_size[channel_index] = 1
        # set filter_size at temporal index to 1
        if x_ndim == 5:
            # check if NCFHW or NFHWC
            temporal_index = 2 if self.channels_first else 1
            filter_size[temporal_index] = 1
        # Note median_filter:
        # * center pixel located lower right
        # * if window size even, use larger value (e.g. median(4,5)=5)
        result = median_filter(x, size=tuple(filter_size), mode="reflect")

        if self.clip_values is not None:
            np.clip(result, self.clip_values[0], self.clip_values[1], out=result)

        return result

    def _check_params(self) -> None:
        if not (isinstance(self.window_size, int) and self.window_size > 0):
            raise ValueError("Sliding window size must be a positive integer.")

        if self.clip_values is not None and len(self.clip_values) != 2:
            raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")

        if self.clip_values is not None and np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid 'clip_values': min >= max.")



if __name__ == "__main__":
    # Initialize the SpatialSmoothing defence. 
    ss = SpatialSmoothing(window_size=3)

    # Apply the defence to the original input and to the adversarial sample, respectively:
    # x_art_def, _ = ss(x_art)