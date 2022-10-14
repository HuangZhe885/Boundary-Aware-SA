from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable


from torch.autograd import Function

from . import knn_ext

def gather_nd(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices
    returns: tensor shaped [m_1, m_2, m_3, m_4]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)

@torch.no_grad()
def boundary_grouping(features,k, src_xyz, q_xyz,boundary_label,use_xyz=True):
    """
        :param src_xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param q_xyz: (B, npoint, 3) tensor of the xyz coordinates of the grouping centers if specified
        :K :neighbor size
        :return:
             grouped_xyz, 
             new_points, 
             idx, 
             grouped_boundary_label
        """
    assert src_xyz.is_contiguous()
    assert q_xyz.is_contiguous()

    B, N, _ = src_xyz.size()
    npoint = q_xyz.size(1)

    point_indices = KNN(k,src_xyz)
    a = torch.reshape(torch.range(B), (-1, 1, 1, 1))
    batch_indices = a.repeat(1, npoint, K, 1)   

    idx = torch.cat([batch_indices, point_indices.unsqueeze(3)], axis = 3)  
    idx.set_shape([batch_size, npoint, K, 2])  

    grouped_xyz = gather_nd(src_xyz, idx)
    b = q_xyz.unsqueeze(2)
    grouped_xyz -= b.repeat([1,1,K,1])# translation normalization

    if boundary_label is None:
        grouped_feature = gather_nd(feature, idx)
        grouped_boundary_label = None
    else:
        feature = torch.cat([feature,boundary_label.unsqueeze([-1])],2)
        grouped_feature = gather_nd(feature, idx)
        grouped_boundary_label = grouped_feature[:,:,:,-1]
        grouped_feature = grouped_feature[:,:,:,:-1]
    if use_xyz:
        new_points = torch.cat([grouped_xyz, grouped_feature], axis = -1)
    else:
        new_points = grouped_feature
    
    return grouped_xyz, new_points, idx, grouped_boundary_label


class KNN(Function):
    r"""KNN (CUDA) based on heap data structure.
    Modified from `PAConv <https://github.com/CVMI-Lab/PAConv/tree/main/
    scene_seg/lib/pointops/src/knnquery_heap>`_.

    Find k-nearest points.
    """

    @staticmethod
    def forward(ctx,
                k: int,
                xyz: torch.Tensor,
                center_xyz: torch.Tensor = None,
                transposed: bool = False) -> torch.Tensor:
        """Forward.

        Args:
            k (int): number of nearest neighbors.
            xyz (Tensor): (B, N, 3) if transposed == False, else (B, 3, N).
                xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) if transposed == False,
                else (B, 3, npoint). centers of the knn query.
            transposed (bool): whether the input tensors are transposed.
                defaults to False. Should not explicitly use this keyword
                when calling knn (=KNN.apply), just add the fourth param.

        Returns:
            Tensor: (B, k, npoint) tensor with the indices of
                the features that form k-nearest neighbours.
        """
        assert k > 0

        if center_xyz is None:
            center_xyz = xyz

        if transposed:
            xyz = xyz.transpose(2, 1).contiguous()
            center_xyz = center_xyz.transpose(2, 1).contiguous()

        assert xyz.is_contiguous()  # [B, N, 3]
        assert center_xyz.is_contiguous()  # [B, npoint, 3]

        center_xyz_device = center_xyz.get_device()
        assert center_xyz_device == xyz.get_device(), \
            'center_xyz and xyz should be put on the same device'
        if torch.cuda.current_device() != center_xyz_device:
            torch.cuda.set_device(center_xyz_device)

        B, npoint, _ = center_xyz.shape
        N = xyz.shape[1]

        idx = center_xyz.new_zeros((B, npoint, k)).int()
        dist2 = center_xyz.new_zeros((B, npoint, k)).float()

        knn_ext.knn_wrapper(B, N, npoint, k, xyz, center_xyz, idx, dist2)
        # idx shape to [B, k, npoint]
        idx = idx.transpose(2, 1).contiguous()
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


knn = KNN.apply