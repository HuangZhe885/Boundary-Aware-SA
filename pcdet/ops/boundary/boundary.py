
import torch
import torch.nn as nn
from torch.autograd import Function
from ..pointnet2.pointnet2_batch import pointnet2_utils


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
def boundary_grouping(feature,k, src_xyz, q_xyz,use_xyz=True,boundary_label=None):
    """
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
        :return:
             grouped_xyz, 
             new_points,
             idx,
             grouped_boundary_label
        """
    batch_size, N, _ = src_xyz.size()
    npoint = q_xyz.size(1)
    
    src_xyz = src_xyz.contiguous()
    q_xyz = q_xyz.contiguous()
      
    radius = 1.0
    _,point_indices= pointnet2_utils.ball_query1(radius, k, src_xyz, q_xyz)
    a = torch.reshape(torch.arange(0,batch_size), (-1, 1, 1, 1))
    batch_indices = a.repeat(1, npoint, k, 1)

    batch_indices = batch_indices.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
   
    idx1 = torch.unsqueeze(point_indices,3)
    idx = torch.cat((batch_indices, idx1), 3)
    idx.reshape([batch_size, npoint, k, 2])
    grouped_xyz = gather_nd(src_xyz, idx)

    b = torch.unsqueeze(q_xyz, 2)
    #grouped_xyz -= b.repeat(1,1,k,1)# translation normalization

    feature = feature.cuda()

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




def get_boundary(labels_pl, point_cloud):
    # generate boundary     1 if boundary else 0
        num_neighbor = 32
        ratio = 0.6
        
        feature = (torch.unsqueeze(labels_pl,-1)).type(torch.FloatTensor)
        g_xyz,g_lables,g_indx,_ = boundary_grouping(feature, num_neighbor, point_cloud[:, :, :3], point_cloud[:, :, :3], use_xyz = False)

        g_lables = torch.squeeze(g_lables,-1)
        self_labels = ((torch.unsqueeze(labels_pl, -1))).repeat(1, 1, num_neighbor)
        self_labels = self_labels.type(torch.FloatTensor)
        self_labels = self_labels.cuda()
        # print('g_lables',g_lables.shape)
        # print('self_labels',self_labels.shape)
        compare_tensor = torch.eq(g_lables, self_labels)  #比较g_lables,和self_labels是否相似，返回TRUE和FALSE矩阵
        # print(compare_tensor)
        compare_tensor = compare_tensor.type(torch.FloatTensor)#torch.Size([4, 65536, 32])

        same_lable_num = torch.sum(compare_tensor, axis=2)  #按axis2求和

        boundary_points =  torch.le(same_lable_num, (num_neighbor * ratio)) #相似的标签小于某个值=1，则为边界点

        boundary_points = boundary_points.type(torch.FloatTensor)

        target_boundary_label = boundary_points.requires_grad_(requires_grad=False)


        return target_boundary_label
