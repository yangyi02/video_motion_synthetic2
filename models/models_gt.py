import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class GtNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(GtNet, self).__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel

    def forward(self, im_input, gt_motion):
        m_mask = label2mask(gt_motion, self.n_class)
        seg = construct_seg(m_mask, self.m_kernel, self.m_range)
        disappear = F.relu(seg - 1)
        appear = F.relu(1 - disappear)
        pred = construct_image(im_input[:, -self.im_channel:, :, :], m_mask, appear, self.m_kernel, self.m_range)
        return pred, m_mask, 1 - appear


class BiGtNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(BiGtNet, self).__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel

    def forward(self, im_input_f, im_input_b, gt_motion_f, gt_motion_b):
        m_mask_f = label2mask(gt_motion_f, self.n_class)
        m_mask_b = label2mask(gt_motion_b, self.n_class)

        seg_f = construct_seg(m_mask_f, self.m_kernel, self.m_range)
        seg_b = construct_seg(m_mask_b, self.m_kernel, self.m_range)

        disappear_f = F.relu(seg_f - 1)
        appear_f = F.relu(1 - disappear_f)
        disappear_b = F.relu(seg_b - 1)
        appear_b = F.relu(1 - disappear_b)

        pred_f = construct_image(im_input_f[:, -self.im_channel:, :, :], m_mask_f, appear_f, self.m_kernel, self.m_range)
        pred_b = construct_image(im_input_b[:, -self.im_channel:, :, :], m_mask_b, appear_b, self.m_kernel, self.m_range)

        seg_f = 1 - F.relu(1 - seg_f)
        seg_b = 1 - F.relu(1 - seg_b)

        attn = (seg_f + 1e-5) / (seg_f + seg_b + 2e-5)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b
        return pred, m_mask_f, 1 - appear_f, attn, m_mask_b, 1 - appear_b, 1 - attn


def construct_seg(m_mask, m_kernel, m_range):
    seg = Variable(torch.Tensor(m_mask.size(0), 1, m_mask.size(2), m_mask.size(3)))
    if torch.cuda.is_available():
        seg = seg.cuda()
    for i in range(m_mask.size(0)):
        seg[i, :, :, :] = F.conv2d(m_mask[i, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
    return seg


def construct_image(im, m_mask, appear, m_kernel, m_range):
    im = im * appear.expand_as(im)
    pred = Variable(torch.Tensor(im.size()))
    if torch.cuda.is_available():
        pred = pred.cuda()
    for i in range(im.size(1)):
        im_expand = im[:, i, :, :].unsqueeze(1).expand_as(m_mask) * m_mask
        for j in range(im.size(0)):
            pred[j, i, :, :] = F.conv2d(im_expand[j, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
    return pred


def label2mask(motion, n_class):
    m_mask = Variable(torch.Tensor(motion.size(0), n_class, motion.size(2), motion.size(3)))
    if torch.cuda.is_available():
        m_mask = m_mask.cuda()
    for i in range(motion.size(0)):
        for j in range(n_class):
            tmp = Variable(torch.zeros((motion.size(2), motion.size(3))))
            if torch.cuda.is_available():
                tmp = tmp.cuda()
            tmp[motion[i, :, :, :] == j] = 1
            m_mask[i, j, :, :] = tmp
    return m_mask

