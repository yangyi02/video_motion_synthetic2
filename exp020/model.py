import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(Net, self).__init__()
        num_hidden = 64
        self.conv0 = nn.Conv2d(n_inputs*im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv7 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)

        self.conv0_d = nn.Conv2d(im_channel, num_hidden, 3, 1, 1)
        self.bn0_d = nn.BatchNorm2d(num_hidden)
        self.conv1_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1_d = nn.BatchNorm2d(num_hidden)
        self.conv2_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_d = nn.BatchNorm2d(num_hidden)
        self.conv3_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3_d = nn.BatchNorm2d(num_hidden)
        self.conv4_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4_d = nn.BatchNorm2d(num_hidden)
        self.conv5_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5_d = nn.BatchNorm2d(num_hidden)
        self.conv6_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6_d = nn.BatchNorm2d(num_hidden)
        self.conv7_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn7_d = nn.BatchNorm2d(num_hidden)
        self.conv8_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn8_d = nn.BatchNorm2d(num_hidden)
        self.conv9_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn9_d = nn.BatchNorm2d(num_hidden)
        self.conv10_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn10_d = nn.BatchNorm2d(num_hidden)
        self.conv11_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn11_d = nn.BatchNorm2d(num_hidden)
        self.conv_depth = nn.Conv2d(num_hidden, 1, 3, 1, 1)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_inputs = n_inputs
        self.n_class = n_class
        self.m_range = m_range
        m_kernel = m_kernel.swapaxes(0, 1)
        self.m_kernel = Variable(torch.from_numpy(m_kernel).float())
        if torch.cuda.is_available():
            self.m_kernel = self.m_kernel.cuda()

    def forward(self, im_input_f, im_input_b, im_output):
        x = self.bn0(self.conv0(im_input_f))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x6 = self.upsample(x6)
        x7 = torch.cat((x6, x5), 1)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x4), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x3), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x2), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x1), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        m_mask_f = F.softmax(self.conv(x11))

        im_f = im_input_f[:, -self.im_channel:, :, :]
        x = self.bn0_d(self.conv0_d(im_f))
        x1 = F.relu(self.bn1_d(self.conv1_d(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2_d(self.conv2_d(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3_d(self.conv3_d(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4_d(self.conv4_d(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5_d(self.conv5_d(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6_d(self.conv6_d(x6)))
        x6 = self.upsample(x6)
        x7 = torch.cat((x6, x5), 1)
        x7 = F.relu(self.bn7_d(self.conv7_d(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x4), 1)
        x8 = F.relu(self.bn8_d(self.conv8_d(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x3), 1)
        x9 = F.relu(self.bn9_d(self.conv9_d(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x2), 1)
        x10 = F.relu(self.bn10_d(self.conv10_d(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x1), 1)
        x11 = F.relu(self.bn11_d(self.conv11_d(x11)))
        depth_f = F.sigmoid(self.conv_depth(x11))


        x = self.bn0(self.conv0(im_input_b))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x6 = self.upsample(x6)
        x7 = torch.cat((x6, x5), 1)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x4), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x3), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x2), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x1), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        m_mask_b = F.softmax(self.conv(x11))

        im_b = im_input_b[:, -self.im_channel:, :, :]
        x = self.bn0_d(self.conv0_d(im_b))
        x1 = F.relu(self.bn1_d(self.conv1_d(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2_d(self.conv2_d(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3_d(self.conv3_d(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4_d(self.conv4_d(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5_d(self.conv5_d(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6_d(self.conv6_d(x6)))
        x6 = self.upsample(x6)
        x7 = torch.cat((x6, x5), 1)
        x7 = F.relu(self.bn7_d(self.conv7_d(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x4), 1)
        x8 = F.relu(self.bn8_d(self.conv8_d(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x3), 1)
        x9 = F.relu(self.bn9_d(self.conv9_d(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x2), 1)
        x10 = F.relu(self.bn10_d(self.conv10_d(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x1), 1)
        x11 = F.relu(self.bn11_d(self.conv11_d(x11)))
        depth_b = F.sigmoid(self.conv_depth(x11))

        mask_f = F.conv2d(m_mask_f, self.m_kernel, None, 1, self.m_range, 1, self.m_kernel.size(0))
        mask_b = F.conv2d(m_mask_b, self.m_kernel, None, 1, self.m_range, 1, self.m_kernel.size(0))
        occl_f = seg2occl(depth_f, mask_f, self.m_kernel, self.m_range)
        occl_b = seg2occl(depth_b, mask_b, self.m_kernel, self.m_range)
        pred_f, unocclude_f = construct_image(im_f, mask_f, occl_f, self.m_kernel, self.m_range)
        pred_b, unocclude_b = construct_image(im_b, mask_b, occl_b, self.m_kernel, self.m_range)
        seg_f = unocclude_f.sum(1)
        seg_b = unocclude_b.sum(1)
        appear_f = F.relu(1 - seg_f)
        conflict_f = F.relu(seg_f - 1)
        appear_b = F.relu(1 - seg_b)
        conflict_b = F.relu(seg_b - 1)
        attn = (seg_f + 1e-5) / (seg_f + seg_b + 2e-5)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b

        x = self.bn0_d(self.conv0_d(im_output))
        x1 = F.relu(self.bn1_d(self.conv1_d(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2_d(self.conv2_d(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3_d(self.conv3_d(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4_d(self.conv4_d(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5_d(self.conv5_d(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6_d(self.conv6_d(x6)))
        x6 = self.upsample(x6)
        x7 = torch.cat((x6, x5), 1)
        x7 = F.relu(self.bn7_d(self.conv7_d(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x4), 1)
        x8 = F.relu(self.bn8_d(self.conv8_d(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x3), 1)
        x9 = F.relu(self.bn9_d(self.conv9_d(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x2), 1)
        x10 = F.relu(self.bn10_d(self.conv10_d(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x1), 1)
        x11 = F.relu(self.bn11_d(self.conv11_d(x11)))
        depth_out = F.sigmoid(self.conv_depth(x11))

        pred_depth_f, _ = construct_image(depth_f, mask_f, occl_f, self.m_kernel, self.m_range)
        pred_depth_b, _ = construct_image(depth_b, mask_b, occl_b, self.m_kernel, self.m_range)
        pred_depth = attn.expand_as(pred_depth_f) * pred_depth_f + (1 - attn.expand_as(pred_depth_b)) * pred_depth_b
        return pred, m_mask_f, depth_f, appear_f, conflict_f, attn, m_mask_b, depth_b, appear_b, conflict_b, 1 - attn, pred_depth, depth_out


class GtNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(GtNet, self).__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        m_kernel = m_kernel.swapaxes(0, 1)
        self.m_kernel = Variable(torch.from_numpy(m_kernel).float())
        if torch.cuda.is_available():
            self.m_kernel = self.m_kernel.cuda()

    def forward(self, im_input_f, im_input_b, gt_motion_f, gt_motion_b, depth_f, depth_b, depth_out):
        m_mask_f = self.label2mask(gt_motion_f, self.n_class)
        m_mask_b = self.label2mask(gt_motion_b, self.n_class)
        mask_f = F.conv2d(m_mask_f, self.m_kernel, None, 1, self.m_range, 1, self.m_kernel.size(0))
        mask_b = F.conv2d(m_mask_b, self.m_kernel, None, 1, self.m_range, 1, self.m_kernel.size(0))
        occl_f = seg2occl(depth_f, mask_f, self.m_kernel, self.m_range)
        occl_b = seg2occl(depth_b, mask_b, self.m_kernel, self.m_range)
        im_f = im_input_f[:, -self.im_channel:, :, :]
        im_b = im_input_b[:, -self.im_channel:, :, :]
        pred_f, unocclude_f = construct_image(im_f, mask_f, occl_f, self.m_kernel, self.m_range)
        pred_b, unocclude_b = construct_image(im_b, mask_b, occl_b, self.m_kernel, self.m_range)
        seg_f = unocclude_f.sum(1)
        seg_b = unocclude_b.sum(1)
        appear_f = F.relu(1 - seg_f)
        conflict_f = F.relu(seg_f - 1)
        appear_b = F.relu(1 - seg_b)
        conflict_b = F.relu(seg_b - 1)
        attn = (seg_f + 1e-5) / (seg_f + seg_b + 2e-5)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b
        pred_depth_f, _ = construct_image(depth_f, mask_f, occl_f, self.m_kernel, self.m_range)
        pred_depth_b, _ = construct_image(depth_b, mask_b, occl_b, self.m_kernel, self.m_range)
        pred_depth = attn.expand_as(pred_depth_f) * pred_depth_f + (1 - attn.expand_as(pred_depth_b)) * pred_depth_b
        return pred, m_mask_f, depth_f, appear_f, conflict_f, attn, m_mask_b, depth_b, appear_b, conflict_b, 1 - attn, pred_depth, depth_out

    def label2mask(self, motion, n_class):
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


def seg2occl(depth, mask, m_kernel, m_range):
    seg_expand = depth.expand_as(mask)
    nearby_seg = F.conv2d(seg_expand, m_kernel, None, 1, m_range, 1, m_kernel.size(0))
    seg_max, _ = torch.max(nearby_seg * mask, 1)
    occl = nearby_seg * mask - seg_max.expand_as(nearby_seg)
    occl = occl + 1
    occl = F.relu(occl)
    return occl


def construct_image(im, mask, occl, m_kernel, m_range):
    unocclude_mask = mask * occl
    pred = Variable(torch.Tensor(im.size()))
    if torch.cuda.is_available():
        pred = pred.cuda()
    for i in range(im.size(1)):
        im_expand = im[:, i, :, :].unsqueeze(1).expand_as(mask)
        nearby_im = F.conv2d(im_expand, m_kernel, None, 1, m_range, 1, m_kernel.size(0))
        pred[:, i, :, :] = (nearby_im * unocclude_mask).sum(1)
    return pred, unocclude_mask
