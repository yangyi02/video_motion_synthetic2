import os
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from learning_args import parse_args
from data.mnist_data_bidirect import MnistDataBidirect
from data.box_data_bidirect import BoxDataBidirect
from base_demo import BaseDemo
from model import Net, GtNet
from visualizer import Visualizer
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class Demo(BaseDemo):
    def __init__(self, args):
        super(Demo, self).__init__(args)
        if args.data == 'box':
            self.data = BoxDataBidirect(args)
        elif args.data == 'mnist':
            self.data = MnistDataBidirect(args)
        self.model, self.model_gt = self.init_model(self.data.m_kernel)
        self.visualizer = Visualizer(args, self.data.reverse_m_dict)
        self.num_inputs = (self.num_frame - 1) / 2

    def init_model(self, m_kernel):
        num_inputs = (self.num_frame - 1) / 2
        self.model = Net(self.im_size, self.im_size, 3, num_inputs,
                             m_kernel.shape[1], self.m_range, m_kernel)
        self.model_gt = GtNet(self.im_size, self.im_size, 3, num_inputs,
                                  m_kernel.shape[1], self.m_range, m_kernel)
        if torch.cuda.is_available():
            # model = torch.nn.DataParallel(model).cuda()
            self.model = self.model.cuda()
            self.model_gt = self.model_gt.cuda()
        if self.init_model_path is not '':
            self.model.load_state_dict(torch.load(self.init_model_path))
        return self.model, self.model_gt

    def train_unsupervised(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        base_loss, train_loss = [], []
        for epoch in range(self.train_epoch):
            optimizer.zero_grad()
            im, motion, motion_r = self.data.get_next_batch(self.data.train_images)
            im_input_f = im[:, :self.num_inputs, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_input_b = im[:, :self.num_inputs:-1, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, self.num_inputs, :, :, :]
            im_input_f = Variable(torch.from_numpy(im_input_f).float())
            im_input_b = Variable(torch.from_numpy(im_input_b).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            if torch.cuda.is_available():
                im_input_f, im_input_b = im_input_f.cuda(), im_input_b.cuda()
                im_output = im_output.cuda()
            im_pred, m_mask_f, disappear_f, attn_f, m_mask_b, disappear_b, attn_b = \
                self.model(im_input_f, im_input_b)
            im_diff = im_pred - im_output
            loss = torch.abs(im_diff).sum()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.data[0])
            if len(train_loss) > 100:
                train_loss.pop(0)
            ave_train_loss = sum(train_loss) / float(len(train_loss))
            base_loss.append(torch.abs(im_input_f[:, -3:, :, :] - im_output).sum().data[0])
            base_loss.append(torch.abs(im_input_b[:, -3:, :, :] - im_output).sum().data[0])
            if len(base_loss) > 100:
                base_loss.pop(0)
            ave_base_loss = sum(base_loss) / float(len(base_loss))
            logging.info('epoch %d, train loss: %.2f, average train loss: %.2f, base loss: %.2f',
                         epoch, loss.data[0], ave_train_loss, ave_base_loss)
            if (epoch+1) % self.test_interval == 0:
                logging.info('epoch %d, testing', epoch)
                self.validate()

    def test_unsupervised(self):
        base_loss, test_loss = [], []
        test_accuracy = []
        for epoch in range(self.test_epoch):
            im, motion, motion_r = self.data.get_next_batch(self.data.test_images)
            im_input_f = im[:, :self.num_inputs, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_input_b = im[:, :self.num_inputs:-1, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, self.num_inputs, :, :, :]
            gt_motion_f = motion[:, self.num_inputs-1, :, :, :]
            gt_motion_b = motion_r[:, self.num_inputs+1, :, :, :]
            im_input_f = Variable(torch.from_numpy(im_input_f).float())
            im_input_b = Variable(torch.from_numpy(im_input_b).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            gt_motion_f = Variable(torch.from_numpy(gt_motion_f))
            gt_motion_b = Variable(torch.from_numpy(gt_motion_b))
            if torch.cuda.is_available():
                im_input_f, im_input_b = im_input_f.cuda(), im_input_b.cuda()
                im_output = im_output.cuda()
                gt_motion_f, gt_motion_b = gt_motion_f.cuda(), gt_motion_b.cuda()
            im_pred, m_mask_f, disappear_f, attn_f, m_mask_b, disappear_b, attn_b = \
                self.model(im_input_f, im_input_b)
            im_diff = im_pred - im_output
            loss = torch.abs(im_diff).sum()

            test_loss.append(loss.data[0])
            base_loss.append(torch.abs(im_input_f[:, -3:, :, :] - im_output).sum().data[0])
            base_loss.append(torch.abs(im_input_b[:, -3:, :, :] - im_output).sum().data[0])
            pred_motion_f = m_mask_f.max(1)[1]
            pred_motion_b = m_mask_b.max(1)[1]
            accuracy_f = pred_motion_f.eq(gt_motion_f).float().sum() / gt_motion_f.numel()
            accuracy_b = pred_motion_b.eq(gt_motion_b).float().sum() / gt_motion_b.numel()
            test_accuracy.append(accuracy_f.cpu().data[0])
            test_accuracy.append(accuracy_b.cpu().data[0])
            if self.display:
                flow_f = self.motion2flow(m_mask_f)
                flow_b = self.motion2flow(m_mask_b)
                self.visualizer.visualize_result_bidirect(im_input_f, im_input_b, im_output,
                                                          im_pred, flow_f, gt_motion_f, disappear_f,
                                                          attn_f, flow_b, gt_motion_b, disappear_b,
                                                          attn_b, 'test_%d.png' % epoch)
        test_loss = numpy.mean(numpy.asarray(test_loss))
        base_loss = numpy.mean(numpy.asarray(base_loss))
        improve_loss = base_loss - test_loss
        improve_percent = improve_loss / (base_loss + 1e-5)
        logging.info('average test loss: %.2f, base loss: %.2f', test_loss, base_loss)
        logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
        test_accuracy = numpy.mean(numpy.asarray(test_accuracy))
        logging.info('average test accuracy: %.2f', test_accuracy)
        return improve_percent

    def test_gt_unsupervised(self):
        base_loss, test_loss = [], []
        test_accuracy = []
        for epoch in range(self.test_epoch):
            im, motion, motion_r = self.data.get_next_batch(self.data.test_images)
            im_input_f = im[:, :self.num_inputs, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_input_b = im[:, :self.num_inputs:-1, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, self.num_inputs, :, :, :]
            gt_motion_f = motion[:, self.num_inputs-1, :, :, :]
            gt_motion_b = motion_r[:, self.num_inputs+1, :, :, :]
            im_input_f = Variable(torch.from_numpy(im_input_f).float())
            im_input_b = Variable(torch.from_numpy(im_input_b).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            gt_motion_f = Variable(torch.from_numpy(gt_motion_f))
            gt_motion_b = Variable(torch.from_numpy(gt_motion_b))
            if torch.cuda.is_available():
                im_input_f, im_input_b = im_input_f.cuda(), im_input_b.cuda()
                im_output = im_output.cuda()
                gt_motion_f, gt_motion_b = gt_motion_f.cuda(), gt_motion_b.cuda()
            im_pred, m_mask_f, disappear_f, attn_f, m_mask_b, disappear_b, attn_b = \
                self.model_gt(im_input_f, im_input_b, gt_motion_f, gt_motion_b)
            im_diff = im_pred - im_output
            loss = torch.abs(im_diff).sum()

            test_loss.append(loss.data[0])
            base_loss.append(torch.abs(im_input_f[:, -3:, :, :] - im_output).sum().data[0])
            base_loss.append(torch.abs(im_input_b[:, -3:, :, :] - im_output).sum().data[0])
            pred_motion_f = m_mask_f.max(1)[1]
            pred_motion_b = m_mask_b.max(1)[1]
            accuracy_f = pred_motion_f.eq(gt_motion_f).float().sum() / gt_motion_f.numel()
            accuracy_b = pred_motion_b.eq(gt_motion_b).float().sum() / gt_motion_b.numel()
            test_accuracy.append(accuracy_f.cpu().data[0])
            test_accuracy.append(accuracy_b.cpu().data[0])
            if self.display:
                flow_f = self.motion2flow(m_mask_f)
                flow_b = self.motion2flow(m_mask_b)
                self.visualizer.visualize_result_bidirect(im_input_f, im_input_b, im_output,
                                                          im_pred, flow_f, gt_motion_f, disappear_f,
                                                          attn_f, flow_b, gt_motion_b, disappear_b,
                                                          attn_b, 'test_gt.png')
        test_loss = numpy.mean(numpy.asarray(test_loss))
        base_loss = numpy.mean(numpy.asarray(base_loss))
        improve_loss = base_loss - test_loss
        improve_percent = improve_loss / (base_loss + 1e-5)
        logging.info('average groundtruth test loss: %.2f, base loss: %.2f', test_loss, base_loss)
        logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
        test_accuracy = numpy.mean(numpy.asarray(test_accuracy))
        logging.info('average groundtruth test accuracy: %.2f', test_accuracy)
        return improve_percent

    def motion2flow(self, m_mask):
        reverse_m_dict = self.data.reverse_m_dict
        [batch_size, num_class, height, width] = m_mask.size()
        kernel_x = Variable(torch.zeros(batch_size, num_class, height, width))
        kernel_y = Variable(torch.zeros(batch_size, num_class, height, width))
        if torch.cuda.is_available():
            kernel_x = kernel_x.cuda()
            kernel_y = kernel_y.cuda()
        for i in range(num_class):
            (m_x, m_y) = reverse_m_dict[i]
            kernel_x[:, i, :, :] = m_x
            kernel_y[:, i, :, :] = m_y
        flow = Variable(torch.zeros(batch_size, 2, height, width))
        flow[:, 0, :, :] = (m_mask * kernel_x).sum(1)
        flow[:, 1, :, :] = (m_mask * kernel_y).sum(1)
        return flow


def main():
    args = parse_args()
    logging.info(args)
    demo = Demo(args)
    if args.train:
        demo.train_unsupervised()
    if args.test:
        demo.test_unsupervised()
    if args.test_gt:
        demo.test_gt_unsupervised()

if __name__ == '__main__':
    main()
