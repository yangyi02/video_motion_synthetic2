import os
import numpy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from learning_args import parse_args
from data import motion_dict, generate_images
from models import Net, BiNet
from models_gt import GtNet, BiGtNet
from visualize import visualize, visualize_bidirection
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def train_unsupervised(args, model, m_kernel, m_dict, reverse_m_dict):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    base_loss = []
    train_loss = []
    best_improve_percent = -1e10
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        if args.bidirection:
            im_input_f, im_input_b, im_output, gt_motion_f, gt_motion_b = generate_images(args, m_dict, reverse_m_dict)
            im_input_f = Variable(torch.from_numpy(im_input_f).float())
            im_input_b = Variable(torch.from_numpy(im_input_b).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            if torch.cuda.is_available():
                im_input_f, im_input_b, im_output = im_input_f.cuda(), im_input_b.cuda(), im_output.cuda()
            im_pred, m_mask_f, disappear_f, attn_f, m_mask_b, disappear_b, attn_b = model(im_input_f, im_input_b)
            base_loss.append(torch.abs(im_input_f[:, -args.num_channel:, :, :] - im_output).sum().data[0])
            base_loss.append(torch.abs(im_input_b[:, -args.num_channel:, :, :] - im_output).sum().data[0])
        else:
            im_input, im_output, gt_motion = generate_images(args, m_dict, reverse_m_dict)
            im_input = Variable(torch.from_numpy(im_input).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
            im_pred, m_mask, disappear = model(im_input)
            base_loss.append(torch.abs(im_input[:, -args.num_channel:, :, :] - im_output).sum().data[0])
        loss = torch.abs(im_pred - im_output).sum()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data[0])
        if len(train_loss) > 1000:
            train_loss.pop(0)
        ave_train_loss = sum(train_loss) / float(len(train_loss))
        if len(base_loss) > 1000:
            base_loss.pop(0)
        ave_base_loss = sum(base_loss) / float(len(base_loss))
        logging.info('epoch %d, training loss: %.2f, average training loss: %.2f, base loss: %.2f', epoch, loss.data[0], ave_train_loss, ave_base_loss)
        if (epoch+1) % args.test_interval == 0:
            logging.info('epoch %d, testing', epoch)
            best_improve_percent = validate(args, model, m_kernel, m_dict, reverse_m_dict, best_improve_percent)
    return model


def validate(args, model, m_kernel, m_dict, reverse_m_dict, best_improve_percent):
    improve_percent = test_unsupervised(args, model, m_kernel, m_dict, reverse_m_dict)
    if improve_percent >= best_improve_percent:
        logging.info('model save to %s', os.path.join(args.save_dir, 'final.pth'))
        with open(os.path.join(args.save_dir, 'final.pth'), 'w') as handle:
            torch.save(model.state_dict(), handle)
        best_improve_percent = improve_percent
    logging.info('current best improved percent: %.2f', best_improve_percent)
    return best_improve_percent


def test_unsupervised(args, model, m_kernel, m_dict, reverse_m_dict):
    m_range = args.motion_range
    base_loss = []
    test_loss = []
    for epoch in range(args.test_epoch):
        if args.bidirection:
            im_input_f, im_input_b, im_output, gt_motion_f, gt_motion_b = generate_images(args, m_dict, reverse_m_dict)
            im_input_f = Variable(torch.from_numpy(im_input_f).float())
            im_input_b = Variable(torch.from_numpy(im_input_b).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            gt_motion_f = Variable(torch.from_numpy(gt_motion_f))
            gt_motion_b = Variable(torch.from_numpy(gt_motion_b))
            if torch.cuda.is_available():
                im_input_f, im_input_b, im_output = im_input_f.cuda(), im_input_b.cuda(), im_output.cuda()
                gt_motion_f, gt_motion_b = gt_motion_f.cuda(), gt_motion_b.cuda()
            im_pred, m_mask_f, disappear_f, attn_f, m_mask_b, disappear_b, attn_b = model(im_input_f, im_input_b)
            base_loss.append(torch.abs(im_input_f[:, -args.num_channel:, :, :] - im_output).sum().data[0])
            base_loss.append(torch.abs(im_input_b[:, -args.num_channel:, :, :] - im_output).sum().data[0])
            if args.display:
                flow_f = motion2flow(m_mask_f, reverse_m_dict)
                flow_b = motion2flow(m_mask_b, reverse_m_dict)
                visualize_bidirection(im_input_f, im_input_b, im_output, im_pred, flow_f, gt_motion_f, disappear_f, attn_f, flow_b, gt_motion_b, disappear_b, attn_b, m_range, reverse_m_dict)
        else:
            im_input, im_output, gt_motion = generate_images(args, m_dict, reverse_m_dict)
            im_input = Variable(torch.from_numpy(im_input).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            gt_motion = Variable(torch.from_numpy(gt_motion))
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
                gt_motion = gt_motion.cuda()
            im_pred, m_mask, disappear = model(im_input)
            base_loss.append(torch.abs(im_input[:, -args.num_channel:, :, :] - im_output).sum().data[0])
            if args.display:
                flow = motion2flow(m_mask, reverse_m_dict)
                visualize(im_input, im_output, im_pred, flow, gt_motion, disappear, m_range, reverse_m_dict)
        loss = torch.abs(im_pred - im_output).sum()
        test_loss.append(loss.data[0])
    test_loss = numpy.mean(numpy.asarray(test_loss))
    base_loss = numpy.mean(numpy.asarray(base_loss))
    improve_loss = base_loss - test_loss
    improve_percent = improve_loss / (base_loss + 1e-5)
    logging.info('average testing loss: %.2f, base loss: %.2f', test_loss, base_loss)
    logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
    return improve_percent


def test_gt(args, model, m_kernel, m_dict, reverse_m_dict):
    m_range = args.motion_range
    base_loss = []
    test_loss = []
    for epoch in range(args.test_epoch):
        if args.bidirection:
            im_input_f, im_input_b, im_output, gt_motion_f, gt_motion_b = generate_images(args, m_dict, reverse_m_dict)
            im_input_f = Variable(torch.from_numpy(im_input_f).float())
            im_input_b = Variable(torch.from_numpy(im_input_b).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            gt_motion_f = Variable(torch.from_numpy(gt_motion_f))
            gt_motion_b = Variable(torch.from_numpy(gt_motion_b))
            if torch.cuda.is_available():
                im_input_f, im_input_b, im_output = im_input_f.cuda(), im_input_b.cuda(), im_output.cuda()
                gt_motion_f, gt_motion_b = gt_motion_f.cuda(), gt_motion_b.cuda()
            im_pred, m_mask_f, disappear_f, attn_f, m_mask_b, disappear_b, attn_b = model(im_input_f, im_input_b, gt_motion_f, gt_motion_b)
            base_loss.append(torch.abs(im_input_f[:, -args.num_channel:, :, :] - im_output).sum().data[0])
            base_loss.append(torch.abs(im_input_b[:, -args.num_channel:, :, :] - im_output).sum().data[0])
            if args.display:
                flow_f = motion2flow(m_mask_f, reverse_m_dict)
                flow_b = motion2flow(m_mask_b, reverse_m_dict)
                visualize_bidirection(im_input_f, im_input_b, im_output, im_pred, flow_f, gt_motion_f, disappear_f, attn_f, flow_b, gt_motion_b, disappear_b, attn_b, m_range, reverse_m_dict)
        else:
            im_input, im_output, gt_motion = generate_images(args, m_dict, reverse_m_dict)
            im_input = Variable(torch.from_numpy(im_input).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            gt_motion = Variable(torch.from_numpy(gt_motion))
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
                gt_motion = gt_motion.cuda()
            im_pred, m_mask, disappear = model(im_input, gt_motion)
            base_loss.append(torch.abs(im_input[:, -args.num_channel:, :, :] - im_output).sum().data[0])
            if args.display:
                flow = motion2flow(m_mask, reverse_m_dict)
                visualize(im_input, im_output, im_pred, flow, gt_motion, disappear, m_range, reverse_m_dict)
        loss = torch.abs(im_pred - im_output).sum()
        test_loss.append(loss.data[0])
    test_loss = numpy.mean(numpy.asarray(test_loss))
    base_loss = numpy.mean(numpy.asarray(base_loss))
    improve_loss = base_loss - test_loss
    improve_percent = improve_loss / (base_loss + 1e-5)
    logging.info('average testing loss: %.2f, base loss: %.2f', test_loss, base_loss)
    logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
    return improve_percent


def motion2flow(m_mask, reverse_m_dict):
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
    m_dict, reverse_m_dict, m_kernel = motion_dict(args.motion_range)
    m_kernel = Variable(torch.from_numpy(m_kernel).float())
    if torch.cuda.is_available():
        m_kernel = m_kernel.cuda()
    height, width, channel, num_inputs, m_range = args.image_size, args.image_size, args.num_channel, args.num_inputs, args.motion_range
    if args.test_gt and args.bidirection:
        model = BiGtNet(height, width, channel, num_inputs, len(m_dict), m_range, m_kernel)
    elif args.test_gt:
        model = GtNet(height, width, channel, num_inputs, len(m_dict), m_range, m_kernel)
    elif args.bidirection:
        model = BiNet(height, width, channel, num_inputs, len(m_dict), m_range, m_kernel)
    else:
        model = Net(height, width, channel, num_inputs, len(m_dict), m_range, m_kernel)
    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
    if args.train:
        model = train_unsupervised(args, model, m_kernel, m_dict, reverse_m_dict)
    if args.test:
        model.load_state_dict(torch.load(args.init_model_path))
        test_unsupervised(args, model, m_kernel, m_dict, reverse_m_dict)
    if args.test_gt:
        test_gt(args, model, m_kernel, m_dict, reverse_m_dict)

if __name__ == '__main__':
    main()
