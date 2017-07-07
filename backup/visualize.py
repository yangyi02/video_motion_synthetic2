import numpy
import cv2
import matplotlib.pyplot as plt
import flowlib
from PIL import Image


def visualize(im_input, im_output, pred, pred_motion, gt_motion, disappear, m_range, reverse_m_dict):
    img = visualize_image(im_input, im_output, pred, pred_motion, gt_motion, disappear, m_range, reverse_m_dict)
    # plt.figure(1)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    img = img * 255.0
    img = img.astype(numpy.uint8)
    img = Image.fromarray(img)
    img.save('tmp.png')


def visualize_image(im_input, im_output, pred, pred_motion, gt_motion, disappear, m_range, reverse_m_dict):
    channel, im_height, im_width = im_output.size(1), im_output.size(2), im_output.size(3)
    width, height = get_img_size(4, 5, im_width, im_height)
    img = numpy.ones((height, width, 3))

    im1 = im_input[0, -channel*2:-channel, :, :].cpu().data.numpy().transpose(1, 2, 0)
    x1, y1, x2, y2 = get_img_coordinate(1, 1, im_width, im_height)
    img[y1:y2, x1:x2, :] = im1

    im2 = im_input[0, -channel:, :, :].cpu().data.numpy().transpose(1, 2, 0)
    x1, y1, x2, y2 = get_img_coordinate(1, 2, im_width, im_height)
    img[y1:y2, x1:x2, :] = im2

    im_diff = numpy.abs(im1 - im2)
    x1, y1, x2, y2 = get_img_coordinate(2, 1, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    im3 = im_output[0].cpu().data.numpy().transpose(1, 2, 0)
    x1, y1, x2, y2 = get_img_coordinate(1, 3, im_width, im_height)
    img[y1:y2, x1:x2, :] = im3

    im_diff = numpy.abs(im2 - im3)
    x1, y1, x2, y2 = get_img_coordinate(2, 2, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    pred_motion = pred_motion[0].cpu().data.numpy().transpose(1, 2, 0)
    optical_flow = flowlib.visualize_flow(pred_motion, m_range)
    x1, y1, x2, y2 = get_img_coordinate(3, 1, im_width, im_height)
    img[y1:y2, x1:x2, :] = optical_flow / 255.0

    gt_motion = label2motion(gt_motion[0].cpu().data.numpy().squeeze(), reverse_m_dict)
    optical_flow = flowlib.visualize_flow(gt_motion, m_range)
    x1, y1, x2, y2 = get_img_coordinate(3, 2, im_width, im_height)
    img[y1:y2, x1:x2, :] = optical_flow / 255.0

    disappear = disappear[0].cpu().data.numpy().squeeze()
    cmap = plt.get_cmap('jet')
    disappear = cmap(disappear)[:, :, 0:3]
    x1, y1, x2, y2 = get_img_coordinate(3, 4, im_width, im_height)
    img[y1:y2, x1:x2, :] = disappear

    pred = pred[0].cpu().data.numpy().transpose(1, 2, 0)
    pred[pred > 1] = 1
    pred[pred < 0] = 0
    x1, y1, x2, y2 = get_img_coordinate(3, 3, im_width, im_height)
    img[y1:y2, x1:x2, :] = pred

    im_diff = numpy.abs(pred - im3)
    x1, y1, x2, y2 = get_img_coordinate(4, 3, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    return img


def visualize_bidirection(im_input_f, im_input_b, im_output, pred, pred_motion_f, gt_motion_f, disappear_f, attn_f, pred_motion_b, gt_motion_b, disappear_b, attn_b, m_range, reverse_m_dict):
    img = visualize_image_bidirection(im_input_f, im_input_b, im_output, pred, pred_motion_f, gt_motion_f, disappear_f, attn_f, pred_motion_b, gt_motion_b, disappear_b, attn_b, m_range, reverse_m_dict)
    # plt.figure(1)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    img = img * 255.0
    img = img.astype(numpy.uint8)
    img = Image.fromarray(img)
    img.save('tmp_bidirection.png')


def visualize_image_bidirection(im_input_f, im_input_b, im_output, pred, pred_motion_f, gt_motion_f, disappear_f, attn_f, pred_motion_b, gt_motion_b, disappear_b, attn_b, m_range, reverse_m_dict):
    channel, im_height, im_width = im_output.size(1), im_output.size(2), im_output.size(3)
    width, height = get_img_size(4, 5, im_width, im_height)
    img = numpy.ones((height, width, 3))

    im1 = im_input_f[0, -channel*2:-channel, :, :].cpu().data.numpy().transpose(1, 2, 0)
    x1, y1, x2, y2 = get_img_coordinate(1, 1, im_width, im_height)
    img[y1:y2, x1:x2, :] = im1

    im2 = im_input_f[0, -channel:, :, :].cpu().data.numpy().transpose(1, 2, 0)
    x1, y1, x2, y2 = get_img_coordinate(1, 2, im_width, im_height)
    img[y1:y2, x1:x2, :] = im2

    im_diff = numpy.abs(im1 - im2)
    x1, y1, x2, y2 = get_img_coordinate(2, 1, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    im3 = im_output[0].cpu().data.numpy().transpose(1, 2, 0)
    x1, y1, x2, y2 = get_img_coordinate(1, 3, im_width, im_height)
    img[y1:y2, x1:x2, :] = im3

    im_diff = numpy.abs(im2 - im3)
    x1, y1, x2, y2 = get_img_coordinate(2, 2, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    im4 = im_input_b[0, -channel:, :, :].cpu().data.numpy().transpose(1, 2, 0)
    x1, y1, x2, y2 = get_img_coordinate(1, 4, im_width, im_height)
    img[y1:y2, x1:x2, :] = im4

    im_diff = numpy.abs(im4 - im3)
    x1, y1, x2, y2 = get_img_coordinate(2, 4, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    im5 = im_input_b[0, -channel*2:-channel, :, :].cpu().data.numpy().transpose(1, 2, 0)
    x1, y1, x2, y2 = get_img_coordinate(1, 5, im_width, im_height)
    img[y1:y2, x1:x2, :] = im5

    im_diff = numpy.abs(im5 - im4)
    x1, y1, x2, y2 = get_img_coordinate(2, 5, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    pred_motion = pred_motion_f[0].cpu().data.numpy().transpose(1, 2, 0)
    optical_flow = flowlib.visualize_flow(pred_motion, m_range)
    x1, y1, x2, y2 = get_img_coordinate(3, 1, im_width, im_height)
    img[y1:y2, x1:x2, :] = optical_flow / 255.0

    gt_motion = label2motion(gt_motion_f[0].cpu().data.numpy().squeeze(), reverse_m_dict)
    optical_flow = flowlib.visualize_flow(gt_motion, m_range)
    x1, y1, x2, y2 = get_img_coordinate(3, 2, im_width, im_height)
    img[y1:y2, x1:x2, :] = optical_flow / 255.0

    disappear = disappear_f[0].cpu().data.numpy().squeeze()
    cmap = plt.get_cmap('jet')
    disappear = cmap(disappear)[:, :, 0:3]
    x1, y1, x2, y2 = get_img_coordinate(3, 4, im_width, im_height)
    img[y1:y2, x1:x2, :] = disappear

    attn = attn_f[0].cpu().data.numpy().squeeze()
    cmap = plt.get_cmap('jet')
    attn = cmap(attn)[:, :, 0:3]
    x1, y1, x2, y2 = get_img_coordinate(3, 5, im_width, im_height)
    img[y1:y2, x1:x2, :] = attn

    pred_motion = pred_motion_b[0].cpu().data.numpy().transpose(1, 2, 0)
    optical_flow = flowlib.visualize_flow(pred_motion, m_range)
    x1, y1, x2, y2 = get_img_coordinate(4, 1, im_width, im_height)
    img[y1:y2, x1:x2, :] = optical_flow / 255.0

    gt_motion = label2motion(gt_motion_b[0].cpu().data.numpy().squeeze(), reverse_m_dict)
    optical_flow = flowlib.visualize_flow(gt_motion, m_range)
    x1, y1, x2, y2 = get_img_coordinate(4, 2, im_width, im_height)
    img[y1:y2, x1:x2, :] = optical_flow / 255.0

    disappear = disappear_b[0].cpu().data.numpy().squeeze()
    cmap = plt.get_cmap('jet')
    disappear = cmap(disappear)[:, :, 0:3]
    x1, y1, x2, y2 = get_img_coordinate(4, 4, im_width, im_height)
    img[y1:y2, x1:x2, :] = disappear

    attn = attn_b[0].cpu().data.numpy().squeeze()
    cmap = plt.get_cmap('jet')
    attn = cmap(attn)[:, :, 0:3]
    x1, y1, x2, y2 = get_img_coordinate(4, 5, im_width, im_height)
    img[y1:y2, x1:x2, :] = attn

    pred = pred[0].cpu().data.numpy().transpose(1, 2, 0)
    pred[pred > 1] = 1
    pred[pred < 0] = 0
    x1, y1, x2, y2 = get_img_coordinate(3, 3, im_width, im_height)
    img[y1:y2, x1:x2, :] = pred

    im_diff = numpy.abs(pred - im3)
    x1, y1, x2, y2 = get_img_coordinate(4, 3, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    return img


def get_img_size(n_row, n_col, im_width, im_height):
    height = n_row * im_height + (n_row - 1) * int(im_height/10)
    width = n_col * im_width + (n_col - 1) * int(im_width/10)
    return width, height


def get_img_coordinate(row, col, im_width, im_height):
    y1 = (row - 1) * im_height + (row - 1) * int(im_height/10)
    y2 = y1 + im_height
    x1 = (col - 1) * im_width + (col - 1) * int(im_width/10)
    x2 = x1 + im_width
    return x1, y1, x2, y2


def label2motion(motion_label, reverse_m_dict):
    motion = numpy.zeros((motion_label.shape[0], motion_label.shape[1], 2))
    for i in range(motion_label.shape[0]):
        for j in range(motion_label.shape[1]):
            motion[i, j, :] = numpy.asarray(reverse_m_dict[motion_label[i, j]])
    return motion

