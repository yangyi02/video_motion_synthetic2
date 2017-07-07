import numpy
import matplotlib.pyplot as plt
import flowlib
from PIL import Image


class Visualizer(object):
    def __init__(self, args):
        self.reverse_m_dict = args.reverse_m_dict
        self.m_range = args.m_range
        self.im_height = args.image_size
        self.im_width = args.image_size

    def visualize_result(self, im_input, im_output, im_pred, pred_motion, gt_motion, disappear, appear):
        img = self.visualize(im_input, im_output, im_pred, pred_motion, gt_motion, disappear, appear)
        if True:
            plt.figure(1)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            img = img * 255.0
            img = img.astype(numpy.uint8)
            img = Image.fromarray(img)
            img.save('tmp.png')

    def visualize(self, im_input, im_output, im_pred, pred_motion, gt_motion, disappear, appear):
        width, height = self.get_img_size(4, 5)
        img = numpy.ones((height, width, 3))
        prev_im = None
        for i in range(im_input.size(1)):
            curr_im = im_input[0, i, :, :, :].squeeze().transpose(1, 2, 0)
            x1, y1, x2, y2 = self.get_img_coordinate(1, i+1)
            img[y1:y2, x1:x2, :] = curr_im

            if i > 0:
                im_diff = abs(curr_im - prev_im)
                x1, y1, x2, y2 = self.get_img_coordinate(2, i+1)
                img[y1:y2, x1:x2, :] = im_diff
            prev_im = curr_im

        im_output = im_output[0].cpu().data.numpy().transpose(1, 2, 0)
        x1, y1, x2, y2 = self.get_img_coordinate(1, im_input.size(1) + 1)
        img[y1:y2, x1:x2, :] = im_output

        im_diff = numpy.abs(im_output - prev_im)
        x1, y1, x2, y2 = self.get_img_coordinate(2, im_input.size(1) + 1)
        img[y1:y2, x1:x2, :] = im_diff

        pred_motion = pred_motion[0].cpu().data.numpy().transpose(1, 2, 0)
        optical_flow = flowlib.visualize_flow(pred_motion, self.m_range)
        x1, y1, x2, y2 = self.get_img_coordinate(3, 1)
        img[y1:y2, x1:x2, :] = optical_flow / 255.0

        gt_motion = gt_motion[0].cpu().data.numpy().squeeze()
        gt_motion = self.label2motion(gt_motion)
        optical_flow = flowlib.visualize_flow(gt_motion, self.m_range)
        x1, y1, x2, y2 = self.get_img_coordinate(3, 2)
        img[y1:y2, x1:x2, :] = optical_flow / 255.0

        disappear = disappear[0].cpu().data.numpy().squeeze()
        cmap = plt.get_cmap('jet')
        disappear = cmap(disappear)[:, :, 0:3]
        x1, y1, x2, y2 = self.get_img_coordinate(3, 4)
        img[y1:y2, x1:x2, :] = disappear

        appear = appear[0].cpu().data.numpy().squeeze()
        cmap = plt.get_cmap('jet')
        appear = cmap(appear)[:, :, 0:3]
        x1, y1, x2, y2 = self.get_img_coordinate(3, 4)
        img[y1:y2, x1:x2, :] = appear

        pred = im_pred[0].cpu().data.numpy().transpose(1, 2, 0)
        pred[pred > 1] = 1
        pred[pred < 0] = 0
        x1, y1, x2, y2 = self.get_img_coordinate(3, 3)
        img[y1:y2, x1:x2, :] = pred

        im_diff = numpy.abs(pred - im_output)
        x1, y1, x2, y2 = self.get_img_coordinate(4, 3)
        img[y1:y2, x1:x2, :] = im_diff

        return img

    def get_img_size(self, n_row, n_col):
        im_width, im_height = self.im_width, self.im_height
        height = n_row * im_height + (n_row - 1) * int(im_height/10)
        width = n_col * im_width + (n_col - 1) * int(im_width/10)
        return width, height

    def get_img_coordinate(self, row, col):
        im_width, im_height = self.im_width, self.im_height
        y1 = (row - 1) * im_height + (row - 1) * int(im_height/10)
        y2 = y1 + im_height
        x1 = (col - 1) * im_width + (col - 1) * int(im_width/10)
        x2 = x1 + im_width
        return x1, y1, x2, y2

    def label2motion(self, motion_label):
        reverse_m_dict = self.reverse_m_dict
        motion = numpy.zeros((motion_label.shape[0], motion_label.shape[1], 2))
        for i in range(motion_label.shape[0]):
            for j in range(motion_label.shape[1]):
                motion[i, j, :] = numpy.asarray(reverse_m_dict[motion_label[i, j]])
        return motion
