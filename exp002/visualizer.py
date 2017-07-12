import os
import numpy
import matplotlib.pyplot as plt
from PIL import Image

from visualize.base_visualizer import BaseVisualizer
from visualize import flowlib


class Visualizer(BaseVisualizer):
    def __init__(self, args, reverse_m_dict):
        super(Visualizer, self).__init__(args, reverse_m_dict)

    def visualize_result(self, im_input, im_output, im_pred, pred_motion, gt_motion, disappear, appear, file_name='tmp.png'):
        width, height = self.get_img_size(3, max(self.num_frame + 1, 4))
        img = numpy.ones((height, width, 3))
        prev_im = None
        for i in range(self.num_frame - 1):
            curr_im = im_input[0, i*3:(i+1)*3, :, :].cpu().data.numpy().transpose(1, 2, 0)
            x1, y1, x2, y2 = self.get_img_coordinate(1, i+1)
            img[y1:y2, x1:x2, :] = curr_im

            if i > 0:
                im_diff = abs(curr_im - prev_im)
                x1, y1, x2, y2 = self.get_img_coordinate(2, i+1)
                img[y1:y2, x1:x2, :] = im_diff
            prev_im = curr_im

        im_output = im_output[0].cpu().data.numpy().transpose(1, 2, 0)
        x1, y1, x2, y2 = self.get_img_coordinate(1, self.num_frame)
        img[y1:y2, x1:x2, :] = im_output

        im_diff = numpy.abs(im_output - prev_im)
        x1, y1, x2, y2 = self.get_img_coordinate(2, self.num_frame)
        img[y1:y2, x1:x2, :] = im_diff

        pred = im_pred[0].cpu().data.numpy().transpose(1, 2, 0)
        x1, y1, x2, y2 = self.get_img_coordinate(1, self.num_frame + 1)
        img[y1:y2, x1:x2, :] = pred

        im_diff = numpy.abs(pred - im_output)
        x1, y1, x2, y2 = self.get_img_coordinate(2, self.num_frame + 1)
        img[y1:y2, x1:x2, :] = im_diff

        pred_motion = pred_motion[0].cpu().data.numpy().transpose(1, 2, 0)
        optical_flow = flowlib.visualize_flow(pred_motion, self.m_range)
        x1, y1, x2, y2 = self.get_img_coordinate(3, 1)
        img[y1:y2, x1:x2, :] = optical_flow / 255.0

        gt_motion = gt_motion[0].cpu().data.numpy().transpose(1, 2, 0)
        optical_flow = flowlib.visualize_flow(gt_motion, self.m_range)
        x1, y1, x2, y2 = self.get_img_coordinate(3, 2)
        img[y1:y2, x1:x2, :] = optical_flow / 255.0

        disappear = disappear[0].cpu().data.numpy().squeeze()
        cmap = plt.get_cmap('jet')
        disappear = cmap(disappear)[:, :, 0:3]
        x1, y1, x2, y2 = self.get_img_coordinate(3, 3)
        img[y1:y2, x1:x2, :] = disappear

        appear = appear[0].cpu().data.numpy().squeeze()
        cmap = plt.get_cmap('jet')
        appear = cmap(appear)[:, :, 0:3]
        x1, y1, x2, y2 = self.get_img_coordinate(3, 4)
        img[y1:y2, x1:x2, :] = appear

        if self.save_display:
            img = img * 255.0
            img = img.astype(numpy.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(self.save_display_dir, file_name))
        else:
            plt.figure(1)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

