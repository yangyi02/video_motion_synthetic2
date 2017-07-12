import os
import numpy
import matplotlib.pyplot as plt
from PIL import Image

from visualize.base_visualizer import BaseVisualizer
from visualize import flowlib
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class SyntheticData(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_objects = args.num_objects
        self.im_size = args.image_size
        self.m_range = args.motion_range
        self.num_frame = args.num_frame
        self.bg_move = args.bg_move
        self.bg_noise = args.bg_noise
        self.m_dict, self.reverse_m_dict, self.m_kernel = self.motion_dict()
        self.visualizer = BaseVisualizer(args, self.reverse_m_dict)
        self.save_display = args.save_display
        self.save_display_dir = args.save_display_dir

    def motion_dict(self):
        m_range = self.m_range
        m_dict, reverse_m_dict = {}, {}
        x = numpy.linspace(-m_range, m_range, 2 * m_range + 1)
        y = numpy.linspace(-m_range, m_range, 2 * m_range + 1)
        m_x, m_y = numpy.meshgrid(x, y)
        m_x, m_y, = m_x.reshape(-1).astype(int), m_y.reshape(-1).astype(int)
        m_kernel = numpy.zeros((1, len(m_x), 2 * m_range + 1, 2 * m_range + 1))
        for i in range(len(m_x)):
            m_dict[(m_x[i], m_y[i])] = i
            reverse_m_dict[i] = (m_x[i], m_y[i])
            m_kernel[:, i, m_y[i] + m_range, m_x[i] + m_range] = 1
        return m_dict, reverse_m_dict, m_kernel

    def generate_data(self, src_image, src_mask):
        batch_size, im_size = self.batch_size, self.im_size
        m_dict = self.m_dict
        # generate foreground motion
        src_motion, src_motion_label, m_x, m_y = self.generate_motion(self.num_objects, src_mask)
        # move foreground
        fg_im, fg_motion, fg_motion_label, fg_mask = \
            self.move_foreground(src_image, src_motion, src_motion_label, src_mask, m_x, m_y)
        # generate background
        src_bg = numpy.random.rand(batch_size, 3, im_size, im_size) * self.bg_noise
        # generate background motion
        if self.bg_move:
            bg_motion, bg_motion_label, m_x, m_y = self.generate_motion(num_objects=1)
        else:
            bg_motion = numpy.zeros((1, batch_size, 2, im_size, im_size))
            bg_motion_label = m_dict[(0, 0)] * numpy.ones((1, batch_size, 1, im_size, im_size))
            m_x = numpy.zeros((1, batch_size)).astype(int)
            m_y = numpy.zeros((1, batch_size)).astype(int)
        # move background
        bg_im, bg_motion, bg_motion_label = \
            self.move_background(src_bg, bg_motion, bg_motion_label, m_x, m_y)
        # merge foreground and background, merge foreground motion and background motion
        mask = fg_mask == 0
        fg_motion_label[mask] = bg_motion_label[mask]
        motion_mask = numpy.concatenate((mask, mask), 2)
        fg_motion[motion_mask] = bg_motion[motion_mask]
        im_mask = numpy.concatenate((mask, mask, mask), 2)
        fg_im[im_mask] = bg_im[im_mask]
        # swap axes between bacth size and frame
        im, motion, motion_label = \
            fg_im.swapaxes(0, 1), fg_motion.swapaxes(0, 1), fg_motion_label.swapaxes(0, 1)
        return im, motion.astype(int), motion_label.astype(int)

    def generate_motion(self, num_objects, src_mask=None):
        batch_size, im_size = self.batch_size, self.im_size
        m_dict, reverse_m_dict = self.m_dict, self.reverse_m_dict
        m_label = numpy.random.randint(0, len(m_dict), size=(num_objects, batch_size))
        m_x = numpy.zeros((num_objects, batch_size)).astype(int)
        m_y = numpy.zeros((num_objects, batch_size)).astype(int)
        for i in range(num_objects):
            for j in range(batch_size):
                (m_x[i, j], m_y[i, j]) = reverse_m_dict[m_label[i, j]]
        src_motion = numpy.zeros((num_objects, batch_size, 2, im_size, im_size))
        src_motion_label = \
            m_dict[(0, 0)] * numpy.ones((num_objects, batch_size, 1, im_size, im_size))
        if src_mask is None:
            src_mask = numpy.ones((num_objects, batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                mask = src_mask[i, j, 0, :, :] > 0
                src_motion[i, j, 0, mask] = m_x[i, j]
                src_motion[i, j, 1, mask] = m_y[i, j]
                src_motion_label[i, j, 0, mask] = m_label[i, j]
        return src_motion, src_motion_label, m_x, m_y

    def move_foreground(self, src_image, src_motion, src_motion_label, src_mask, m_x, m_y):
        batch_size, num_frame, im_size = self.batch_size, self.num_frame, self.im_size
        im = numpy.zeros((num_frame, batch_size, 3, im_size, im_size))
        motion = numpy.zeros((num_frame, batch_size, 2, im_size, im_size))
        motion_label = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        mask = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        for i in range(num_frame):
            im[i, ...], motion[i, ...], motion_label[i, ...], mask[i, ...] = \
                self.merge_objects(src_image, src_motion, src_motion_label, src_mask)
            src_image = self.move_image_fg(src_image, m_x, m_y)
            src_motion = self.move_motion(src_motion, m_x, m_y)
            src_motion_label = self.move_motion_label(src_motion_label, m_x, m_y)
            src_mask = self.move_mask(src_mask, m_x, m_y)
        return im, motion, motion_label, mask

    def merge_objects(self, src_image, src_motion, src_motion_label, src_mask):
        batch_size, num_objects, im_size = self.batch_size, self.num_objects, self.im_size
        im = numpy.zeros((batch_size, 3, im_size, im_size))
        motion = numpy.zeros((batch_size, 2, im_size, im_size))
        motion_label = numpy.zeros((batch_size, 1, im_size, im_size))
        mask = numpy.zeros((batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            zero_mask = mask == 0
            zero_motion_mask = numpy.concatenate((zero_mask, zero_mask), 1)
            zero_im_mask = numpy.concatenate((zero_mask, zero_mask, zero_mask), 1)
            im[zero_im_mask] = src_image[i, zero_im_mask]
            motion[zero_motion_mask] = src_motion[i, zero_motion_mask]
            motion_label[zero_mask] = src_motion_label[i, zero_mask]
            mask[zero_mask] = src_mask[i, zero_mask]
        return im, motion, motion_label, mask

    def move_image_fg(self, im, m_x, m_y):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        num_objects = self.num_objects
        im_big = numpy.zeros(
            (num_objects, batch_size, 3, im_size + m_range * 2, im_size + m_range * 2))
        im_big[:, :, :, m_range:-m_range, m_range:-m_range] = im
        im_new = numpy.zeros((num_objects, batch_size, 3, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                x = m_range + m_x[i, j]
                y = m_range + m_y[i, j]
                im_new[i, j, :, :, :] = im_big[i, j, :, y:y + im_size, x:x + im_size]
        return im_new

    def move_motion(self, motion, m_x, m_y):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        num_objects = self.num_objects
        m_dict = self.m_dict
        motion_big = m_dict[(0, 0)] * numpy.ones(
            (num_objects, batch_size, 2, im_size + m_range * 2, im_size + m_range * 2))
        motion_big[:, :, :, m_range:-m_range, m_range:-m_range] = motion
        motion_new = numpy.zeros((num_objects, batch_size, 2, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                x = m_range + m_x[i, j]
                y = m_range + m_y[i, j]
                motion_new[i, j, :, :, :] = motion_big[i, j, :, y:y + im_size, x:x + im_size]
        return motion_new

    def move_motion_label(self, motion_label, m_x, m_y):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        num_objects = self.num_objects
        motion_label_big = numpy.zeros(
            (num_objects, batch_size, 1, im_size + m_range * 2, im_size + m_range * 2))
        motion_label_big[:, :, :, m_range:-m_range, m_range:-m_range] = motion_label
        motion_label_new = numpy.zeros((num_objects, batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                x = m_range + m_x[i, j]
                y = m_range + m_y[i, j]
                motion_label_new[i, j, :, :, :] = \
                    motion_label_big[i, j, :, y:y + im_size, x:x + im_size]
        return motion_label_new

    def move_mask(self, mask, m_x, m_y):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        num_objects = self.num_objects
        mask_big = numpy.zeros(
            (num_objects, batch_size, 1, im_size + m_range * 2, im_size + m_range * 2))
        mask_big[:, :, :, m_range:-m_range, m_range:-m_range] = mask
        mask_new = numpy.zeros((num_objects, batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                x = m_range + m_x[i, j]
                y = m_range + m_y[i, j]
                mask_new[i, j, :, :, :] = mask_big[i, j, :, y:y + im_size, x:x + im_size]
        return mask_new

    def move_background(self, src_image, src_motion, src_motion_label, m_x, m_y):
        batch_size, num_frame, im_size = self.batch_size, self.num_frame, self.im_size
        im = numpy.zeros((num_frame, batch_size, 3, im_size, im_size))
        motion = numpy.zeros((num_frame, batch_size, 2, im_size, im_size))
        motion_label = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        for i in range(num_frame):
            im[i, :, :, :, :] = src_image
            src_image = self.move_image_bg(src_image, m_x, m_y)
            motion[i, :, :, :, :] = src_motion
            motion_label[i, :, :, :, :] = src_motion_label
        return im, motion, motion_label

    def move_image_bg(self, bg_im, m_x, m_y):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        bg_noise = self.bg_noise
        im_big = numpy.random.rand(batch_size, 3, im_size + m_range * 2,
                                   im_size + m_range * 2) * bg_noise
        im_big[:, :, m_range:-m_range, m_range:-m_range] = bg_im
        im_new = numpy.zeros((batch_size, 3, im_size, im_size))
        for i in range(batch_size):
            x = m_range + m_x[0, i]
            y = m_range + m_y[0, i]
            im_new[i, :, :, :] = im_big[i, :, y:y + im_size, x:x + im_size]
        return im_new

    def display(self, im, motion):
        num_frame = self.num_frame
        width, height = self.visualizer.get_img_size(3, num_frame)
        img = numpy.ones((height, width, 3))
        prev_im = None
        for i in range(num_frame):
            curr_im = im[0, i, :, :, :].squeeze().transpose(1, 2, 0)
            x1, y1, x2, y2 = self.visualizer.get_img_coordinate(1, i + 1)
            img[y1:y2, x1:x2, :] = curr_im

            if i > 0:
                im_diff = abs(curr_im - prev_im)
                x1, y1, x2, y2 = self.visualizer.get_img_coordinate(2, i + 1)
                img[y1:y2, x1:x2, :] = im_diff
            prev_im = curr_im

            flow = motion[0, i, :, :, :].squeeze().transpose(1, 2, 0)
            optical_flow = flowlib.visualize_flow(flow, self.m_range)
            x1, y1, x2, y2 = self.visualizer.get_img_coordinate(3, i + 1)
            img[y1:y2, x1:x2, :] = optical_flow / 255.0

        if self.save_display:
            img = img * 255.0
            img = img.astype(numpy.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(self.save_display_dir, 'data.png'))
        else:
            plt.figure(1)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
