import os
import numpy
import matplotlib.pyplot as plt
from skimage import io, transform
import pickle
from PIL import Image
import h5py
import learning_args
import visualize
import logging

logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class SyntheticData(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_objects = args.num_objects
        self.im_size = args.image_size
        self.m_range = args.motion_range
        self.num_inputs = args.num_inputs
        self.bg_move = args.bg_move
        self.bg_noise = args.bg_noise
        self.m_dict, self.reverse_m_dict, self.m_kernel = self.motion_dict()

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

    def generate_data(self, source_image):
        batch_size, im_size, num_inputs = self.batch_size, self.im_size, self.num_inputs
        m_dict = self.m_dict
        # generate foreground motion
        m_label, m_x, m_y = self.generate_motion(self.num_objects)
        # move foreground
        fg_im, fg_motion = self.move_foreground(source_image, m_label, m_x, m_y)
        # generate background
        source_bg = numpy.random.rand(batch_size, 3, im_size, im_size) * self.bg_noise
        # generate background motion
        if self.bg_move:
            # move background
            m_x, m_y = self.generate_motion(num_objects=1)
            bg_im, bg_motion = self.move_background(source_bg, m_x, m_y)
        else:
            # does not move background
            bg_motion = m_dict[(0, 0)] * numpy.ones(num_inputs + 1, batch_size, 1, im_size, im_size)
            bg_im = numpy.zeros((num_inputs + 1, batch_size, 3, im_size, im_size))
            for i in range(self.num_inputs):
                bg_im[i, :, :, :, :] = source_bg
        # merge foreground and background, merge foreground motion and background motion
        for i in range(self.num_inputs):
            mask = numpy.expand_dims(fg_im[i, :, :, :, :].sum(2), 1) == 0
            fg_motion[i, mask] = bg_motion[i, mask]
            fg_im[i, mask] = bg_im[i, mask]
        im_input = fg_im[:-1, :, :, :, :].reshape()
        im_output = fg_im[-1, :, :, :, :]
        gt_motion = fg_motion[-1, :, :, :, :]
        return im_input, im_output, gt_motion.astype(int)

    def generate_motion(self, num_objects):
        batch_size, im_size = self.batch_size, self.im_size
        m_dict, reverse_m_dict = self.m_dict, self.reverse_m_dict
        m_label = numpy.random.randint(0, len(m_dict), size=(batch_size, num_objects))
        m_x = numpy.zeros((batch_size, num_objects)).astype(int)
        m_y = numpy.zeros((batch_size, num_objects)).astype(int)
        for i in range(batch_size):
            for j in range(num_objects):
                (m_x[i, j], m_y[i, j]) = reverse_m_dict[m_label[i, j]]
        return m_label, m_x, m_y

    def move_foreground(self, source_image, m_label, m_x, m_y):
        batch_size, num_inputs, im_size = self.batch_size, self.num_inputs, self.im_size
        im = numpy.zeros(num_inputs + 1, batch_size, 3, im_size, im_size)
        motion = numpy.zeros(num_inputs + 1, batch_size, 1, im_size, im_size)
        for i in range(num_inputs):
            im[i, :, :, :, :] = self.merge_objects(source_image)
            source_image = move_source_image(source_image, m_x, m_y)
            motion[i, :, :, :, :] = self.merge_motion(m_label)
        return im, motion


    def generate_bidirectional_data(self, im3):
        # generate foreground motion
        m_f_x, m_f_y, m_mask_f, m_b_x, m_b_y, m_mask_b = self.generate_motion(self.num_objects)
        m_b_x, m_b_y, m_mask_b = self.reverse_motion(m_f_x, m_f_y, m_mask_f)
        # move foreground
        im1, im2, im4, im5 = self.move_foreground(im3, m_f_x, m_f_y, m_b_x, m_b_y)
        # generate background
        bg3 = numpy.random.rand(batch_size, 3, im_size, im_size) * self.bg_noise
        # generate background motion
        if self.bg_move:
            m_f_x, m_f_y, m_mask_f, m_b_x, m_b_y, m_mask_b = self.generate_motion(num_objects=1)
            # move background
            bg1, bg2, bg4, bg5 = self.move_background(bg3, m_f_x, m_f_y, m_b_x, m_b_y)
        else:
            bg_mask_f = m_dict[(0, 0)] * numpy.ones(batch_size, 1, im_size, im_size)
            bg_mask_b = m_dict[(0, 0)] * numpy.ones(batch_size, 1, im_size, im_size)
            bg1, bg2, bg4, bg5 = bg3, bg3, bg3, bg3

        # generate final image and ground truth motion
        final_im1 = self.merge_objects(im1)

        final_im1 = numpy.zeros((batch_size, 3, im_size, im_size))
        final_im2 = numpy.zeros((batch_size, 3, im_size, im_size))
        final_im3 = numpy.zeros((batch_size, 3, im_size, im_size))
        final_im4 = numpy.zeros((batch_size, 3, im_size, im_size))
        final_im5 = numpy.zeros((batch_size, 3, im_size, im_size))
        gt_motion_f = numpy.zeros((batch_size, 1, im_size, im_size))
        gt_motion_b = numpy.zeros((batch_size, 1, im_size, im_size))
        for i in range(batch_size):
            for j in range(num_objects):
                final_im1[i, j, :, :, :] += im1[i, j, :, :, :] * mask

                mask = numpy.expand_dims(im2[i, j, :, :, :].sum(2), 1) == 0
                gt_motion_f[mask] = m_mask_f[i, j, :, :, :] * mask
                mask = numpy.expand_dims(im4[:, j, :, :, :].sum(2), 1) == 0
                gt_motion_b[mask] = m_mask_b[i, j, :, :, :] * mask
        mask = numpy.expand_dims(final_im2.sum(1), 1) == 0
        gt_motion_f[mask] = bg_mask_f[mask]
        mask = numpy.expand_dims(final_im4.sum(1), 1) == 0
        gt_motion_b[mask] = bg_mask_b[mask]
        final_im1[final_im1 == 0] = bg1[final_im1 == 0]
        final_im2[final_im2 == 0] = bg2[final_im2 == 0]
        final_im3[final_im3 == 0] = bg3[final_im3 == 0]
        final_im4[final_im4 == 0] = bg4[final_im4 == 0]
        final_im5[final_im5 == 0] = bg5[final_im5 == 0]

        im_input_f = numpy.concatenate((im1, im2), 1)
        im_input_b = numpy.concatenate((im5, im4), 1)
        im_output = im3
        if self.bidirection:
            return im_input_f, im_input_b, im_output, gt_motion_f.astype(int), gt_motion_b.astype(
                int)
        else:
            return im_input_f, im_output, gt_motion_f.astype(int)
        return final_im1, final_im2, final_im3,

    def generate_motion_old(self, num_objects):
        batch_size, im_size = self.batch_size, self.im_size
        m_dict, reverse_m_dict = self.m_dict, self.reverse_m_dict
        m_label = numpy.random.randint(0, len(m_dict), size=(batch_size, num_objects))
        m_f_x = numpy.zeros((batch_size, num_objects)).astype(int)
        m_f_y = numpy.zeros((batch_size, num_objects)).astype(int)
        m_b_x = numpy.zeros((batch_size, num_objects)).astype(int)
        m_b_y = numpy.zeros((batch_size, num_objects)).astype(int)
        for i in range(batch_size):
            for j in range(num_objects):
                (m_f_x[i, j], m_f_y[i, j]) = reverse_m_dict[m_label[i, j]]
                (m_b_x[i, j], m_b_y[i, j]) = (-m_f_x[i, j], -m_f_y[i, j])
        m_mask_f = numpy.zeros((batch_size, num_objects, im_size, im_size))
        m_mask_b = numpy.zeros((batch_size, num_objects, im_size, im_size))
        for i in range(batch_size):
            for j in range(num_objects):
                m_mask_f[i, j, :, :] = m_label[i, j]
                m_mask_b[i, j, :, :] = m_dict[(m_b_x[i, j], m_b_y[i, j])]
        return m_f_x, m_f_y, m_mask_f, m_b_x, m_b_y, m_mask_b

    def move_foreground_old(self, im3, m_f_x, m_f_y, m_b_x, m_b_y):
        im2 = self.move_image_fg(im3, m_b_x, m_b_y)
        im1 = self.move_image_fg(im2, m_b_x, m_b_y)
        im4 = self.move_image_fg(im3, m_f_x, m_f_y)
        im5 = self.move_image_fg(im4, m_f_x, m_f_y)
        return im1, im2, im4, im5

    def move_image_fg(self, im, m_x, m_y):
        batch_size, num_objects, im_size, m_range = \
            self.batch_size, self.num_objects, self.im_size, self.m_range
        im_big = numpy.zeros(
            (batch_size, num_objects, 3, im_size + m_range * 2, im_size + m_range * 2))
        im_big[:, :, m_range:-m_range, m_range:-m_range] = im
        im_new = numpy.zeros((batch_size, num_objects, 3, im_size, im_size))
        for i in range(batch_size):
            for j in range(num_objects):
                x = m_range + m_x[i, j]
                y = m_range + m_y[i, j]
                im_new[i, j, :, :, :] = im_big[i, j, :, y:y + im_size, x:x + im_size]
        return im_new

    def move_background(self, bg3, m_f_x, m_f_y, m_b_x, m_b_y):
        bg2 = self.move_image_bg(bg3, m_b_x, m_b_y)
        bg1 = self.move_image_bg(bg2, m_b_x, m_b_y)
        bg4 = self.move_image_bg(bg3, m_f_x, m_f_y)
        bg5 = self.move_image_bg(bg4, m_f_x, m_f_y)
        return bg1, bg2, bg4, bg5

    def move_image_bg(self, im, m_x, m_y):
        batch_size, im_size, m_range, bg_noise = \
            self.batch_size, self.im_size, self.m_range, self.bg_noise
        im_big = numpy.random.rand(batch_size, 3, im_size + m_range * 2, im_size + m_range * 2) \
            * bg_noise
        im_big[:, :, m_range:-m_range, m_range:-m_range] = im
        im_new = numpy.zeros((batch_size, 3, im_size, im_size))
        for i in range(batch_size):
            x = m_range + m_x[i]
            y = m_range + m_y[i]
            im_new[i, :, :, :] = im_big[i, :, y:y + im_size, x:x + im_size]
        return im_new

    def merge_objects(im):
        [batch_size, num_objects, _, im_size, _] = im.shape
        final_im = numpy.zeros((batch_size, 3, im_size, im_size))
