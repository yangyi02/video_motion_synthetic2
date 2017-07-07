from visualize import get_img_size, get_img_coordinate, label2motion
from flowlib import visualize_flow

import numpy
import matplotlib.pyplot as plt
from PIL import Image
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
        batch_size, im_size, num_frame = self.batch_size, self.im_size, self.num_frame
        m_dict = self.m_dict
        # generate foreground motion
        src_motion, m_x, m_y = self.generate_motion(self.num_objects, src_mask)
        # move foreground
        fg_im, fg_motion, fg_mask = self.move_foreground(src_image, src_mask, src_motion, m_x, m_y)
        # generate background
        src_bg = numpy.random.rand(batch_size, 3, im_size, im_size) * self.bg_noise
        # generate background motion
        if self.bg_move:
            bg_motion, m_x, m_y = self.generate_motion(num_objects=1)
        else:
            bg_motion = m_dict[(0, 0)] * numpy.ones((1, batch_size, 1, im_size, im_size))
            m_x = numpy.zeros((1, batch_size)).astype(int)
            m_y = numpy.zeros((1, batch_size)).astype(int)
        # move background
        bg_im, bg_motion = self.move_background(src_bg, bg_motion, m_x, m_y)
        # merge foreground and background, merge foreground motion and background motion
        fg_motion[fg_mask == 0] = bg_motion[fg_mask == 0]
        fg_im[fg_mask.repeat(3, 2) == 0] = bg_im[fg_mask.repeat(3, 2) == 0]
        return fg_im, fg_motion.astype(int)

    def generate_motion(self, num_objects, src_mask=None):
        batch_size, im_size = self.batch_size, self.im_size
        m_dict, reverse_m_dict = self.m_dict, self.reverse_m_dict
        m_label = numpy.random.randint(0, len(m_dict), size=(num_objects, batch_size))
        src_motion = m_dict[(0, 0)] * numpy.ones((num_objects, batch_size, 1, im_size, im_size))
        if src_mask is None:
            src_mask = numpy.ones((num_objects, batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                mask = src_mask[i, j, :, :, :] > 0
                src_motion[i, j, mask] = m_label[i, j]
        m_x = numpy.zeros((num_objects, batch_size)).astype(int)
        m_y = numpy.zeros((num_objects, batch_size)).astype(int)
        for i in range(num_objects):
            for j in range(batch_size):
                (m_x[i, j], m_y[i, j]) = reverse_m_dict[m_label[i, j]]
        return src_motion, m_x, m_y

    def move_foreground(self, src_image, src_mask, src_motion, m_x, m_y):
        batch_size, num_frame, im_size = self.batch_size, self.num_frame, self.im_size
        im = numpy.zeros((num_frame, batch_size, 3, im_size, im_size))
        motion = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        mask = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        for i in range(num_frame):
            im[i, ...], motion[i, ...], mask[i, ...] = self.merge_objects(src_image, src_motion,
                                                                          src_mask)
            src_image = self.move_image_fg(src_image, m_x, m_y)
            src_motion = self.move_motion(src_motion, m_x, m_y)
            src_mask = self.move_mask(src_mask, m_x, m_y)
        return im, motion, mask

    def merge_objects(self, src_image, src_motion, src_mask):
        batch_size, num_objects, im_size = self.batch_size, self.num_objects, self.im_size
        im = numpy.zeros((batch_size, 3, im_size, im_size))
        motion = numpy.zeros((batch_size, 1, im_size, im_size))
        mask = numpy.zeros((batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            im[mask.repeat(3, 1) == 0] = src_image[i, mask.repeat(3, 1) == 0]
            motion[mask == 0] = src_motion[i, mask == 0]
            mask[mask == 0] = src_mask[i, mask == 0]
        return im, motion, mask

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
            (num_objects, batch_size, 1, im_size + m_range * 2, im_size + m_range * 2))
        motion_big[:, :, :, m_range:-m_range, m_range:-m_range] = motion
        motion_new = numpy.zeros((num_objects, batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                x = m_range + m_x[i, j]
                y = m_range + m_y[i, j]
                motion_new[i, j, :, :, :] = motion_big[i, j, :, y:y + im_size, x:x + im_size]
        return motion_new

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

    def move_background(self, src_image, src_motion, m_x, m_y):
        batch_size, num_frame, im_size = self.batch_size, self.num_frame, self.im_size
        im = numpy.zeros((num_frame, batch_size, 3, im_size, im_size))
        motion = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        for i in range(num_frame):
            im[i, :, :, :, :] = src_image
            src_image = self.move_image_bg(src_image, m_x, m_y)
            motion[i, :, :, :, :] = src_motion
        return im, motion

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

    def visualize(self, im, motion):
        num_frame, m_range = self.num_frame, self.m_range
        im_width, im_height = self.im_size, self.im_size
        reverse_m_dict = self.reverse_m_dict
        width, height = get_img_size(3, num_frame, im_width, im_height)
        img = numpy.ones((height, width, 3))
        prev_im = None
        for i in range(num_frame):
            curr_im = im[i, 0, :, :, :].squeeze().transpose(1, 2, 0)
            x1, y1, x2, y2 = get_img_coordinate(1, i+1, im_width, im_height)
            img[y1:y2, x1:x2, :] = curr_im

            if i > 0:
                im_diff = abs(curr_im - prev_im)
                x1, y1, x2, y2 = get_img_coordinate(2, i+1, im_width, im_height)
                img[y1:y2, x1:x2, :] = im_diff
            prev_im = curr_im

            flow = label2motion(motion[i, 0, :, :, :].squeeze(), reverse_m_dict)
            optical_flow = visualize_flow(flow, m_range)
            x1, y1, x2, y2 = get_img_coordinate(3, i+1, im_width, im_height)
            img[y1:y2, x1:x2, :] = optical_flow / 255.0

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
