import os
import numpy
import matplotlib.pyplot as plt
from skimage import io, transform
import pickle
from PIL import Image
import h5py
import learning_args
import visualize
from synthetic_data import SyntheticData
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class BoxData(SyntheticData):
    def __init__(self, args):
        super(BoxData, self).__init__(args)
        self.fg_noise = args.fg_noise
        self.bg_noise = args.bg_noise
        self.bidirection = args.bidirection

    def generate_source_image(self):
        batch_size, num_objects, im_size = self.batch_size, self.num_objects, self.im_size
        im = numpy.zeros((num_objects, batch_size, 3, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                width = numpy.random.randint(im_size/8, im_size*3/4)
                height = numpy.random.randint(im_size/8, im_size*3/4)
                x = numpy.random.randint(0, im_size - width)
                y = numpy.random.randint(0, im_size - height)
                color = numpy.random.uniform(self.bg_noise, 1 - self.fg_noise, 3)
                for k in range(3):
                    im[i, j, k, y:y+height, x:x+width] = color[k]
                noise = numpy.random.rand(3, height, width) * self.fg_noise
                im[i, j, :, y:y+height, x:x+width] = im[i, j, :, y:y+height, x:x+width] + noise
        return im

    def get_next_batch(self, images):
        source_image = self.generate_source_image(images)
        if self.bidirection:
            im_input_f, im_input_b, im_output, gt_motion_f, gt_motion_b = self.generate_bidirectional_data(source_image)
            return im_input_f, im_input_b, im_output, gt_motion_f, gt_motion_b
        else:
            im_input, im_output, gt_motion = self.generate_data(source_image)
            return im_input, im_output, gt_motion

