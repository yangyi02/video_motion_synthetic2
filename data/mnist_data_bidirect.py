import os
import numpy
import h5py

from synthetic_data_bidirect import SyntheticDataBidirect
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class MnistDataBidirect(SyntheticDataBidirect):
    def __init__(self, args):
        super(MnistDataBidirect, self).__init__(args)
        script_dir = os.path.dirname(__file__)  # absolute dir the script is in
        self.file_name = os.path.join(script_dir, 'mnist.h5')
        self.train_images, self.test_images = self.load_mnist()

    def load_mnist(self):
        f = h5py.File(self.file_name)
        train_images = f['train'].value.reshape(-1, 28, 28)
        train_images = numpy.expand_dims(train_images, 1)
        train_images = numpy.concatenate((train_images, train_images, train_images), 1)
        test_images = f['test'].value.reshape(-1, 28, 28)
        test_images = numpy.expand_dims(test_images, 1)
        test_images = numpy.concatenate((test_images, test_images, test_images), 1)
        return train_images, test_images

    def generate_source_image(self, images):
        batch_size, num_objects, im_size = self.batch_size, self.num_objects, self.im_size
        im = numpy.zeros((num_objects, batch_size, 3, im_size, im_size))
        for i in range(num_objects):
            idx = numpy.random.permutation(images.shape[0])
            mnist_im = images[idx[0:batch_size], :, :, :]
            for j in range(batch_size):
                width = mnist_im.shape[3]
                height = mnist_im.shape[2]
                x = numpy.random.randint(0, im_size - width)
                y = numpy.random.randint(0, im_size - height)
                im[i, j, :, y:y+height, x:x+width] = mnist_im[j, :, :, :]
        mask = numpy.expand_dims(im.sum(2) > 0, 2)
        return im, mask

    def get_next_batch(self, images):
        src_image, src_mask = self.generate_source_image(images)
        im, motion, motion_r = self.generate_data(src_image, src_mask)
        return im, motion, motion_r


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = MnistDataBidirect(args)
    im, motion, motion_r = data.get_next_batch(data.train_images)
    data.display(im, motion, motion_r)

if __name__ == '__main__':
    unit_test()
