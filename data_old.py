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


def motion_dict(m_range):
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


def load_mnist(file_name='./mnist.h5'):
    script_dir = os.path.dirname(__file__)  # absolute dir the script is in
    f = h5py.File(os.path.join(script_dir, file_name))
    train_images = f['train'].value.reshape(-1, 28, 28)
    train_images = numpy.expand_dims(train_images, 1).repeat(3, 1)
    test_images = f['test'].value.reshape(-1, 28, 28)
    test_images = numpy.expand_dims(test_images, 1).repeat(3, 1)
    return train_images, test_images


def generate_images(args, train_images, m_dict, reverse_m_dict):
    batch_size, num_objects, im_size, m_range = \
        args.batch_size, args.num_objects, args.image_size, args.motion_range
    # generate foreground
    if args.data == 'box':
        im3 = generate_box(args, batch_size, num_objects, im_size)
    elif args.data == 'mnist':
        im3 = generate_mnist(train_images, batch_size, num_objects, im_size)
    # generate foreground motion
    m_f_x, m_f_y, m_mask_f, m_b_x, m_b_y, m_mask_b = \
        generate_motion(batch_size, num_objects, im_size, m_dict, reverse_m_dict)
    # move foreground
    im2 = move_image_fg(im3, m_b_x, m_b_y, m_range)
    im1 = move_image_fg(im2, m_b_x, m_b_y, m_range)
    im4 = move_image_fg(im3, m_f_x, m_f_y, m_range)
    im5 = move_image_fg(im4, m_f_x, m_f_y, m_range)
    # generate background
    bg3 = numpy.random.rand(batch_size, 3, im_size, im_size) * args.bg_noise
    if args.bg_move:
        # generate background motion
        m_f_x, m_f_y, bg_mask_f, m_b_x, m_b_y, bg_mask_b = \
            generate_motion(batch_size, 1, im_size, im_size, m_dict, reverse_m_dict)
        bg2 = move_image_bg(args, bg3, m_b_x, m_b_y, m_range)
        bg1 = move_image_bg(args, bg2, m_b_x, m_b_y, m_range)
        bg4 = move_image_bg(args, bg3, m_f_x, m_f_y, m_range)
        bg5 = move_image_bg(args, bg4, m_f_x, m_f_y, m_range)
    else:
        bg1, bg2, bg4, bg5 = bg3, bg3, bg3, bg3
        bg_mask_f = m_dict[(0, 0)] * numpy.ones(batch_size, 1, im_size, im_size)
        bg_mask_b = m_dict[(0, 0)] * numpy.ones(batch_size, 1, im_size, im_size)
    # generate final image and ground truth motion
    final_im1 = merge_objects(im1)

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
    im_input_f = numpy.concatenate((final_im1, final_im2), 1)
    im_input_b = numpy.concatenate((final_im5, final_im4), 1)
    im_output = final_im3
    if args.bidirection:
        return im_input_f, im_input_b, im_output, gt_motion_f.astype(int), gt_motion_b.astype(int)
    else:
        return im_input_f, im_output, gt_motion_f.astype(int)


def generate_box(args, batch_size, num_objects, im_size):
    im = numpy.zeros((batch_size, num_objects, 3, im_size, im_size))
    for i in range(batch_size):
        for j in range(num_objects):
            width = numpy.random.randint(im_size/8, im_size*3/4)
            height = numpy.random.randint(im_size/8, im_size*3/4)
            x = numpy.random.randint(0, im_size - width)
            y = numpy.random.randint(0, im_size - height)
            color = numpy.random.uniform(args.bg_noise, 1 - args.fg_noise, 3)
            for k in range(3):
                im[i, j, k, y:y+height, x:x+width] = color[k]
            noise = numpy.random.rand(3, height, width) * args.fg_noise
            im[i, j, :, y:y+height, x:x+width] = im[i, j, :, y:y+height, x:x+width] + noise
    return im


def generate_mnist(images, batch_size, im_size):
    idx = numpy.random.permutation(images.shape[0])
    mnist_im = images[idx[0:batch_size], :, :, :]
    im = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        width = mnist_im.shape[3]
        height = mnist_im.shape[2]
        x = numpy.random.randint(0, im_size - width)
        y = numpy.random.randint(0, im_size - height)
        im[i, :, y:y+height, x:x+width] = mnist_im[i, :, :, :]
    return im


def generate_motion(batch_size, num_objects, im_size, m_dict, reverse_m_dict):
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


def move_image_fg(im, m_x, m_y, m_range):
    [batch_size, num_objects, num_channel, im_size, _] = im.shape
    im_big = numpy.zeros(
        (batch_size, num_objects, num_channel, im_size + m_range * 2, im_size + m_range * 2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im
    im_new = numpy.zeros((batch_size, num_objects, num_channel, im_size, im_size))
    for i in range(batch_size):
        for j in range(num_objects):
            y = m_range + m_y[i, j]
            x = m_range + m_x[i, j]
            im_new[i, j, :, :, :] = im_big[i, j, :, y:y + im_size, x:x + im_size]
    return im_new


def move_image_bg(args, im, m_x, m_y, m_range):
    [batch_size, _, im_size, _] = im.shape
    im_big = numpy.random.rand(batch_size, 3, im_size + m_range * 2, im_size + m_range * 2) * args.bg_noise
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im
    im_new = numpy.zeros((batch_size, 3, im_size, im_size))
    for i in range(batch_size):
            y = m_range + m_y[i]
            x = m_range + m_x[i]
            im_new[i, :, :, :] = im_big[i, :, y:y + im_size, x:x + im_size]
    return im_new


def display(args, images1, images2, images3, images4=None, images5=None):
    im_width, im_height = args.image_size, args.image_size
    width, height = visualize.get_img_size(2, 5, im_width, im_height)
    img = numpy.ones((height, width, 3))

    im1 = images1[0, :, :, :].squeeze().transpose(1, 2, 0)
    x1, y1, x2, y2 = visualize.get_img_coordinate(1, 1, im_width, im_height)
    img[y1:y2, x1:x2, :] = im1

    im2 = images2[0, :, :, :].squeeze().transpose(1, 2, 0)
    x1, y1, x2, y2 = visualize.get_img_coordinate(1, 2, im_width, im_height)
    img[y1:y2, x1:x2, :] = im2

    im_diff = abs(im2 - im1)
    x1, y1, x2, y2 = visualize.get_img_coordinate(2, 2, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    im3 = images3[0, :, :, :].squeeze().transpose(1, 2, 0)
    x1, y1, x2, y2 = visualize.get_img_coordinate(1, 3, im_width, im_height)
    img[y1:y2, x1:x2, :] = im3

    im_diff = abs(im3 - im2)
    x1, y1, x2, y2 = visualize.get_img_coordinate(2, 3, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    im4 = images4[0, :, :, :].squeeze().transpose(1, 2, 0)
    x1, y1, x2, y2 = visualize.get_img_coordinate(1, 4, im_width, im_height)
    img[y1:y2, x1:x2, :] = im4

    im_diff = abs(im4 - im3)
    x1, y1, x2, y2 = visualize.get_img_coordinate(2, 4, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    im5 = images5[0, :, :, :].squeeze().transpose(1, 2, 0)
    x1, y1, x2, y2 = visualize.get_img_coordinate(1, 5, im_width, im_height)
    img[y1:y2, x1:x2, :] = im5

    im_diff = abs(im5 - im4)
    x1, y1, x2, y2 = visualize.get_img_coordinate(2, 5, im_width, im_height)
    img[y1:y2, x1:x2, :] = im_diff

    if False:
        plt.figure(1)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        img = img * 255.0
        img = img.astype(numpy.uint8)
        img = Image.fromarray(img)
        img.save('tmp.png')


def unit_test():
    m_dict, reverse_m_dict, m_kernel = motion_dict(1)
    args = learning_args.parse_args()
    if args.data == 'mnist':
        train_images, test_images = load_mnist()
    else:
        train_images, test_images = None, None
    logging.info(args)
    if args.bidirection:
        im_input_f, im_input_b, im_output, gt_motion_f, gt_motion_b = \
            generate_images(args, train_images, m_dict, reverse_m_dict)
        im1 = im_input_f[:, -6:-3, :, :]
        im2 = im_input_f[:, -3:, :, :]
        im3 = im_output
        im4 = im_input_b[:, -3:, :, :]
        im5 = im_input_b[:, -6:-3, :, :]
        display(im1, im2, im3, im4, im5)
    else:
        im_input, im_output, gt_motion = generate_images(args, train_images, m_dict, reverse_m_dict)
        im1 = im_input[:, -6:-3, :, :]
        im2 = im_input[:, -3:, :, :]
        im3 = im_output
        display(im1, im2, im3)

if __name__ == '__main__':
    unit_test()
