import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import numpy as np
from util.util import tensor2im
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = (
        True  # no flip; comment this line if results on flipped images are needed.
    )
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(
        opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch)
    )  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = "{:s}_iter{:d}".format(web_dir, opt.load_iter)
    summary_writer = SummaryWriter(web_dir)
    print("creating web directory", web_dir)
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        show_imgs = []

        for j, (label, image) in enumerate(visuals.items()):
            image_numpy = tensor2im(image)
            show_imgs.append(image_numpy)
        if i % 5 == 0:
            print("processing (%04d)-th image... " % (i))

        label = "-".join(visuals.keys())
        show_imgs = np.stack(show_imgs, axis=0)
        summary_writer.add_images(
            "test_img-%.3d: %s" % (i, label), show_imgs, i, dataformats="NHWC"
        )
        summary_writer.flush()
