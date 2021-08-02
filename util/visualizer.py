import numpy as np
import os
import time
from . import util
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'tensorboard' for display.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create an SummaryWriter(tensorboard) object for saveing results
        Step 3: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                "================ Training Loss (%s) ================\n" % now
            )

        self.use_tb = True
        if self.use_tb:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, "tb_log")
            self.summary_writer = SummaryWriter(self.log_dir)
            util.mkdirs([self.log_dir])

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on tensorboard.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to tensorboard
        """
        if self.use_tb and save_result:
            show_imgs = []

            for i, (label, image) in enumerate(visuals.items()):
                image_numpy = util.tensor2im(image)
                show_imgs.append(image_numpy)

            label = "-".join(visuals.keys())
            show_imgs = np.stack(show_imgs, axis=0)
            self.summary_writer.add_images(
                "epoch%.3d: %s" % (epoch, label), show_imgs, epoch, dataformats="NHWC"
            )
            self.summary_writer.flush()

    def plot_current_losses(self, epoch, epoch_iter, dataset_size, losses):
        """display the current losses on tensorboard: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        step = epoch * dataset_size + epoch_iter
        for k, v in losses.items():
            self.summary_writer.add_scalar(k, v, step)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = "(epoch: %d, iters: %d, time: %.3f, data: %.3f) " % (
            epoch,
            iters,
            t_comp,
            t_data,
        )
        for k, v in losses.items():
            message += "%s: %.3f " % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)  # save the message
