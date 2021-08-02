import os
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import cv2

CELEBA_SET_SIZE = 202599


def get_img_paths(d):
    all_files = os.listdir(d)
    filter_paths = [f for f in all_files if os.path.splitext(f)[1] in [".jpg", ".png"]]
    filter_paths = sorted(
        filter_paths, key=lambda x: int(x.split(".")[0])
    )  # needs be changed
    abs_paths = [os.path.join(d, f) for f in filter_paths]
    return abs_paths


def get_attr_label(p):
    attr_dict = {}
    with open(p, "r") as f:
        lines = f.read().strip().split("\n")
        attrs = lines[1].split()
    for i in range(2, len(lines)):
        line = lines[i].split()
        img_name = line[0]
        sub_attr_dict = {}
        for attr, label in zip(attrs, line[1:]):
            sub_attr_dict[attr] = int(label)
        attr_dict[img_name] = sub_attr_dict
    assert len(list(attr_dict.keys())) == CELEBA_SET_SIZE
    return attr_dict


def get_eval_label(p):
    eval_dict = {}
    with open(p, "r") as f:
        lines = f.read().strip().split("\n")
    for i in range(len(lines)):
        line = lines[i].split()
        img_name = line[0]
        label = int(line[1])
        eval_dict[img_name] = label
    assert len(list(eval_dict.keys())) == CELEBA_SET_SIZE
    return eval_dict


def get_filted_data(img_paths, attr_label, eval_label, task=None, phase=None):
    A_paths = []
    B_paths = []
    for img_path in img_paths:
        name = os.path.basename(img_path)
        eval_id = eval_label[name]
        if (
            (phase == "train" and eval_id == 0)
            or (phase == "val" and eval_id == 1)
            or (phase == "test" and eval_id == 2)
        ):
            attr = attr_label[name][task]
            if attr == 1:  # pos
                A_paths.append(img_path)
            elif attr == -1:  # neg
                B_paths.append(img_path)
    return A_paths, B_paths


class UnalignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        img_dir = "datasets/celebA/align_5p"
        self.img_paths = get_img_paths(img_dir)
        print("Total img numbers: ", len(self.img_paths))
        attr_label_txt = "datasets/celebA/list_attr_celeba.txt"
        eval_label_txt = "datasets/celebA/list_eval_partition.txt"
        self.attr_label = get_attr_label(attr_label_txt)
        self.eval_label = get_eval_label(eval_label_txt)

        self.A_paths, self.B_paths = get_filted_data(
            self.img_paths,
            self.attr_label,
            self.eval_label,
            task=opt.task,
            phase=opt.phase,
        )
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        assert (
            self.opt.load_size >= self.opt.crop_size
        )  # crop_size should be smaller than the size of loaded image
        self.input_nc = (
            self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        )
        self.output_nc = (
            self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc
        )
        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[
            index % self.A_size
        ]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = cv2.imread(A_path)
        B_img = cv2.imread(B_path)
        A_img = Image.fromarray(A_img[:, :, ::-1])
        B_img = Image.fromarray(B_img[:, :, ::-1])

        # apply the same transform to both A and B
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
