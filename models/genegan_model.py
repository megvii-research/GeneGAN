import torch
import itertools
from .base_model import BaseModel
from . import networks


class GeneGANModel(BaseModel):
    """This class implements the GeneGAN model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.set_defaults(norm="batch", netG="genenet", netD="genenet")
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.set_defaults(pool_size=50, gan_mode="wgangp")
            parser.add_argument(
                "--lambda_L1", type=float, default=1.0, help="weight for L1 loss"
            )
            parser.add_argument(
                "--lambda_parallel",
                type=float,
                default=0.01,
                help="weight for parallel loss",
            )

        return parser

    def __init__(self, opt):
        """Initialize the GeneGANModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "G_e", "G_parallel", "D_fake", "D_real"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["Ax", "Be", "Ax2", "Be2", "Bx", "Ae"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G_split", "G_join", "D1", "D2"]
        else:  # during test time, only load G
            self.model_names = ["G_split", "G_join"]
        # define networks (both generator and discriminator)
        self.netG_split, self.netG_join = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

        if (
            self.isTrain
        ):  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD1 = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            self.netD2 = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionAbs_zero = networks.AbsLossZero().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.RMSprop(
                itertools.chain(
                    self.netG_split.parameters(), self.netG_join.parameters()
                ),
                lr=opt.lr,
                alpha=0.8,
                weight_decay=1e-5,
                eps=1e-10,
            )
            self.optimizer_D = torch.optim.RMSprop(
                itertools.chain(self.netD1.parameters(), self.netD2.parameters()),
                lr=opt.lr,
                alpha=0.8,
                weight_decay=1e-5,
                eps=1e-10,
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.Ax = input["A"].to(self.device)
        self.Be = input["B"].to(self.device)
        self.image_paths = input["A_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.A, self.x = self.netG_split(self.Ax)
        self.B, self.e = self.netG_split(self.Be)
        self.Ax2 = self.netG_join(self.A, self.x)
        self.Be2 = self.netG_join(self.B, torch.zeros_like(self.e))
        self.Bx = self.netG_join(self.B, self.x)
        self.Ae = self.netG_join(self.A, torch.zeros_like(self.e))

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake_Bx = self.netD1(self.Bx.detach())
        pred_fake_Ae = self.netD2(self.Ae.detach())
        self.loss_D_Bx_fake = self.criterionGAN(pred_fake_Bx, False)
        self.loss_D_Ae_fake = self.criterionGAN(pred_fake_Ae, False)
        self.loss_D_fake = self.loss_D_Bx_fake + self.loss_D_Ae_fake
        # Real
        pred_real_Ax = self.netD1(self.Ax)
        pred_real_Be = self.netD2(self.Be)
        self.loss_D_Ax_real = self.criterionGAN(pred_real_Ax, True)
        self.loss_D_Be_real = self.criterionGAN(pred_real_Be, True)
        self.loss_D_real = self.loss_D_Ax_real + self.loss_D_Be_real
        # combine loss and calculate gradients
        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G should fake the discriminator
        pred_fake_Bx = self.netD1(self.Bx)
        pred_fake_Ae = self.netD2(self.Ae)
        self.loss_G_Bx_fake_GAN = self.criterionGAN(pred_fake_Bx, True)
        self.loss_G_Ae_fake_GAN = self.criterionGAN(pred_fake_Ae, True)
        self.loss_G_GAN = self.loss_G_Bx_fake_GAN + self.loss_G_Ae_fake_GAN

        # Second, cycle l1 loss
        self.loss_G_A_L1 = self.criterionL1(self.Ax, self.Ax2) * self.opt.lambda_L1
        self.loss_G_B_L1 = self.criterionL1(self.Be, self.Be2) * self.opt.lambda_L1
        self.loss_G_L1 = self.loss_G_A_L1 + self.loss_G_B_L1

        # e loss
        self.loss_G_e = self.criterionAbs_zero(self.e)

        # parallelogram loss
        parallel = self.Ax + self.Be - self.Bx - self.Ae
        self.loss_G_parallel = (
            self.criterionAbs_zero(parallel) * self.opt.lambda_parallel
        )

        # combine loss and calculate gradients
        self.loss_G = (
            self.loss_G_GAN + self.loss_G_L1 + self.loss_G_e + self.loss_G_parallel
        )
        self.loss_G.backward()

    def clip_weight(self, model, clip_value=(-0.01, 0.01)):
        for param in model.parameters():
            param.data = torch.clip(param.data, min=clip_value[0], max=clip_value[1])

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update G
        self.set_requires_grad(
            self.netD1, False
        )  # D requires no gradients when optimizing G
        self.set_requires_grad(
            self.netD2, False
        )  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()

        # update D
        self.set_requires_grad(self.netD1, True)  # enable backprop for D
        self.set_requires_grad(self.netD2, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        self.clip_weight(self.netD1)
        self.clip_weight(self.netD2)
