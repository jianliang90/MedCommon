from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # distributed training parameters
        # 不要改该参数，系统会自动分配
        parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
        # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
        parser.add_argument('--world-size', default=4, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

        # add 
        parser.add_argument('--crop_size', nargs='+', type=int, default=[128,128,128])
        parser.add_argument('--aug', default='GAN')
        parser.add_argument('--src_ww_wl', nargs='+', type=int, default=[400, 40])
        parser.add_argument('--dst_ww_wl', nargs='+', type=int, default=[150, 75])
        parser.add_argument('--dst_vis_lut', default=None)
        parser.add_argument('--src_pattern', default='cropped_cta.nii.gz')
        parser.add_argument('--dst_pattern', default='cropped_mbf.nii.gz')
        parser.add_argument('--mask_pattern', default=None)
        parser.add_argument('--mask_label', type=int, default=1)
        parser.add_argument('--lambda_L1_Mask', type=float, default=0.05, help='weight for L1 Mask loss')
        parser.add_argument('--inference_mode', default='eval')
        # add ssl
        parser.add_argument('--ssl_sr', action='store_true', help='train with super resolution')
        parser.add_argument('--ssl_arch', default='resnet10')
        parser.add_argument('--ssl_pretrained_file', default=None)
        # add remove discriminator
        parser.add_argument('--no_discriminator', action='store_true')
        self.isTrain = True
        return parser
