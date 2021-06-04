import argparse
import os

import torch

from util import util

class GraphOptions():
    '''
    This class defines options used during Graph Construction.
    '''
    def __init__(self):
        '''Reset the class; indicates the class hasn't been initailized'''
        self.initialized = False

    def initialize(self, parser):

        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders with slide names etc)')
        parser.add_argument('--slideroot', default='/SeaExp/RushinTest/SPixel-FCN/CPTAC', type=str, help='path to slides (should have svs files with slide names etc)')
        parser.add_argument('--slide_list', default='slides.csv', type=str, help='path to csv file of slides. Should have columns: slide_name, label')
        parser.add_argument('--nspix', default=49, type=int, help='number of superpixels')
        parser.add_argument('--n_iter', default=10, type=int, help='number of iterations for differentiable SLIC')
        parser.add_argument('--fdim', default=20, type=int, help='embedding dimension')
        parser.add_argument('--color_scale', default=0.26, type=float)
        parser.add_argument('--pos_scale', default=2.5, type=float)
        parser.add_argument('--weight', default=None, type=str, help='/path/to/pretrained_weight')
        parser.add_argument('--enforce_connectivity', action='store_true', help='if specified, cluster spixels to maintain connectivity among pixels')

        parser.add_argument('--downsample_factor', default=4, type=int, help='By what factor should the visualization image be downsampled')

        self.initialized = True
        return parser

    def gather_options(self):

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        '''Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        '''
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.name))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        '''Parse our options, create checkpoints directory suffix, and set up gpu device.'''
        opt = self.gather_options()

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
