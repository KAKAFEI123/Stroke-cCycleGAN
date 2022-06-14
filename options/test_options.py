from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser) 
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--eval', default=True, action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=10000000, help='how many test images to run')
        parser.add_argument('--dataset_mode', type=str, default='testcondition', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization | test]')
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
