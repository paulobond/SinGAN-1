from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')

    parser.add_argument('--experiment', type=int, help='Experiment number',default=0)
    parser.add_argument('--inpainting_mask_size', type=int, help='Random mask (inpainting)', default=30)
    parser.add_argument('--overlap_mask_disc', type=float, help='ratio', default=0.8)

    opt = parser.parse_args()
    opt = functions.post_config(opt)

    Gs = []
    Zs = []
    reals = []  # ranged from most downsampled to the original image
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)

        opt.input_name = opt.input_name + f"_exp_{opt.experiment}"

        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
