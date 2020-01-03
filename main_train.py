from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')

    parser.add_argument('--inpainting_mask_size', type=int, help='Random mask (inpainting)', default=0)

    parser.add_argument('--half_rec_loss', type=bool, help='Rec loss computed on the left half only (used as a test)',
                        default=False)

    opt = parser.parse_args()
    opt = functions.post_config(opt)

    if opt.half_rec_loss:
        # Don't allow both experiment at the same time
        opt.inpainting_mask_size = 0

    Gs = []
    Zs = []
    reals = []  # ranged from most downsampled to the original image
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if opt.half_rec_loss:
        print(f"EXPERIMENT: rec loss computed on the left half of the image only")

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
