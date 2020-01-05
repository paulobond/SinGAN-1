from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions


if __name__ == '__main__':

    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    opt = parser.parse_args()
    opt = functions.post_config(opt)

    # LOAD MODEL #
    input_name = opt.input_name
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    Gs = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/Gs.pth', map_location=map_location)
    Zs = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/Zs.pth', map_location=map_location)
    Ds = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/Ds.pth', map_location=map_location)
    reals = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/reals.pth', map_location=map_location)
    NoiseAmp = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/NoiseAmp.pth', map_location=map_location)





