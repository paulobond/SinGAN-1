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


def SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0):

    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    image_cur = None

    for G, Z_opt, noise_amp in zip(Gs, Zs, NoiseAmp):

        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        image_prev = image_cur

        if n == 0:
            z_curr = functions.generate_noise([1, nzx, nzy], device=opt.device)
            z_curr = z_curr.expand(1, 3, z_curr.shape[2], z_curr.shape[3])
            z_curr = m(z_curr)
        else:
            z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
            z_curr = m(z_curr)

        if image_prev is None:
            I_prev = m(in_s)
        else:
            I_prev = image_prev
            I_prev = imresize(I_prev, 1/opt.scale_factor, opt)
            I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
            I_prev = m(I_prev)
            I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
            I_prev = functions.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])

        if n < gen_start_scale:
            z_curr = Z_opt

        z_in = noise_amp*z_curr + I_prev
        image_cur = G(z_in.detach(), I_prev)

        n += 1

    return image_cur.detach()
