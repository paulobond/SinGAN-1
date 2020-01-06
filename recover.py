from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
import os

if __name__ == '__main__':

    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)

    parser.add_argument('--fake_input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--fake_input_name', help='training image name', required=True)

    parser.add_argument('--reg', help='regularization parameter', type=float, default=0.05)
    parser.add_argument('--disc_loss', help='discrimination loss', type=float, default=0.01)

    parser.add_argument('--use_zopt', help='use z_opt to initialize z', type=bool, default=False)


    opt = parser.parse_args()
    opt.mode = 'train'
    opt = functions.post_config(opt)

    # Output dir
    dir_name = f'Recover/{opt.input_name[:-4]}_{opt.fake_input_name[:-4]}_{opt.reg}_{opt.disc_loss}_{opt.use_zopt}'
    os.makedirs(dir_name, exist_ok=True)

    # LOAD MODEL #
    input_name = opt.input_name
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    input_name = opt.input_name[:-4]
    Gs = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/Gs.pth', map_location=map_location)
    Zs = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/Zs.pth', map_location=map_location)
    Ds = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/Ds.pth', map_location=map_location)
    reals = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/reals.pth', map_location=map_location)
    NoiseAmp = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/NoiseAmp.pth', map_location=map_location)
    optbis = torch.load(f'TrainedModels/{input_name}/scale_factor=0.750000,alpha=10/opt.pth', map_location=map_location)
    opt.scale_factor = optbis.scale_factor
    opt.scale1 = optbis.scale1
    opt.stop_scale = optbis.stop_scale

    fake = img.imread('%s/%s' % (opt.fake_input_dir, opt.fake_input_name))
    fake = functions.np2torch(fake, opt)
    fake = imresize(fake, opt.scale1, opt)
    fakes = []
    fakes = functions.creat_reals_pyramid(fake, fakes, opt)

    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    image_cur = None
    pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2
    m = nn.ZeroPad2d(int(pad1))
    n = 0

    for G, Z_opt, noise_amp, fake, D in zip(Gs, Zs, NoiseAmp, fakes, Ds):

        print(f"\n\n*******  Scale {n}  ***********\n")

        nzx = (Z_opt.shape[2] - pad1 * 2)
        nzy = (Z_opt.shape[3] - pad1 * 2)
        image_prev = image_cur

        if opt.use_zopt:
            z_curr = Z_opt
        else:
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
            I_prev = imresize(I_prev.detach(), 1/opt.scale_factor, opt)
            I_prev = I_prev[:, :, 0:round(1 * reals[n].shape[2]), 0:round(1 * reals[n].shape[3])]
            I_prev = m(I_prev)
            I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
            I_prev = functions.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])

        z_curr.requires_grad_()

        optimizer_z = optim.Adam([z_curr], lr=opt.lr_d)
        scheduler_z = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_z, milestones=[1600], gamma=opt.gamma)

        os.mkdir(f"{dir_name}/{n}")
        for i in range(10000):
            image_cur = G(noise_amp*z_curr + I_prev, I_prev)
            loss = nn.MSELoss()
            diff = loss(fake, image_cur)
            errD = - D(image_cur).mean()
            (diff + opt.reg * z_curr.abs().mean() + opt.disc_loss * errD).backward(retain_graph=True)
            optimizer_z.step()
            # print(z_curr[0,0,10:15,10:15])
            if i % 1000 == 0:
                print(f"** Iteration {i} ** (reg: {opt.reg}; disc loss weight: {opt.disc_loss}; use zopt:"
                      f" {opt.use_zopt})")
                print(f"MSE Loss: {loss(fake, image_cur)}")
                print(f"Mean |z|: {z_curr.abs().mean()}")
                print(f"Max |z|: {z_curr.abs().max()}")
                print(f"Error Discriminator: {- D(image_cur).mean()}")

                with open(f"{dir_name}/{n}/report.txt", 'a') as txt_f:
                    txt_f.write(f'Iteration {i}  (reg: {opt.reg}; disc loss weight: {opt.disc_loss};'
                                f' use zopt: {opt.use_zopt})\n'
                                f'MSE loss: {loss(fake, image_cur)}\n'
                                f'Mean |z|: {z_curr.abs().mean()}\n'
                                f'Max |z|: {z_curr.abs().max()}\n'
                                f'Error Discriminator: {- D(image_cur).mean()}\n\n\n')

        plt.imsave(f'{dir_name}/{n}/reconstructed_image.png', functions.convert_image_np(image_cur.detach()), vmin=0, vmax=1)
        plt.imsave(f'{dir_name}/{n}/target_image.png', functions.convert_image_np(fake), vmin=0, vmax=1)

        n += 1
        opt.reg = 0.5 * opt.reg  # decrease regularization param over time
        opt.disc_loss = 0.5 * opt.disc_loss


def SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0):

    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    image_cur = None

    pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2
    m = nn.ZeroPad2d(int(pad1))

    for G, Z_opt, noise_amp in zip(Gs, Zs, NoiseAmp):

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
