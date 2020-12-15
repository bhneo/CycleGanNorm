import fid_score
import lpips
import torch
import os


def scores(path1, path2, batch_size=50, fid_dims=2048, device=None):
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)

    fid_value = fid_score.calculate_fid_given_paths([path1, path2],
                                                    batch_size,
                                                    device,
                                                    fid_dims)
    print("fid score: {}".format(fid_value))

    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

    import torch
    img0 = torch.zeros(1, 3, 64, 64)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = torch.zeros(1, 3, 64, 64)
    d = loss_fn_alex(img0, img1)


def get_lpips(device, version='0.1', ):
    # Initializing the model
    loss_fn = lpips.LPIPS(net='alex', version=version)
    loss_fn.to(device)

    # crawl directories
    f = open(opt.out, 'w')
    files = os.listdir(opt.dir0)

    for file in files:
        if (os.path.exists(os.path.join(opt.dir1, file))):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0, file)))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1, file)))

            if (opt.use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0, img1)
            print('%s: %.3f' % (file, dist01))
            f.writelines('%s: %.6f\n' % (file, dist01))

    f.close()

