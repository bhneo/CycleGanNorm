import os
import torch
import fid_score
import lpips_score
from metrics import lpips


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

    get_lpips(path1, path2, output, device)


def get_lpips(path1, path2, output, device, version='0.1'):
    # Initializing the model
    loss_fn = lpips_score.LPIPS(net='alex', version=version)
    loss_fn.to(device)

    # crawl directories
    f = open(output, 'w')
    files = os.listdir(path1)

    for file in files:
        if os.path.exists(os.path.join(path1, file)):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(path1, file)))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(path2, file)))

            img0 = img0.to(device)
            img1 = img1.to(device)

            # Compute distance
            dist01 = loss_fn.forward(img0, img1)
            print('%s: %.3f' % (file, dist01))
            f.writelines('%s: %.6f\n' % (file, dist01))

    f.close()

