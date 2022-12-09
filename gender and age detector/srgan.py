#python inference_gfpgan.py -i inputs/upload -o results -v 1.3 -s 2 --bg_upsampler realesrgan

import argparse
import cv2
import glob
import numpy as np
import os
import torch
#https://github.com/sailfish009/BasicSR.git
"""
os.system("sudo apt list --installed")
os.system('locate -b "\activate" | grep "/home"')

os.system("conda info --envs  # or ")
os.system("conda info -e")
os.system("conda env list")
os.system("pip list")
os.system("pip check")
os.system("pip install torch")
os.system("BASICSR_EXT=True -m pip install basicsr")
os.system("pip list")
#BASICSR_EXT=True 
"""
#os.system('BASICSR_EXT=True python setup.py develop')
from basicsr.utils import imwrite

from gfpgan import GFPGANer

#define main prediction function
#os.system("curl -LJO https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth")
#modelname ='GFPGANv1.pth'
#os.system("mv GFPGANv1.pth pretrained_models")
print("--------ls pretrained_models---------")
print(os.system("ls experiments/pretrained_models"))

def predictSrgan(image_path):
    print(image_path, "image_path")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default=image_path,
        help='Input image or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
    # we use version to select models, which is more user-friendly
    parser.add_argument(
        '-v', '--version', type=str, default='1.3', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    parser.add_argument(
        '-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')

    parser.add_argument(
        '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    args = parser.parse_args()

    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)

    # ------------------------ set up background upsampler ------------------------
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    if args.version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
    elif args.version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
    elif args.version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
    else:
        raise ValueError(f'Wrong model version {args.version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {model_name} does not exist.')

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    # read image
    img_name = os.path.basename(image_path)
    print(f'Processing {img_name} ...')
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img, has_aligned=args.aligned, only_center_face=args.only_center_face, paste_back=True)
    
    # save restored img
    if restored_img is not None:
        if args.ext == 'auto':
            extension = ext[1:]
        else:
            extension = args.ext

        if args.suffix is not None:
            save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}_gan.{extension}')
        else:
            save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
        imwrite(restored_img, save_restore_path)

    return save_restore_path