import cv2
import numpy as np
import torch
import torch.nn.functional as F
from eccv16 import eccv16
from skimage import color
from siggraph17 import siggraph17
from PIL import Image


def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if out_np.ndim == 2:
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def resize_img(img, HW=(256, 256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


def preprocess_image(img_rgb, HW=(256, 256), resample=3):
    resized_img_rgb = resize_img(img_rgb, HW=HW, resample=resample)

    orig_lab_img = color.rgb2lab(img_rgb)
    resized_lab_img = color.rgb2lab(resized_img_rgb)

    resized_img_l_channel = resized_lab_img[:, :, 0]
    orig_img_l_channel = orig_lab_img[:, :, 0]

    resized_tensor_l = torch.Tensor(resized_img_l_channel)[None, None, :, :]
    orig_tensor_l = torch.Tensor(orig_img_l_channel)[None, None, :, :]

    return orig_tensor_l, resized_tensor_l


def postprocess_image(orig_tensor_l, out_ab, mode="bilinear"):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = orig_tensor_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((orig_tensor_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))


def colorize(orig_tensor_l, resized_tensor_l):
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    out_img_eccv16 = postprocess_image(orig_tensor_l, colorizer_eccv16(resized_tensor_l).cpu())
    out_img_siggraph17 = postprocess_image(orig_tensor_l, colorizer_siggraph17(resized_tensor_l).cpu())

    return out_img_siggraph17
