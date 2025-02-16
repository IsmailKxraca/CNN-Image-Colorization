"""
This script predicts a new image on the model.
- preprocess function for new images
- transform prediciton output of model to readable rgb-image
"""

import torch
import numpy as np
import cv2
from lab_quantization import *
from cnn import *
import matplotlib.pyplot as plt
import imageio
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)

def show_rgb_image(rgb_image):

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.show()


def preprocess_image(image_path, target_size=(256,256)):

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # Resize + convert to lab
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    # normalize L
    L_norm = lab[:, :, 0] / 255.0  

    # make tensor
    L_tensor = torch.tensor(L_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

    # output: normalized l_channel and l tensor
    return L_norm, L_tensor



def predictions_to_rgb(predictions, ab_grid, L_channel):
  
    # argmax
    pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
 
    pred_classes = pred_classes[0]

    # get ab values from the grid
    ab_pred = ab_grid[pred_classes]  

    # unnormalize L-Channel for Opencv
    L_channel = L_channel * 255

    # combine the L-Channel with the AB-Channels, to get LAB image
    lab_pred = np.concatenate([L_channel[..., np.newaxis], ab_pred], axis=-1)

    rgb_pred = cv2.cvtColor(lab_pred.astype(np.float32), cv2.COLOR_LAB2RGB)

    return rgb_pred


def predict(model, image_path, device="cpu"):
    L_channel, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor) 
        output = torch.softmax(output, dim=1)

    ab_grid = lab_bins(17, 237)

    rgb_image = predictions_to_rgb(output, ab_grid, L_channel)

    # show rgb
    # show_rgb_image(rgb_image)

    return rgb_image


def beispiel():
    model = ColorizationCNN()

    model_ = os.path.join(os.getcwd(), "colorization_model_with_rebalancing.pth")
    print(model_)

    # load weights
    state_dict = torch.load(model_, map_location=torch.device("cpu"))

    # delete "_orig_mod." Präfix from keys, that is made by torch.compile()
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    # test
    predicted_img= predict(model, os.path.join(os.getcwd(), f"scripts{os.sep}ILSVRC2012_val_00001933.JPEG"), device)

    predicted_img = (predicted_img * 255).astype("uint8")

    imageio.imwrite("test.png", predicted_img)

