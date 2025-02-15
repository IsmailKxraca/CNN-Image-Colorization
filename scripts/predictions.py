import torch
import numpy as np
import cv2
from lab_quantization import *
from cnn import *
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt
import imageio


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

    return L_norm, L_tensor



def predictions_to_rgb(predictions, ab_grid, L_channel):
  
    # argmax
    pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()

    print(pred_classes.shape)
 
    pred_classes = pred_classes[0]

    print(pred_classes.shape)
   
    # get ab values from the grid
    ab_pred = ab_grid[pred_classes]  

    print(ab_pred)

    # unnormalize L-Channel for Opencv
    L_channel = L_channel * 255

    # combine the L-Channel with the AB-Channels, to get LAB image
    lab_pred = np.concatenate([L_channel[..., np.newaxis], ab_pred], axis=-1)

    rgb_pred = cv2.cvtColor(lab_pred.astype(np.float32), cv2.COLOR_LAB2RGB)

    return rgb_pred


def predict(model, image_path, device="cuda"):
    L_channel, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor) 
        output = torch.softmax(output, dim=1)

    ab_grid = lab_bins(17, 237)

    rgb_image = predictions_to_rgb(output, ab_grid, L_channel)


    def show_rgb_image(rgb_image):
        """
        Zeigt ein RGB-Bild mit Matplotlib an.
        :param rgb_image: NumPy-Array mit Shape [H, W, 3]
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb_image)
        plt.axis("off")
        plt.show()

    # show rgb
    show_rgb_image(rgb_image)

    return rgb_image


# Model
model = ColorizationCNN()

model_ = r"/workspace/CNN-Image-Colorization/colorization_model.pth"

# load weights
state_dict = torch.load(model_, map_location=torch.device("cpu"))


# delete "_orig_mod." Präfix from keys, that is made by torch.compile()
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(new_state_dict)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

# test
predicted_img= predict(model, r"/workspace/CNN-Image-Colorization/scripts/ILSVRC2012_val_00001933.JPEG", device)

predicted_img = (predicted_img * 255).astype("uint8")

imageio.imwrite("Biber_ohne_rebalancing.png", predicted_img)
