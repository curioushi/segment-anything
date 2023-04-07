import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from glob import glob

sam_checkpoint = "/home/shq/Models/SAM/sam_vit_h_4b8939.pth"
device = "cuda"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image_folder = '/home/shq/Downloads/shefule/gun50/'
images = sorted(glob(image_folder + '*.jpg'))

index = 0
image = cv2.imread(images[index])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image[410:1306, 670:1556]
predictor.set_image(image)

cur_x, cur_y = 0, 0
positive_points = []
negative_points = []

def mouse_callback(event, x, y, flags, param):
    global cur_x, cur_y, positive_points, negative_points
    cur_x, cur_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        positive_points.append((x, y))
    elif event == cv2.EVENT_MBUTTONDOWN:
        negative_points.append((x, y))

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)


while True:
    if len(positive_points) == 0:
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[cur_x, cur_y]]),
            point_labels=np.array([1]),
            multimask_output=True,
            return_logits=True,
        )
        predicts = []
        for i, mask in enumerate(masks):
            mask = torch.sigmoid(torch.from_numpy(mask)).numpy()
            h, w = mask.shape
            mask = cv2.resize(mask, (h//2, w//2))
            predicts.append(mask)
        cv2.imshow(f"predict", np.hstack(predicts))
    elif len(positive_points) > 0:
        masks, scores, logits = predictor.predict(
            point_coords=np.array(positive_points + negative_points),
            point_labels=np.array([1] * len(positive_points) + [0] * len(negative_points)),
            multimask_output=True,
            return_logits=True,
        )
        predicts = []
        for i, mask in enumerate(masks):
            mask = torch.sigmoid(torch.from_numpy(mask)).numpy()
            h, w = mask.shape
            mask = cv2.resize(mask, (h//2, w//2))
            predicts.append(mask)
        cv2.imshow(f"predict", np.hstack(predicts))
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord('c'):
        positive_points.clear()
        negative_points.clear()
    elif key == ord('n'):
        index = (index + 1)
        image = cv2.imread(images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[410:1306, 670:1556]
        predictor.set_image(image)
        
    cv2.imshow("image", image)
