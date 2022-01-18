"""
This code is used to batch detect images in a folder.
"""
import os
import cv2
from plantcv import plantcv as pcv
import numpy as np

import cv2

from vision.ssd.config.fd_config import define_img_size

define_img_size(
    320
)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

result_path = "./detect_imgs_results"
label_path = "./models/voc-model-labels.txt"
device = 'cpu'

candidate_size = 1000

model_path = "models/Epoch-95-Loss-3.3913408120473227.pth"
# model_path = "models/pretrained/version-RFB-640.pth"
net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
net.load(model_path)
predictor = create_Mb_Tiny_RFB_fd_predictor(net,
                                            candidate_size=candidate_size,
                                            device=device)

path = 'data/images/koronawirus_A320/weti0037'

listdir = os.listdir(path)
listdir = sorted(listdir, key=lambda img: int(img.split(".")[0]))
sum = 0

for file_path in listdir:
    img_path = os.path.join(path, file_path)
    orig_image = cv2.imread(img_path)

    img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    img = pcv.transform.rescale(gray_img=img)
    image = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, 0.7)
    if boxes.size(0):
        print(probs)
        print(boxes)
        print(labels)
    sum += boxes.size(0)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{probs[i]:.2f}"

        cv2.rectangle(img, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (0, 0, 255), 2)
        # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""

        # cv2.putText(orig_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('im', img)
    cv2.waitKey(10)
    print(f"Found {len(probs)} faces. The output image is {result_path}")
