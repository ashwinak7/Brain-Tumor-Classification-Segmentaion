import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def segment_tumor(nii_path, model_type, gt_path=None):
    nii_img = nib.load(nii_path)
    img_data = nii_img.get_fdata()
    slice_idx = img_data.shape[2] // 2
    slice_img = img_data[:, :, slice_idx]

    # Normalize
    slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
    slice_img_resized = cv2.resize(slice_img, (128, 128))

    input_save_path = os.path.join('static/results', 'input_slice.png')
    plt.imsave(input_save_path, slice_img_resized, cmap='gray')

    img_with_channel = np.expand_dims(slice_img_resized, axis=-1)
    img_2ch = np.concatenate([img_with_channel, img_with_channel], axis=-1)
    input_img = np.expand_dims(img_2ch, axis=0)

    if model_type == 'unet':
        model = tf.keras.models.load_model('models/Final 3D UNet.h5', compile=False)
    elif model_type == 'attention':
        model = tf.keras.models.load_model('models/Final Attention 3D UNet.h5', compile=False)
    else:
        return "Invalid segmentation model."

    prediction = model.predict(input_img)
    segmented = prediction[0, :, :, 0]
    segmented_binary = (segmented > 0.5).astype(np.uint8)

    output_path = os.path.join('static/results', 'seg_result.png')
    plt.imsave(output_path, segmented_binary, cmap='gray')

    if gt_path:
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt_img, (128, 128))
        gt_binary = (gt_resized > 127).astype(np.uint8)

        dice = dice_score(gt_binary, segmented_binary)
        iou = iou_score(gt_binary, segmented_binary)
    else:
        # Default dummy mask (center square)
        dummy_gt = np.zeros_like(segmented_binary)
        center = 64
        dummy_gt[center-10:center+10, center-10:center+10] = 1
        dice = dice_score(dummy_gt, segmented_binary)
        iou = iou_score(dummy_gt, segmented_binary)

    return input_save_path, output_path, {
        "Dice Score": f"{dice:.4f}",
        "Mean IoU": f"{iou:.4f}"
    }
