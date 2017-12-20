import cv2
import numpy as np


def main():
    face = cv2.imread('images/child.jpg')
    mask = cv2.imread('images/dog.jpg')

    mask_h, mask_w, _ = mask.shape
    face_h, face_w, _ = face.shape

    factor = min(face_h / mask_h, face_w / mask_w)
    new_mask_w = int(factor * mask_w)
    new_mask_h = int(factor * mask_h)
    new_mask_shape = (new_mask_w, new_mask_h)
    resized_mask = cv2.resize(mask, new_mask_shape)
    cv2.imwrite('resized_dog.jpg', resized_mask)

    face_with_mask = face.copy()
    non_white_pixels = (resized_mask < 250).all(axis=2)
    face_with_mask[:new_mask_h, :new_mask_w][non_white_pixels] = resized_mask[non_white_pixels]

    cv2.imwrite('child_with_dog_mask.jpg', face_with_mask)

if __name__ == '__main__':
    main()
