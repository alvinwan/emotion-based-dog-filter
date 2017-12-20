"""Test dog mask application for a single, static image."""

import argparse
import os
import cv2


def main():
    args = argparse.ArgumentParser('Dog Mask for single images')
    args.add_argument('--mask', default='assets/dog.png')
    args = args.parse_args()

    face = cv2.imread('assets/child.png')
    mask = cv2.imread(args.mask)

    mask_h, mask_w, _ = mask.shape
    face_h, face_w, _ = face.shape

    factor = min(face_h / mask_h, face_w / mask_w)
    new_mask_w = int(factor * mask_w)
    new_mask_h = int(factor * mask_h)
    new_mask_shape = (new_mask_w, new_mask_h)
    resized_mask = cv2.resize(mask, new_mask_shape)
    cv2.imwrite('outputs/resized_dog.png', resized_mask)

    face_with_mask = face.copy()
    non_white_pixels = (resized_mask < 250).all(axis=2)
    face_with_mask[:new_mask_h, :new_mask_w][non_white_pixels] = resized_mask[non_white_pixels]

    mask_name = os.path.basename(args.mask).split('.')[0]
    cv2.imwrite('outputs/child_with_%s_mask.png' % mask_name, face_with_mask)

if __name__ == '__main__':
    main()
