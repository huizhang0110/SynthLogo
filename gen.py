import os
import argparse
from utils import ImageMergeHelper, LogoImage
from logo_use_config import *
import numpy as np


def main(args):

    def random_add_logo(merge_helper):
        logo_name = np.random.choice(list(logo_use_config.keys()))
        logo_images_name = list(logo_use_config[logo_name].keys())
        p = [v[0] for k, v in logo_use_config[logo_name].items()]
        p = [x / sum(p) for x in p]
        use_logo_image_name = np.random.choice(logo_images_name, p=p)
        use_logo_path = os.path.join(args.logo_dir, logo_name, use_logo_image_name)
        # print(use_logo_path)
        logo_image = LogoImage(use_logo_path, logo_name)
        operations = logo_use_config[logo_name][use_logo_image_name][1:]

        if 'change_color' in operations:
            p = [x[1] for x in all_colors]
            p = [x / sum(p) for x in p]
            choice_idx = np.random.choice(len(all_colors), 1, p=p)[0]
            color = all_colors[choice_idx][0]
            logo_image.change_color(color)

        if 'rotate' in operations:
            logo_image.rotate(np.random.randint(-100, 100))

        if 'perspective' in operations:
            logo_image.perspective()

        if 'scale' in operations:
            bg_h, bg_w = merge_helper.bg_shape[:2]
            logo_h, logo_w = logo_image.logo_shape[:2]
            # Truncated normal distribution
            scale_ratio = min(bg_h / logo_h, bg_w / logo_w) * np.clip(abs(np.random.normal()), 0.1, 0.5)
            logo_image.scale(scale_ratio, scale_ratio)

        merge_helper.add_logo(logo_image)

    xml_save_dir = os.path.join(args.save_dir, 'Annotations')
    img_save_dir = os.path.join(args.save_dir, 'JPEGImages')
    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir, exist_ok=True)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir, exist_ok=True)

    count = 0
    for i in range(args.epoch):
        for bg_filename in os.listdir(args.bg_dir):
            bg_path = os.path.join(args.bg_dir, bg_filename)
            try:
                # merge_helper = ImageMergeHelper(bg_path, debug=True)
                merge_helper = ImageMergeHelper(bg_path)
                num_logo_per_bg = np.random.randint(1, 3)
                for i in range(num_logo_per_bg):
                    random_add_logo(merge_helper)
                # merge_helper.show_result()
                img_save_path = os.path.join(img_save_dir, '{}.jpg'.format(count))
                xml_save_path = os.path.join(xml_save_dir, '{}.jpg'.format(count))
                merge_helper.save_result(img_save_path, xml_save_path)
            except Exception as e:
                continue
            count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logo_dir', type=str, default='./data/logo',
                        help='The directory containing logo images and a using probability  file about each logo.')
    parser.add_argument('--bg_dir', type=str, default='./data/bg',
                        help='The directory containing background images.')
    parser.add_argument('--save_dir', type=str, default='./output',
                        help='The directory containing pascal-voc format synthetic data.')
    parser.add_argument('--epoch', type=int, default=2,
                        help='Background images usage time.')
    args = parser.parse_args()
    main(args)
