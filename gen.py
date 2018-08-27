import os
import argparse
from utils import ImageMergeHelper, LogoPngImage
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logo_dir", type=str, default="./data/logo", help="") 
    parser.add_argument("--bg_dir", type=str, default="./data/bg", help="")
    parser.add_argument("--save_dir", type=str, default="./output", help="")
    parser.add_argument("--epoch", type=int, default=1, help="Background images usage time.")
    return parser.parse_args()


def synth_logo_images(args):
    logo_png_dict = {}
    for root, dirs, files in os.walk(args.logo_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            logo_name = filepath.split("/")[-2]
            if logo_name in logo_png_dict:
                logo_png_dict[logo_name].append(filepath)
            else:
                logo_png_dict[logo_name] = [filepath]

    logo_classes = list(logo_png_dict.keys())

    def add_logo(merge_helper):
        use_logo_name = np.random.choice(logo_classes)
        use_logo_path = np.random.choice(logo_png_dict[use_logo_name])

        logo_image = LogoPngImage(use_logo_path, use_logo_name)

        # change_color
        # logo_image.change_color()

        # Scale
        bg_h, bg_w = merge_helper.bg_shape[:2]
        logo_h, logo_w = logo_image.image.shape[:2]
        scale_range = [0.2, 0.5]
        scale_ratio = scale_range[0] + (scale_range[1] - scale_range[0]) * np.random.random()  
        ratio_h = scale_ratio * bg_h / logo_h
        ratio_w = scale_ratio * bg_w / logo_w
        logo_image.scale(ratio_w, ratio_h)
        # perspective
        # logo_image.perspective()
        # rotate
        # logo_image.rotate(np.random.randint(0, 360))

        merge_helper.add_logo(logo_image)



    xml_save_dir = os.path.join(args.save_dir, "Annotations")
    img_save_dir = os.path.join(args.save_dir, "JPEGImages")

    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir)

    count = 0
    for i in range(args.epoch):
        for bg_filename in os.listdir(args.bg_dir):
            bg_path = os.path.join(args.bg_dir, bg_filename)
            merge_helper = ImageMergeHelper(bg_path)
            num_logo_per_bg = np.random.randint(1, 5)
            for i in range(num_logo_per_bg):
                add_logo(merge_helper)
            img_save_path = os.path.join(img_save_dir, "{}.jpg".format(count))
            xml_save_path = os.path.join(xml_save_dir, "{}.xml".format(count))
            merge_helper.save_result(img_save_path, xml_save_path)
            count += 1
    print("Finished gen {} images".format(count))


if __name__ == "__main__":
    synth_logo_images(get_args())

