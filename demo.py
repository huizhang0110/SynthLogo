from utils import LogoImage, ImageMergeHelper
import numpy as np

merge_helper = ImageMergeHelper(background_image_path='./data/bg/test.jpg')

logo = LogoImage(
    './data/logo/adidas/1.jpg').scale(scale_x=0.5, scale_y=0.5).rotate(45).change_color((0, 0, 255)).perspective()
merge_helper.add_logo(logo, logo_name='adidas')
logo = LogoImage(
    './data/logo/nike/1.jpg').scale().rotate(np.random.randint(0, 100)).perspective()
merge_helper.add_logo(logo, logo_name='nike')

merge_helper.save_result('./output.jpg', './output.xml')
merge_helper.show_result()
