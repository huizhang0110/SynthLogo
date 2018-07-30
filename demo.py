from utils import LogoPngImage, ImageMergeHelper
import numpy as np

merge_helper = ImageMergeHelper('./data/bg/test.jpg', debug=True)

logo = LogoPngImage(
    logo_path='./data/logo/adidas/adidas_1.png',
    logo_name='adidas'
).scale(scale_x=0.5, scale_y=0.5).rotate(45).perspective()
merge_helper.add_logo(logo)

logo = LogoPngImage(
    logo_path='./data/logo/supreme/supreme_1.png',
    logo_name='nike'
).rotate(np.random.randint(0, 100)).perspective()
merge_helper.add_logo(logo)

merge_helper.save_result('./output.jpg', './output.xml')
merge_helper.show_result()
