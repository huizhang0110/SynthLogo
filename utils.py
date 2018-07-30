import cv2
import numpy as np
import xml.dom.minidom as minidom


class LogoPngImage:

    def __init__(self, logo_path, logo_name):
        self._logo = cv2.imread(logo_path, -1)
        assert self._logo is not None, 'Damaged picture： {}'.format(logo_path)
        assert self._logo.shape[-1] == 4, 'Only support png format logo file. Please check {}.'.format(logo_path)
        self._logo_name = logo_name
        _, self._logo[:, :, -1] = cv2.threshold(self._logo[:, :, -1], 250, 255, cv2.THRESH_BINARY)

    def scale(self, scale_x=0.1, scale_y=0.1):
        self._logo = cv2.resize(self._logo, dsize=None, fx=scale_x, fy=scale_y)
        return self.update()

    def change_color(self, color):
        image_copy = self._logo[:, :, :3].copy()
        image_copy[:, :] = color
        self._logo[:, :, :3] = cv2.bitwise_and(self._logo[:, :, :3], 0, mask=self._logo[:, :, -1:])
        self._logo[:, :, :3] = cv2.add(image_copy, self._logo[:, :, :3], mask=self._logo[:, :, -1:])
        return self.update()

    def rotate(self, angle=None):
        if angle is None:
            angle = np.random.randint(-45, 45)
        h, w = self._logo.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += (nw / 2) - cx
        M[1, 2] += (nh / 2) - cy
        self._logo = cv2.warpAffine(self._logo, M, dsize=(nw, nh))
        return self.update()

    def perspective(self):
        h, w = self._logo.shape[:2]
        self._logo = cv2.warpAffine(self._logo,
                                    np.float32([[1, 0, w * .5], [0, 1, h * .5]]),
                                    (int(w * 1.5), int(h * 1.5)))

        pts1 = [[0, 0], [h-1, 0], [0, w-1], [h-1, w-1]]
        pts1 = [[x + int(w * .5), y + int(h * .5)] for x, y in pts1]
        pts2 = []
        tmp_h, tmp_w = h * 5, w * 5
        for x, y in pts1:
            x += np.random.uniform(-0.1, 0.1) * w
            y += np.random.uniform(-0.1, 0.1) * h
            pts2.append([np.clip(int(x), 0, tmp_w), np.clip(int(y), 0, tmp_h)])
        M = cv2.getPerspectiveTransform(src=np.float32(pts1), dst=np.float32(pts2))
        self._logo = cv2.warpPerspective(self._logo, M, (tmp_w, tmp_h))
        return self.update()

    def update(self):
        _, logo_mask = cv2.threshold(self._logo[:, :, -1], 200, 255, cv2.THRESH_BINARY)
        nonzero_mask = np.nonzero(logo_mask)
        min_y, max_y = np.min(nonzero_mask[0]), np.max(nonzero_mask[0])
        min_x, max_x = np.min(nonzero_mask[1]), np.max(nonzero_mask[1])
        self._logo = self._logo[min_y:max_y, min_x:max_x, ...]
        return self

    @property
    def image(self):
        return self._logo

    @property
    def name(self):
        return self._logo_name

    def show(self):
        cv2.imshow('logo', self._logo)
        cv2.waitKey()


class ImageMergeHelper:

    def __init__(self, bg_path, debug=False, add_erode=True):
        self.image_bg = cv2.imread(bg_path, 1)
        self.annotations = []
        self._debug = debug
        self.add_erode = add_erode

    @property
    def bg_shape(self):
        return self.image_bg.shape

    def add_logo(self, logo_obj, random_place=True, place_points=(0, 0)):
        assert isinstance(logo_obj, LogoPngImage), '{} is not a LogoPngImage object'
        logo_image = logo_obj.image
        bg_h, bg_w = self.image_bg.shape[:2]
        logo_h, logo_w = logo_image.shape[:2]
        assert bg_w >= logo_w and bg_h >= logo_h, 'Logo is big than background image.'
        if random_place:
            min_x, min_y = np.random.randint(0, bg_w - logo_w), np.random.randint(0, bg_h - logo_h)
        else:
            min_x, min_y = place_points
        max_x, max_y = min_x + logo_w, min_y + logo_h
        if self.impact_check([min_x, min_y, max_x, max_y]):
            print('crack')
            return self

        roi = self.image_bg[min_y:max_y, min_x:max_x].copy()
        mask = logo_image[:, :, -1]
        mask_not = cv2.bitwise_not(mask)
        logo_image = logo_image[:, :, :3]
        tmp1 = cv2.bitwise_and(roi, roi, mask=mask_not)
        tmp2 = cv2.bitwise_and(logo_image, logo_image, mask=mask)
        self.image_bg[min_y:max_y, min_x:max_x] = cv2.add(tmp1, tmp2)

        if self._debug:
            cv2.rectangle(self.image_bg, (min_x, min_y), (max_x, max_y), (0, 255, 0), thickness=1)
        self.annotations.append([logo_obj.name, min_x, min_y, max_x, max_y])
        return self

    def save_result(self, image_path, annotation_xml_path):
        if self.add_erode:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            self.image_bg = cv2.erode(self.image_bg, kernel)
        cv2.imwrite(image_path, self.image_bg)

        # Write result to annotation xml file as VOC format
        doc = minidom.Document()
        annotation_node = doc.createElement('annotation')
        doc.appendChild(annotation_node)

        folder_node = doc.createElement('folder')
        folder_node.appendChild(doc.createTextNode('synth_logo'))
        annotation_node.appendChild(folder_node)

        filename_node = doc.createElement('filename')
        filename_node.appendChild(doc.createTextNode('{}'.format(image_path)))
        annotation_node.appendChild(filename_node)

        size_node = doc.createElement('size')
        for _str, value in zip(['height', 'width', 'depth'], self.image_bg.shape):
            child_node = doc.createElement(_str)
            child_node.appendChild(doc.createTextNode(str(value)))
            size_node.appendChild(child_node)
        annotation_node.appendChild(size_node)

        for annotation in self.annotations:
            obj_node = doc.createElement('object')

            name_node = doc.createElement('name')
            name_node.appendChild(doc.createTextNode(str(annotation[0])))  # logo_name
            obj_node.appendChild(name_node)

            pose_node = doc.createElement('pose')
            pose_node.appendChild(doc.createTextNode(str('Right')))
            obj_node.appendChild(pose_node)

            truncated_node = doc.createElement('truncated')
            truncated_node.appendChild(doc.createTextNode(str('0')))
            obj_node.appendChild(truncated_node)

            difficult_node = doc.createElement('difficult')
            difficult_node.appendChild(doc.createTextNode(str('0')))
            obj_node.appendChild(difficult_node)

            bndbox_node = doc.createElement('bndbox')
            for _str, val in zip(['xmin', 'ymin', 'xmax', 'ymax'], annotation[1:]):
                child_node = doc.createElement(_str)
                child_node.appendChild(doc.createTextNode(str(val)))
                bndbox_node.appendChild(child_node)
            obj_node.appendChild(bndbox_node)
            annotation_node.appendChild(obj_node)

        with open(annotation_xml_path, 'w') as fo:
            doc.writexml(fo, indent='\t', addindent='\t', newl='\n', encoding='utf-8')

    def impact_check(self, box, iou_threshold=0.2):

        def calc_iou(box1, box2):
            s1 = (box1[3] - box1[1]) * (box1[2] - box1[0])
            s2 = (box2[3] - box2[1]) * (box2[2] - box2[0])

            dw = min(box2[2], box1[2]) - max(box2[0], box1[0])
            dh = min(box2[3], box1[3]) - max(box2[1], box1[1])
            if dw <= 0 or dh <= 0:
                intersation = 0
            else:
                intersation = dw * dh
            return intersation / (s1 + s2 - intersation)

        if len(self.annotations) != 0:
            for annotation in self.annotations:
                iou = calc_iou(box, annotation[1:])
                if iou > iou_threshold:
                    return True
        return False

    def show_result(self):
        """debug"""
        cv2.imshow('test', self.image_bg)
        cv2.waitKey(0)