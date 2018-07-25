import cv2
import numpy as np
import xml.dom.minidom as minidom


class LogoImage:
    """
    Logo image must be placed in a white background image so that logo_mask can be valid.
    """
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_RED = (0, 0, 255)

    def __init__(self, logo_path, logo_name):
        # Update these values each time you add a new transformation
        self.logo = cv2.imread(logo_path, 1)
        self._logo_name = logo_name
        self.h, self.w, self.c = self.logo.shape
        self.loc_points = [[0, 0], [self.w - 1, 0], [self.w - 1, self.h - 1], [0, self.h - 1]]

    @property
    def logo_name(self):
        return self._logo_name

    @property
    def logo_shape(self):
        return self.logo.shape

    def change_color(self, color=COLOR_BLACK):
        image_copy = self.logo.copy()
        image_copy[:, :] = color
        image2 = cv2.bitwise_and(image_copy, image_copy, self._get_mask())
        self.logo = cv2.add(self.logo, image2)
        return self

    def scale(self, scale_x=0.1, scale_y=0.1):
        self.logo = cv2.resize(self.logo, dsize=None, fx=scale_x, fy=scale_y)
        self.h, self.w = self.logo.shape[:2]
        self.loc_points = [[int(x * scale_x), int(y * scale_y)] for (x, y) in self.loc_points]
        return self

    def perspective(self, pts1=None, pts2=None):
        if pts1 is None or pts2 is None:
            pts1 = np.float32(self.loc_points)
            pts2 = []
            for x, y in self.loc_points:
                x += np.random.uniform(-0.2, 0.2) * self.w * np.random.normal()
                y += np.random.uniform(-0.2, 0.2) * self.h * np.random.normal()
                pts2.append([np.clip(int(x), 0, 1000), np.clip(int(y), 0, 1000)])
            self.loc_points = pts2
            pts2 = np.float32(pts2)
        M = cv2.getPerspectiveTransform(src=pts1, dst=pts2)
        image = cv2.warpPerspective(self.logo, M, (1000, 1000),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=self.COLOR_WHITE)
        xmin, ymin = int(min(pts2[:, 0])), int(min(pts2[:, 1]))
        xmax, ymax = int(max(pts2[:, 0])), int(max(pts2[:, 1]))
        self.logo = image[ymin:ymax, xmin:xmax, :]
        self.h, self.w = self.logo.shape[:2]
        self.loc_points = [[x - xmin, y - ymin] for x, y in self.loc_points]
        return self

    def rotate(self, angle=None):
        if angle is None:
            angle = np.random.randint(-120, 120)
        h, w = self.logo.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += (nw / 2) - cx
        M[1, 2] += (nh / 2) - cy
        self.logo = cv2.warpAffine(self.logo, M, dsize=(nw, nh),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=self.COLOR_WHITE)
        self.h, self.w = self.logo.shape[:2]
        points = []
        for x, y in self.loc_points:
            points.append([
                int(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
                int(M[1, 0] * x + M[1, 1] * y + M[1, 2])
            ])
        self.loc_points = points
        return self

    def _get_mask(self):
        logo_gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        _, logo_mask = cv2.threshold(logo_gray, 220, 255, cv2.THRESH_BINARY)
        return logo_mask

    def get_output(self):
        """Return the processed logo image, logo bbox and mask"""
        return self.logo, self.loc_points, self._get_mask()

    def highlight_loc_points(self):
        """For debug"""
        image = self.logo
        for x, y in self.loc_points:
            image = cv2.circle(image, (x, y), radius=5, color=self.COLOR_RED, thickness=-1)
        return image


class ImageMergeHelper:

    def __init__(self, background_image_path, debug=False):
        self.image_bg = cv2.imread(background_image_path, 1)
        self.annotations = []
        self._debug = debug

    @property
    def bg_shape(self):
        return self.image_bg.shape

    def add_logo(self, logo_image_obj, random_place=True, place_points=(0, 0)):
        assert isinstance(logo_image_obj, LogoImage), '{} is not a LogoImage object'
        logo, loc_point, mask = logo_image_obj.get_output()
        mask_not = cv2.bitwise_not(mask)
        bg_h, bg_w = self.image_bg.shape[:2]
        logo_h, logo_w = logo.shape[:2]
        if random_place:
            start_x = np.random.randint(0, bg_w - logo_w) if bg_w >= logo_w else 0
            start_y = np.random.randint(0, bg_h - logo_h) if bg_h >= logo_h else 0
        else:
            start_x, start_y = place_points
        roi = self.image_bg[start_y:start_y + logo_h, start_x:start_x + logo_w].copy()
        tmp1 = cv2.bitwise_and(roi, roi, mask=mask)
        tmp2 = cv2.bitwise_and(logo, logo, mask=mask_not)
        tmp = cv2.add(tmp1, tmp2)
        self.image_bg[start_y:start_y + logo_h, start_x:start_x + logo_w] = tmp
        pts = np.array(loc_point)
        xmin, ymin = int(min(pts[:, 0])) + start_x, int(min(pts[:, 1])) + start_y
        xmax, ymax = int(max(pts[:, 0])) + start_x, int(max(pts[:, 1])) + start_y

        if self._debug:
            cv2.rectangle(self.image_bg, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=1)

        self.annotations.append({
            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'logo_name': logo_image_obj.logo_name})
        return self

    def save_result(self, image_path, annotation_xml_path):
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
            name_node.appendChild(doc.createTextNode(str(annotation['logo_name'])))
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
            for _str in ['xmin', 'ymin', 'xmax', 'ymax']:
                child_node = doc.createElement(_str)
                child_node.appendChild(doc.createTextNode(str(annotation[_str])))
                bndbox_node.appendChild(child_node)
            obj_node.appendChild(bndbox_node)
            annotation_node.appendChild(obj_node)

        with open(annotation_xml_path, 'w') as fo:
            doc.writexml(fo, indent='\t', addindent='\t', newl='\n', encoding='utf-8')

    def show_result(self):
        """debug"""
        cv2.imshow('test', self.image_bg)
        cv2.waitKey(0)
