

from ..preprocessing.generator import Generator
from ..utils.image import read_image_bgr

import os
import numpy as np

from pycocotools.coco import COCO

# 구현 클래스
class CocoGenerator(Generator):

    def __init__(self, data_dir, set_name, **kwargs):
        """ Initialize a COCO data generator.

        Args
            data_dir: Path to where the COCO dataset is stored.
            set_name: Name of the set to parse.
        """
        self.data_dir  = data_dir
        self.set_name  = set_name
        self.coco      = COCO(os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

        super(CocoGenerator, self).__init__(**kwargs)

    # loads classes, labels, class_to_label, label_to_class
    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    # 구현 메소드
    def size(self):

        return len(self.image_ids)

    def num_classes(self):

        return len(self.classes)

    def has_label(self, label):

        return label in self.labels

    def has_name(self, name):

        return name in self.classes

    def name_to_label(self, name):

        return self.classes[name]

    def label_to_name(self, label):

        return self.labels[label]

    def coco_label_to_label(self, coco_label):

        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):

        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):

        return self.coco_labels[label]

    def image_path(self, image_index):

        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        return path

    def image_aspect_ratio(self, image_index):

        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):

        path  = self.image_path(image_index)
        return read_image_bgr(path)

    def load_annotations(self, image_index):

        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate([annotations['labels'], [self.coco_label_to_label(a['category_id'])]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)

        return annotations
