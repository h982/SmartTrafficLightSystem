
import numpy as np
import random
import warnings

import keras

from ..utils.anchors import (
    anchor_targets_bbox,
    anchors_for_shape,
    guess_shapes
)
from ..utils.config import parse_anchor_parameters, parse_pyramid_levels
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from ..utils.transform import transform_aabb

# 추상클래스
class Generator(keras.utils.Sequence):

    def __init__(
        self,
        transform_generator = None,
        visual_effect_generator=None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        no_resize=False,
        transform_parameters=None,
        compute_anchor_targets=anchor_targets_bbox,
        compute_shapes=guess_shapes,
        preprocess_image=preprocess_image,
        config=None
    ):
        """ Initialize Generator object.

        Args
            transform_generator    : A generator used to randomly transform images and annotations.
            batch_size             : The size of the batches to generate.
            group_method           : Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            no_resize              : If True, no image/annotation resizing is performed.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
            compute_shapes         : Function handler for computing the shapes of the pyramid for a given input.
            preprocess_image       : Function handler for preprocessing an image (scaling / normalizing) for passing through a network.
        """
        self.transform_generator            = transform_generator
        self.visual_effect_generator        = visual_effect_generator
        self.batch_size                     = int(batch_size)
        self.group_method                   = group_method
        self.shuffle_groups                 = shuffle_groups
        self.image_min_side                 = image_min_side
        self.image_max_side                 = image_max_side
        self.no_resize                      = no_resize
        self.transform_parameters           = transform_parameters or TransformParameters()
        self.compute_anchor_targets         = compute_anchor_targets
        self.compute_shapes                 = compute_shapes
        self.preprocess_image               = preprocess_image
        self.config                         = config

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    # if shuffle_group = true, epoch 종료마다 섞음
    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    """ 추상메소드 """
    # size : return size of dataset
    def size(self):
        raise NotImplementedError('size method not implemented')

    # num_classes : return the number of classes
    def num_classes(self):
        raise NotImplementedError('num_classes method not implemented')

    # has_label : return True if label is in the labels
    def has_label(self, label):

        raise NotImplementedError('has_label method not implemented')

    # has_name : return True if name is in the classes
    def has_name(self, name):

        raise NotImplementedError('has_name method not implemented')

    # name_to_label : mapping name to label
    def name_to_label(self, name):

        raise NotImplementedError('name_to_label method not implemented')

    # label_to_name : mapping label to name
    def label_to_name(self, label):

        raise NotImplementedError('label_to_name method not implemented')

    # image_aspect_ratio : compute the aspect ratio of image with image_index
    def image_aspect_ratio(self, image_index):

        raise NotImplementedError('image_aspect_ratio method not implemented')

    # image_path : return a path of image with image_index
    def image_path(self, image_index):

        raise NotImplementedError('image_path method not implemented')

    # load_image : load an image with image_index
    def load_image(self, image_index):

        raise NotImplementedError('load_image method not implemented')

    # load_annotaions : load annotaions with image_index
    def load_annotations(self, image_index):

        raise NotImplementedError('load_annotations method not implemented')

    # load_annotations_group : load annotations with group of image_indexes
    def load_annotations_group(self, group):
        annotations_group = [self.load_annotations(image_index) for image_index in group]

        # check annotation_group is forming dictionary And has 'labels', 'bboxes'
        for annotations in annotations_group:
            assert (isinstance(annotations,dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert ('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert ('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    # annotation 확인
    # filter_annotations : Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
    def filter_annotations(self, image_group, annotations_group, group):

        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image {} with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    self.image_path(group[index]),
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)
        return image_group, annotations_group

    # load_image_group : return loaded images in a group
    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    # visual effect 사용하는지 찾아보기
    # random_visual_effect_group_entry : randomly transform image and annotation
    def random_visual_effect_group_entry(self, image, annotations):
        visual_effect = next(self.visual_effect_generator)

        # apply visual effect
        image = visual_effect(image)
        return image, annotations

    # random_visual_effect_group : Randomly apply visual effect on each image.
    def random_visual_effect_group(self, image_group, annotations_group):
        assert(len(image_group) == len(annotations_group))

        if self.visual_effect_generator is None:
            # do nothing
            return image_group, annotations_group

        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], annotations_group[index] = self.random_visual_effect_group_entry(
                image_group[index], annotations_group[index]
            )
        return image_group, annotations_group

    # random_transform_group_entry : Randomly transforms image and annotation.
    def random_transform_group_entry(self, image, annotations, transform=None):

        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations

    # random_transform_group : Randomly transforms each image and its annotations.
    def random_transform_group(self, image_group, annotations_group):

        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    # resize_image : Resize an image using image_min_side and image_max_side.
    def resize_image(self, image):

        if self.no_resize:
            return image, 1
        else:
            return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    # preprocess_group_entry : Preprocess image and its annotations.
    def preprocess_group_entry(self, image, annotations):

        # resize image
        image, image_scale = self.resize_image(image)

        # preprocess the image
        image = self.preprocess_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    # Preprocess_group : Preprocess each image and its annotations in its group.
    def preprocess_group(self, image_group, annotations_group):
        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    # Group_images : Order the images according to self.order and makes groups of self.batch_size.
    def group_images(self):

        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    # compute_inputs : Compute inputs for the network using an image_group.
    def compute_inputs(self, image_group):

        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    # generate_anchors : generate anchors
    def generate_anchors(self, image_shape):
        anchor_params = None
        pyramid_levels = None
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = parse_anchor_parameters(self.config)
        if self.config and 'pyramid_levels' in self.config:
            pyramid_levels = parse_pyramid_levels(self.config)

        return anchors_for_shape(image_shape, anchor_params=anchor_params, pyramid_levels=pyramid_levels, shapes_callback=self.compute_shapes)

    # compute_targets : Compute target outputs for the network using images and their annotations.
    def compute_targets(self, image_group, annotations_group):

        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        return list(batches)

    # compute_input_output : Compute inputs and target outputs for the network.
    def compute_input_output(self, group):

        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    # Number of batches for generator.
    def __len__(self):

        return len(self.groups)

    # Keras sequence method for generating batches.
    def __getitem__(self, index):

        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)

        return inputs, targets
