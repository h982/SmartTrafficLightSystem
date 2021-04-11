from __future__ import print_function
import sys


class Backbone(object):

    def __init__(self, backbone):

        from .. import layers
        from .. import losses
        from .. import initializers
        self.custom_objects = {
            'UpsampleLike'     : layers.UpsampleLike,
            'PriorProbability' : initializers.PriorProbability,
            'RegressBoxes'     : layers.RegressBoxes,
            'FilterDetections' : layers.FilterDetections,
            'Anchors'          : layers.Anchors,
            'ClipBoxes'        : layers.ClipBoxes,
            '_smooth_l1'       : losses.smooth_l1(),
            '_focal'           : losses.focal(),
        }

        self.backbone = backbone
        self.validate()

    def retinanet(self, *args, **kwargs):
        raise NotImplementedError('retinanet method not implemented.')

    def download_imagenet(self):
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        raise NotImplementedError('preprocess_image method not implemented.')


def backbone(backbone_name):
    if 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)


def load_model(filepath, backbone_name='resnet50'):

    import keras.models
    return keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)


def convert_model(model, nms=True, class_specific_filter=True, anchor_params=None, **kwargs):

    from .retinanet import retinanet_bbox
    return retinanet_bbox(model=model, nms=nms, class_specific_filter=class_specific_filter, anchor_params=anchor_params, **kwargs)


def assert_training_model(model):

    assert(all(output in model.output_names for output in ['regression', 'classification'])), \
        "Input is not a training model (no 'regression' and 'classification' outputs were found, outputs are: {}).".format(model.output_names)


def check_training_model(model):

    try:
        assert_training_model(model)
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
