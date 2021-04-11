
import keras
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):

    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # sigmoid 적용
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):

    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(backbone_layers, pyramid_levels, feature_size=256):
    output_layers = {}

    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(backbone_layers['C5'])
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, backbone_layers['C4']])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)
    output_layers["P5"] = P5

    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(backbone_layers['C4'])
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, backbone_layers['C3']])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    output_layers["P4"] = P4

    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(backbone_layers['C3'])
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    if 'C2' in backbone_layers and 2 in pyramid_levels:
        P3_upsampled = layers.UpsampleLike(name='P3_upsampled')([P3, backbone_layers['C2']])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)
    output_layers["P3"] = P3

    if 'C2' in backbone_layers and 2 in pyramid_levels:
        P2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(backbone_layers['C2'])
        P2 = keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
        P2 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2')(P2)
        output_layers["P2"] = P2

    if 6 in pyramid_levels:
        P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(backbone_layers['C5'])
        output_layers["P6"] = P6

    if 7 in pyramid_levels:
        if 6 not in pyramid_levels:
            raise ValueError("P6 is required to use P7")
        P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
        P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)
        output_layers["P7"] = P7

    return output_layers


def default_submodels(num_classes, num_anchors):

    return [
        ('regression', default_regression_model(4, num_anchors)),
        ('classification', default_classification_model(num_classes, num_anchors))
    ]


def __build_model_pyramid(name, model, features):

    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):

    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):

    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet(
    inputs,
    backbone_layers,
    num_classes,
    num_anchors             = None,
    create_pyramid_features = __create_pyramid_features,
    pyramid_levels          = None,
    submodels               = None,
    name                    = 'retinanet'
):

    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if 2 in pyramid_levels and 'C2' not in backbone_layers:
        raise ValueError("C2 not provided by backbone model. Cannot create P2 layers.")

    if 3 not in pyramid_levels or 4 not in pyramid_levels or 5 not in pyramid_levels:
        raise ValueError("pyramid levels 3, 4, and 5 required for functionality")

    features = create_pyramid_features(backbone_layers, pyramid_levels)
    feature_list = [features['P{}'.format(p)] for p in pyramid_levels]

    pyramids = __build_pyramid(submodels, feature_list)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def retinanet_bbox(
    model                 = None,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    anchor_params         = None,
    pyramid_levels        = None,
    nms_threshold         = 0.5,
    score_threshold       = 0.05,
    max_detections        = 300,
    parallel_iterations   = 32,
    **kwargs
):

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if model is None:
        model = retinanet(num_anchors=anchor_params.num_anchors(), **kwargs)
    else:
        assert_training_model(model)

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    assert len(pyramid_levels) == len(anchor_params.sizes), \
        "number of pyramid levels {} should match number of anchor parameter sizes {}".format(len(pyramid_levels),
                                                                                              len(anchor_params.sizes))

    pyramid_layer_names = ['P{}'.format(p) for p in pyramid_levels]

    features = [model.get_layer(p_name).output for p_name in pyramid_layer_names]
    anchors  = __build_anchors(anchor_params, features)

    regression     = model.outputs[0]
    classification = model.outputs[1]

    other = model.outputs[2:]

    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    detections = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        name                  = 'filtered_detections',
        nms_threshold         = nms_threshold,
        score_threshold       = score_threshold,
        max_detections        = max_detections,
        parallel_iterations   = parallel_iterations
    )([boxes, classification] + other)

    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)
