

import keras
from .. import backend

def filter_detections(
    boxes,
    classification,
    other                 = [],
    class_specific_filter = True,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 300,
    nms_threshold         = 0.5
):

    def _filter_detections(scores, labels):

        indices = backend.where(keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes  = backend.gather_nd(boxes, indices)
            filtered_scores = keras.backend.gather(scores, indices)[:, 0]

            nms_indices = backend.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

            indices = keras.backend.gather(indices, nms_indices)

        labels = backend.gather_nd(labels, indices)
        indices = keras.backend.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []

        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores  = keras.backend.max(classification, axis    = 1)
        labels  = keras.backend.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)

    scores              = backend.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    indices             = keras.backend.gather(indices[:, 0], top_indices)
    boxes               = keras.backend.gather(boxes, indices)
    labels              = keras.backend.gather(labels, top_indices)
    other_              = [keras.backend.gather(o, indices) for o in other]

    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes    = backend.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = keras.backend.cast(labels, 'int32')
    other_   = [backend.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_


class FilterDetections(keras.layers.Layer):

    def __init__(
        self,
        nms                   = True,
        class_specific_filter = True,
        nms_threshold         = 0.5,
        score_threshold       = 0.05,
        max_detections        = 300,
        parallel_iterations   = 32,
        **kwargs
    ):

        self.nms                   = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        self.parallel_iterations   = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        boxes          = inputs[0]
        classification = inputs[1]
        other          = inputs[2:]

        def _filter_detections(args):
            boxes          = args[0]
            classification = args[1]
            other          = args[2]

            return filter_detections(
                boxes,
                classification,
                other,
                nms                   = self.nms,
                class_specific_filter = self.class_specific_filter,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold,
            )

        dtypes = [keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [o.dtype for o in other]
        shapes = [(self.max_detections,4), (self.max_detections,), (self.max_detections),]
        shapes.extend([(self.max_detections,) + o.shape[2:] for o in other])
        outputs = backend.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=dtypes,
            shapes=shapes,
            parallel_iterations=self.parallel_iterations,
        )

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):

        return (len(inputs) + 1) * [None]

    def get_config(self):

        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                   : self.nms,
            'class_specific_filter' : self.class_specific_filter,
            'nms_threshold'         : self.nms_threshold,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : self.parallel_iterations,
        })

        return config
