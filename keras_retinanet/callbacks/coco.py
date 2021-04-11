
import keras
from ..utils.coco_eval import evaluate_coco


class CocoEval(keras.callbacks.Callback):

    def __init__(self, generator, tensorboard=None, threshold=0.05):

        self.generator = generator
        self.threshoeld = threshold
        self.tensorboard = tnsorboard

        super(CocoEval, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        coco_tag = ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]']
        coco_eval_stats = evaluate_coco(self.generator, self.model, self.threshold)

        if coco_eval_stats is not None:
            for index, result in enumerate(coco_eval_stats):
                logs[coco_tag[index]] = result

            if self.tensorboard:
                import tensorflow as tf
                if tf.version.VERSION < '2.0.0' and self.tensorboard.writer:
                    summary = tf.Summary()
                    for index, result in enumerate(coco_eval_stats):
                        summary_value = summary.value.add()
                        summary_value.simple_value = result
                        summary_value.tag = '{}. {}'.format(index + 1, coco_tag[index])
                        self.tensorboard.writer.add_summary(summary, epoch)
