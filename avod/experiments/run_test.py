"""Detection model inferencing.

This runs the DetectionModel evaluator in test mode to output detections.
"""

import time
import argparse
import os
import sys
import numpy as np
import tensorflow as tf

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel
from avod.core.evaluator import Evaluator

from avod.core import trainer_utils
from wavedata.wavedata.tools.core import calib_utils
from wavedata.wavedata.tools.visualization import vis_utils
from avod.core import box_3d_encoder
from avod.core import box_3d_projector
from demos.show_predictions_2d import draw_predictions
from demos.kitti_bev_vis import draw_boxes
import cv2
from PIL import Image

def get_avod_predicted_boxes_3d_and_scores(predictions):
    """Returns the predictions and scores stacked for saving to file.

    Args:
        predictions: A dictionary containing the model outputs.

    Returns:
        predictions_and_scores: A numpy array of shape
            (number_of_predicted_boxes, 9), containing the final prediction
            boxes, orientations, scores, and types.
    """
        # boxes_3d from boxes_4c
    final_pred_boxes_3d = predictions[
        AvodModel.PRED_TOP_PREDICTION_BOXES_3D]

    # Predicted orientation from layers
    final_pred_orientations = predictions[
        AvodModel.PRED_TOP_ORIENTATIONS]

    # Calculate difference between box_3d and predicted angle
    ang_diff = final_pred_boxes_3d[:, 6] - final_pred_orientations

    # Wrap differences between -pi and pi
    two_pi = 2 * np.pi
    ang_diff[ang_diff < -np.pi] += two_pi
    ang_diff[ang_diff > np.pi] -= two_pi

    def swap_boxes_3d_lw(boxes_3d):
        boxes_3d_lengths = np.copy(boxes_3d[:, 3])
        boxes_3d[:, 3] = boxes_3d[:, 4]
        boxes_3d[:, 4] = boxes_3d_lengths
        return boxes_3d

    pi_0_25 = 0.25 * np.pi
    pi_0_50 = 0.50 * np.pi
    pi_0_75 = 0.75 * np.pi

    # Rotate 90 degrees if difference between pi/4 and 3/4 pi
    rot_pos_90_indices = np.logical_and(pi_0_25 < ang_diff,
                                        ang_diff < pi_0_75)
    final_pred_boxes_3d[rot_pos_90_indices] = \
        swap_boxes_3d_lw(final_pred_boxes_3d[rot_pos_90_indices])
    final_pred_boxes_3d[rot_pos_90_indices, 6] += pi_0_50

    # Rotate -90 degrees if difference between -pi/4 and -3/4 pi
    rot_neg_90_indices = np.logical_and(-pi_0_25 > ang_diff,
                                        ang_diff > -pi_0_75)
    final_pred_boxes_3d[rot_neg_90_indices] = \
        swap_boxes_3d_lw(final_pred_boxes_3d[rot_neg_90_indices])
    final_pred_boxes_3d[rot_neg_90_indices, 6] -= pi_0_50

    # Flip angles if abs difference if greater than or equal to 135
    # degrees
    swap_indices = np.abs(ang_diff) >= pi_0_75
    final_pred_boxes_3d[swap_indices, 6] += np.pi

    # Wrap to -pi, pi
    above_pi_indices = final_pred_boxes_3d[:, 6] > np.pi
    final_pred_boxes_3d[above_pi_indices, 6] -= two_pi

    # Append score and class index (object type)
    final_pred_softmax = predictions[
        AvodModel.PRED_TOP_CLASSIFICATION_SOFTMAX]

    # Find max class score index
    not_bkg_scores = final_pred_softmax[:, 1:]
    final_pred_types = np.argmax(not_bkg_scores, axis=1)

    # Take max class score (ignoring background)
    final_pred_scores = np.array([])
    for pred_idx in range(len(final_pred_boxes_3d)):
        all_class_scores = not_bkg_scores[pred_idx]
        max_class_score = all_class_scores[final_pred_types[pred_idx]]
        final_pred_scores = np.append(final_pred_scores, max_class_score)

    # Stack into prediction format
    predictions_and_scores = np.column_stack(
        [final_pred_boxes_3d,
         final_pred_scores,
         final_pred_types])

    return predictions_and_scores

def test(model_config, eval_config,
              dataset_config, data_split,
              ckpt_indices):

    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)

    dataset_config.data_split = data_split
    dataset_config.data_split_dir = 'training'
    if data_split == 'test':
        dataset_config.data_split_dir = 'testing'

    eval_config.eval_mode = 'test'
    eval_config.evaluate_repeatedly = False

    dataset_config.has_labels = False
    # Enable this to see the actually memory being used
    eval_config.allow_gpu_mem_growth = True

    eval_config = config_builder.proto_to_obj(eval_config)
    # Grab the checkpoint indices to evaluate
    eval_config.ckpt_indices = ckpt_indices

    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    # Setup the model
    model_name = model_config.model_name
    # Overwrite repeated field
    model_config = config_builder.proto_to_obj(model_config)
    # Switch path drop off during evaluation
    model_config.path_drop_probabilities = [1.0, 1.0]

    with tf.Graph().as_default():
        if model_name == 'avod_model':
            model = AvodModel(model_config,
                              train_val_test=eval_config.eval_mode,
                              dataset=dataset)
        elif model_name == 'rpn_model':
            model = RpnModel(model_config,
                             train_val_test=eval_config.eval_mode,
                             dataset=dataset)
        else:
            raise ValueError('Invalid model name {}'.format(model_name))

        #model_evaluator = Evaluator(model, dataset_config, eval_config)
        #model_evaluator.run_latest_checkpoints()

        # Create a variable tensor to hold the global step
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        allow_gpu_mem_growth = eval_config.allow_gpu_mem_growth
        if allow_gpu_mem_growth:
            # GPU memory config
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = allow_gpu_mem_growth
            _sess = tf.Session(config=config)
        else:
            _sess = tf.Session()

        _prediction_dict = model.build()
        _saver = tf.train.Saver()

        trainer_utils.load_checkpoints(model_config.paths_config.checkpoint_dir,
                                       _saver)
        num_checkpoints = len(_saver.last_checkpoints)
        print("test:",num_checkpoints)
        checkpoint_to_restore = _saver.last_checkpoints[num_checkpoints-1]

        _saver.restore(_sess, checkpoint_to_restore)

        num_samples = model.dataset.num_samples
        num_valid_samples = 0

        current_epoch = model.dataset.epochs_completed
        while current_epoch == model.dataset.epochs_completed:
            # Keep track of feed_dict speed
            start_time = time.time()
            feed_dict = model.create_feed_dict()
            feed_dict_time = time.time() - start_time

            # Get sample name from model
            sample_name = model.sample_info['sample_name']

            num_valid_samples += 1
            print("Step: {} / {}, Inference on sample {}".format(
                num_valid_samples, num_samples,
                sample_name))

            print("test mode")
            inference_start_time = time.time()
            # Don't calculate loss or run summaries for test
            predictions = _sess.run(_prediction_dict,
                                         feed_dict=feed_dict)
            inference_time = time.time() - inference_start_time

            print("inference time:", inference_time)

            predictions_and_scores = get_avod_predicted_boxes_3d_and_scores(predictions)

            #print(predictions_and_scores)
            #im_path = os.path.join(dataset_dir, 'training/image_2/{:06d}.png'.format(img_idx))
            #im = cv2.imread(im_path)
            #cv2.imshow('result',im)
            #cv2.waitKey(30)

            prediction_boxes_3d = predictions_and_scores[:, 0:7]
            prediction_scores = predictions_and_scores[:, 7]
            prediction_class_indices = predictions_and_scores[:, 8]
            gt_classes = ['Car']
            fig_size = (10, 6.1)

            avod_score_threshold = 0.1
            if len(prediction_boxes_3d) > 0:

                # Apply score mask
                avod_score_mask = prediction_scores >= avod_score_threshold
                prediction_boxes_3d = prediction_boxes_3d[avod_score_mask]
                prediction_scores = prediction_scores[avod_score_mask]
                prediction_class_indices = \
                    prediction_class_indices[avod_score_mask]

            if len(prediction_boxes_3d) > 0:

                dataset_dir = model.dataset.dataset_dir
                sample_name = (model.dataset.sample_names[model.dataset._index_in_epoch - 1])
                img_idx = int(sample_name)
                print("frame_index",img_idx)
                image_path = model.dataset.get_rgb_image_path(sample_name)
                image = Image.open(image_path)
                image_size = image.size

                if model.dataset.has_labels:
                    gt_objects = obj_utils.read_labels(dataset.label_dir, img_idx)
                else:
                    gt_objects = []
                filtered_gt_objs = model.dataset.kitti_utils.filter_labels(
                    gt_objects, classes=gt_classes)

                stereo_calib = calib_utils.read_calibration(dataset.calib_dir,
                                                            img_idx)
                calib_p2 = stereo_calib.p2
                # Project the 3D box predictions to image space
                image_filter = []
                final_boxes_2d = []
                for i in range(len(prediction_boxes_3d)):
                    box_3d = prediction_boxes_3d[i, 0:7]
                    img_box = box_3d_projector.project_to_image_space(
                        box_3d, calib_p2,
                        truncate=True, image_size=image_size,
                        discard_before_truncation=False)
                    if img_box is not None:
                        image_filter.append(True)
                        final_boxes_2d.append(img_box)
                    else:
                        image_filter.append(False)
                final_boxes_2d = np.asarray(final_boxes_2d)
                final_prediction_boxes_3d = prediction_boxes_3d[image_filter]
                final_scores = prediction_scores[image_filter]
                final_class_indices = prediction_class_indices[image_filter]

                num_of_predictions = final_boxes_2d.shape[0]

                # Convert to objs
                final_prediction_objs = \
                    [box_3d_encoder.box_3d_to_object_label(
                        prediction, obj_type='Prediction')
                        for prediction in final_prediction_boxes_3d]
                for (obj, score) in zip(final_prediction_objs, final_scores):
                    obj.score = score

                pred_fig, pred_2d_axes, pred_3d_axes = \
                    vis_utils.visualization(dataset.rgb_image_dir,
                                            img_idx,
                                            display=False,
                                            fig_size=fig_size)

                draw_predictions(filtered_gt_objs,
                                 calib_p2,
                                 num_of_predictions,
                                 final_prediction_objs,
                                 final_class_indices,
                                 final_boxes_2d,
                                 pred_2d_axes,
                                 pred_3d_axes,
                                 True,
                                 True,
                                 gt_classes,
                                 False)

                #cv2.imshow('result',pred_fig)
                print(type(pred_fig))
                pred_fig.canvas.draw()
                img = np.fromstring(pred_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img  = img.reshape(pred_fig.canvas.get_width_height()[::-1] + (3,))
                cv2.imshow('result',img)

                #draw bird view
                kitti_utils = model.dataset.kitti_utils
                print(img.shape[0:2])
                point_cloud = kitti_utils.get_point_cloud(
                    'lidar', img_idx, (370, 1242))
                ground_plane = kitti_utils.get_ground_plane(sample_name)
                bev_images = kitti_utils.create_bev_maps(point_cloud, ground_plane)

                density_map = np.array(bev_images.get("density_map"))
                _, box_points_norm = box_3d_projector.project_to_bev(
                    final_prediction_boxes_3d, [[-40, 40], [0, 70]])
                density_map = draw_boxes(density_map, box_points_norm)
                cv2.imshow('lidar',density_map)
                cv2.waitKey(-1)

def main(_):

    experiment_config = 'avod_cars_example.config'

    # Read the config from the experiment folder
    experiment_config_path = avod.root_dir() + '/data/outputs/' +\
        'avod_cars_example' + '/' + experiment_config

    model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            experiment_config_path, is_training=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test(model_config, eval_config,
              dataset_config, 'val', -1)


if __name__ == '__main__':
    tf.app.run()
