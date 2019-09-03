#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：Create on 2019/7/18 14:31 by jade
# 邮箱：jadehh@live.com
# 描述：目标检测类
# 最近修改：2019/7/18 14:31 modify by jade
import tensorflow as tf
from jade import *
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
import cv2
import os
import os.path as ops
import re
import time
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.core import standard_fields as fields
from tensorflow.python.training import saver as saver_lib
slim = tf.contrib.slim

def _image_tensor_input_placeholder(input_shape=None):
  """Returns input placeholder and a 4-D uint8 image tensor."""
  if input_shape is None:
    input_shape = (None, None, None, 3)
  input_tensor = tf.placeholder(
      dtype=tf.uint8, shape=input_shape, name='image_tensor')
  return input_tensor, input_tensor


def _tf_example_input_placeholder():
  """Returns input that accepts a batch of strings with tf examples.

  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_tf_example_placeholder = tf.placeholder(
      tf.string, shape=[None], name='tf_example')
  def decode(tf_example_string_tensor):
    tensor_dict = tf_example_decoder.TfExampleDecoder().decode(
        tf_example_string_tensor)
    image_tensor = tensor_dict[fields.InputDataFields.image]
    return image_tensor
  return (batch_tf_example_placeholder,
          tf.map_fn(decode,
                    elems=batch_tf_example_placeholder,
                    dtype=tf.uint8,
                    parallel_iterations=32,
                    back_prop=False))


def _encoded_image_string_tensor_input_placeholder():
  """Returns input that accepts a batch of PNG or JPEG strings.

  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_image_str_placeholder = tf.placeholder(
      dtype=tf.string,
      shape=[None],
      name='encoded_image_string_tensor')
  def decode(encoded_image_string_tensor):
    image_tensor = tf.image.decode_image(encoded_image_string_tensor,
                                         channels=3)
    image_tensor.set_shape((None, None, 3))
    return image_tensor
  return (batch_image_str_placeholder,
          tf.map_fn(
              decode,
              elems=batch_image_str_placeholder,
              dtype=tf.uint8,
              parallel_iterations=32,
              back_prop=False))

input_placeholder_fn_map = {
    'image_tensor': _image_tensor_input_placeholder,
    'encoded_image_string_tensor':
    _encoded_image_string_tensor_input_placeholder,
    'tf_example': _tf_example_input_placeholder,
}
def _add_output_tensor_nodes(postprocessed_tensors,
                             output_collection_name='inference_op'):
  """Adds output nodes for detection boxes and scores.

  Adds the following nodes for output tensors -
    * num_detections: float32 tensor of shape [batch_size].
    * detection_boxes: float32 tensor of shape [batch_size, num_boxes, 4]
      containing detected boxes.
    * detection_scores: float32 tensor of shape [batch_size, num_boxes]
      containing scores for the detected boxes.
    * detection_classes: float32 tensor of shape [batch_size, num_boxes]
      containing class predictions for the detected boxes.
    * detection_keypoints: (Optional) float32 tensor of shape
      [batch_size, num_boxes, num_keypoints, 2] containing keypoints for each
      detection box.
    * detection_masks: (Optional) float32 tensor of shape
      [batch_size, num_boxes, mask_height, mask_width] containing masks for each
      detection box.

  Args:
    postprocessed_tensors: a dictionary containing the following fields
      'detection_boxes': [batch, max_detections, 4]
      'detection_scores': [batch, max_detections]
      'detection_classes': [batch, max_detections]
      'detection_masks': [batch, max_detections, mask_height, mask_width]
        (optional).
      'num_detections': [batch]
    output_collection_name: Name of collection to add output tensors to.

  Returns:
    A tensor dict containing the added output tensor nodes.
  """
  detection_fields = fields.DetectionResultFields
  label_id_offset = 1
  boxes = postprocessed_tensors.get(detection_fields.detection_boxes)
  scores = postprocessed_tensors.get(detection_fields.detection_scores)
  classes = postprocessed_tensors.get(
      detection_fields.detection_classes) + label_id_offset
  keypoints = postprocessed_tensors.get(detection_fields.detection_keypoints)
  masks = postprocessed_tensors.get(detection_fields.detection_masks)
  num_detections = postprocessed_tensors.get(detection_fields.num_detections)
  outputs = {}
  outputs[detection_fields.detection_boxes] = tf.identity(
      boxes, name=detection_fields.detection_boxes)
  outputs[detection_fields.detection_scores] = tf.identity(
      scores, name=detection_fields.detection_scores)
  outputs[detection_fields.detection_classes] = tf.identity(
      classes, name=detection_fields.detection_classes)
  outputs[detection_fields.num_detections] = tf.identity(
      num_detections, name=detection_fields.num_detections)
  if keypoints is not None:
    outputs[detection_fields.detection_keypoints] = tf.identity(
        keypoints, name=detection_fields.detection_keypoints)
  if masks is not None:
    outputs[detection_fields.detection_masks] = tf.identity(
        masks, name=detection_fields.detection_masks)
  for output_key in outputs:
    tf.add_to_collection(output_collection_name, outputs[output_key])

  return outputs

def _get_outputs_from_inputs(input_tensors, detection_model,
                             output_collection_name):
  inputs = tf.to_float(input_tensors)
  preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)
  output_tensors = detection_model.predict(
      preprocessed_inputs, true_image_shapes)
  postprocessed_tensors = detection_model.postprocess(
      output_tensors, true_image_shapes)
  return _add_output_tensor_nodes(postprocessed_tensors,
                                  output_collection_name)

def _build_detection_graph(input_type, detection_model, input_shape,
                           output_collection_name, graph_hook_fn):
  """Build the detection graph."""
  if input_type not in input_placeholder_fn_map:
    raise ValueError('Unknown input type: {}'.format(input_type))
  placeholder_args = {}
  if input_shape is not None:
    if input_type != 'image_tensor':
      raise ValueError('Can only specify input shape for `image_tensor` '
                       'inputs.')
    placeholder_args['input_shape'] = input_shape
  placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type](
      **placeholder_args)
  outputs = _get_outputs_from_inputs(
      input_tensors=input_tensors,
      detection_model=detection_model,
      output_collection_name=output_collection_name)

  # Add global step to the graph.
  slim.get_or_create_global_step()

  if graph_hook_fn: graph_hook_fn()

  return outputs, placeholder_tensor


def profile_inference_graph(graph):
  """Profiles the inference graph.

  Prints model parameters and computation FLOPs given an inference graph.
  BatchNorms are excluded from the parameter count due to the fact that
  BatchNorms are usually folded. BatchNorm, Initializer, Regularizer
  and BiasAdd are not considered in FLOP count.

  Args:
    graph: the inference graph.
  """
  tfprof_vars_option = (
      tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  tfprof_flops_option = tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS

  # Batchnorm is usually folded during inference.
  tfprof_vars_option['trim_name_regexes'] = ['.*BatchNorm.*']
  # Initializer and Regularizer are only used in training.
  tfprof_flops_option['trim_name_regexes'] = [
      '.*BatchNorm.*', '.*Initializer.*', '.*Regularizer.*', '.*BiasAdd.*'
  ]

  # tf.contrib.tfprof.model_analyzer.print_model_analysis(
  #     graph,
  #     tfprof_options=tfprof_vars_option)
  #
  # tf.contrib.tfprof.model_analyzer.print_model_analysis(
  #     graph,
  #     tfprof_options=tfprof_flops_option)


def export_inference_graph(input_type,
                            detection_model,
                            use_moving_averages,
                            trained_checkpoint_prefix,
                            additional_output_tensor_names=None,
                            input_shape=None,
                            output_collection_name='inference_op',
                            graph_hook_fn=None,
                            write_inference_graph=False,
                            gpu_memory_fraction=0.8):
  """Export helper."""
  outputs, placeholder_tensor = _build_detection_graph(
      input_type=input_type,
      detection_model=detection_model,
      input_shape=input_shape,
      output_collection_name=output_collection_name,
      graph_hook_fn=graph_hook_fn)

  profile_inference_graph(tf.get_default_graph())
  saver_kwargs = {}
  if use_moving_averages:
    # This check is to be compatible with both version of SaverDef.
    if os.path.isfile(trained_checkpoint_prefix):
      saver_kwargs['write_version'] = saver_pb2.SaverDef.V1
      temp_checkpoint_prefix = tempfile.NamedTemporaryFile().name
    else:
      temp_checkpoint_prefix = tempfile.mkdtemp()
    replace_variable_values_with_moving_averages(
        tf.get_default_graph(), trained_checkpoint_prefix,
        temp_checkpoint_prefix)
    checkpoint_to_use = temp_checkpoint_prefix
  else:
    checkpoint_to_use = trained_checkpoint_prefix

  saver = tf.train.Saver(**saver_kwargs)
  input_saver_def = saver.as_saver_def()

  sess = write_graph_and_checkpoint(
      inference_graph_def=tf.get_default_graph().as_graph_def(),
      input_saver_def=input_saver_def,
      trained_checkpoint_prefix=checkpoint_to_use,
      gpu_memory_fraction=gpu_memory_fraction)
  return sess



def write_graph_and_checkpoint(inference_graph_def,
                               input_saver_def,
                               trained_checkpoint_prefix,
                               gpu_memory_fraction=0.8):
  """Writes the graph and the checkpoint into disk."""
  gpu_options1 = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options1))
  sess.run(tf.global_variables_initializer())
  tf.import_graph_def(inference_graph_def, name='')
  saver = saver_lib.Saver(saver_def=input_saver_def,
                          save_relative_paths=True)
  saver.restore(sess, trained_checkpoint_prefix)
  return sess











def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        print("restore the last model")
    ckpt = tf.train.get_checkpoint_state(model_dir)
    meta_file = ""
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        meta_file = ckpt_file + ".meta"
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

class ObjectModel:
    def __init__(self,args):
        self.args = args
        self.model_path = args.model_path
        self.label_path = args.label_path
        self.num_classes = args.num_classes
        self.category_index,_ = ReadProTxt(args.label_path)
        self.image_size = (12, 8)
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.pipline_config_path = os.path.join(self.model_path,"pipeline.config")
        self.sess = self.setup_net()


    def setup_net(self):
        if (os.path.isfile(self.model_path)):
            model_exp = os.path.expanduser(self.model_path)
            g1 = tf.Graph()
            gpu_options1 = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options1), graph=g1)
            print('Model filename: %s' % model_exp)
            with sess.as_default():
                with g1.as_default():
                    with tf.gfile.GFile(self.model_path, 'rb') as fid:
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(fid.read())
                        tf.import_graph_def(graph_def, name="")
                    tf.global_variables_initializer().run()
            print("load {} into graph done ..".format(self.model_path))
        else:
            meta,ckpt = get_model_filenames(self.model_path)
            print('Model filename: %s' % (self.model_path+"/"+meta))
            trained_checkpoint_prefix = os.path.join(self.model_path,ckpt)
            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            with tf.gfile.GFile(self.pipline_config_path, 'r') as f:
                text_format.Merge(f.read(), pipeline_config)
            text_format.Merge("", pipeline_config)
            detection_model = model_builder.build(pipeline_config.model,
                                                  is_training=False)
            graph_rewriter_fn = None
            if pipeline_config.HasField('graph_rewriter'):
                graph_rewriter_config = pipeline_config.graph_rewriter
                graph_rewriter_fn = graph_rewriter_builder.build(graph_rewriter_config,
                                                                 is_training=False)
            sess = export_inference_graph(
                    "image_tensor",
                    detection_model,
                    pipeline_config.eval_config.use_moving_averages,
                    trained_checkpoint_prefix,
                    None,
                    None,
                    "inference_op",
                    graph_hook_fn=None,
                    write_inference_graph=False,
                    gpu_memory_fraction=self.gpu_memory_fraction)
            print("load {} into graph done ..".format(self.model_path))

        return sess


    def predict(self,img,select_threshold=0.6):
        def predict_pb(img):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes',"detection_masks"
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, img.shape[0], img.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')
            output_dict = self.sess.run(tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(img, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            classes = output_dict["detection_classes"]
            scores = output_dict["detection_scores"]
            bboxes = output_dict["detection_boxes"]
            height = img.shape[0]
            width = img.shape[1]
            boxes_out = []
            scores_out = []
            classes_out = []
            class_names_out = []
            for i in range(classes.shape[0]):
                cls_id = int(classes[i])
                if cls_id in self.category_index.keys():
                    class_name = self.category_index[cls_id]['display_name']
                else:
                    class_name = 'N/A'
                if cls_id >= 0:  # and class_name == "person":
                    score = scores[i]
                    if score >= select_threshold:
                        ymin = int(bboxes[i, 0] * height)
                        xmin = int(bboxes[i, 1] * width)
                        ymax = int(bboxes[i, 2] * height)
                        xmax = int(bboxes[i, 3] * width)
                        boxes_out.append([xmin,ymin,xmax,ymax])
                        scores_out.append(score)
                        class_names_out.append(class_name)
                        classes_out.append(cls_id)
            detectionResult = DetectResultModel(boxes_out,class_names_out,classes_out,scores_out)
            return detectionResult

        if (os.path.isfile(self.model_path)):
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    output_dict = predict_pb(img)
            return output_dict
        else:
            output_dict = predict_pb(img)
        return output_dict

    def show_result_plt(self,output_dict,image_np):
         vis_util.visualize_boxes_and_labels_on_image_array(
             image_np,
             output_dict['detection_boxes'],
             output_dict['detection_classes'],
             output_dict['detection_scores'],
             self.category_index,
             instance_masks=output_dict.get('detection_masks'),
             use_normalized_coordinates=True,
             line_thickness=8)
         plt.figure(figsize=self.image_size)
         plt.imshow(image_np)


    def show_result_cv(self,output_dict,image_np,select_threshold=0.2,wait_key=2000,windows_name="out_puts"):
        classes = output_dict["detection_classes"]
        scores = output_dict["detection_scores"]
        bboxes = output_dict["detection_boxes"]
        height = image_np.shape[0]
        width = image_np.shape[1]
        for i in range(classes.shape[0]):
            cls_id = int(classes[i])
            if cls_id in self.label_map:
                class_name = self.category_index[cls_id]['name']
            else:
                class_name = 'N/A'
            if cls_id >= 0: #and class_name == "person":
                score = scores[i]
                if score >= select_threshold:
                    ymin = int(bboxes[i, 0] * height)
                    xmin = int(bboxes[i, 1] * width)
                    ymax = int(bboxes[i, 2] * height)
                    xmax = int(bboxes[i, 3] * width)
                    colors_tableau = [(0, 0, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120)]*self.num_classes
                    cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), colors_tableau[cls_id - 1], 2, 18)
                    cv2.putText(image_np, class_name, (xmin, ymin + 90), cv2.FONT_HERSHEY_PLAIN, 2,
                                colors_tableau[cls_id - 1],
                                3)
        return image_np

    def cut_mask_img(self, output_dict, img, select_threshold=0.6):
        classes = output_dict["detection_classes"]
        scores = output_dict["detection_scores"]
        masks = output_dict["detection_masks"]
        images = []
        for i in range(len(classes)):
            score = scores[i]
            if score >= select_threshold and classes[i] == 1:
                img2 = img.copy()
                for c in range(3):
                    if self.mask_rcnn:
                        img2[:, :, c] = np.where(masks[:, :, i] == 0,
                                                 0,
                                                 img2[:, :, c])
                    else:
                        img2[:, :, c] = np.where(masks[0, :, :, ] == 0,
                                                 0,
                                                 img2[:, :, c])

                images.append(img2)
        return images
    def cut_img(self,output_dict,image_np,select_threshold=0.2,car_detect=True):
        plate_img = None
        classes = output_dict["detection_classes"]
        scores = output_dict["detection_scores"]
        bboxes = output_dict["detection_boxes"]
        height = image_np.shape[0]
        width = image_np.shape[1]
        for i in range(classes.shape[0]):
            cls_id = int(classes[i])
            if cls_id in self.category_index.keys():
                class_name = self.category_index[cls_id]['name']
            else:
                class_name = 'N/A'
            if cls_id >= 0: #and class_name == "person":
                score = scores[i]
                if score >= select_threshold:
                    ymin = int(bboxes[i, 0] * height)
                    xmin = int(bboxes[i, 1] * width)
                    ymax = int(bboxes[i, 2] * height)
                    xmax = int(bboxes[i, 3] * width)
                    if car_detect:
                        colors_tableau = [(0, 0, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120)]*200
                        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), colors_tableau[cls_id - 1], 2, 18)
                        cv2.putText(image_np, class_name, (xmin, ymin + 90), cv2.FONT_HERSHEY_PLAIN, 2,
                                    colors_tableau[cls_id - 1],
                                    3)
                    else:
                        plate_img = image_np[ymin:ymax,xmin:xmax,:]
        return image_np,plate_img

    def person_in_image(self,output_dict,image_np,select_threshold=0.6):
        classes = output_dict["detection_classes"]
        scores = output_dict["detection_scores"]
        bboxes = output_dict["detection_boxes"]
        height = image_np.shape[0]
        width = image_np.shape[1]
        for i in range(classes.shape[0]):
            cls_id = int(classes[i])
            if cls_id in self.category_index.keys():
                class_name = self.category_index[cls_id]['name']
            else:
                class_name = 'N/A'
            if cls_id >= 0 and class_name == "person":
                score = scores[i]
                if score >= select_threshold:
                    return True
        return False


    def predict_all_boxes(self, img, i, allboxes,select_threshold=0.6):
        def predict_pb(img,allboxes):
            all_boxes = allboxes
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes',"detection_masks"
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, img.shape[0], img.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')
            start_time = time.time()
            output_dict = self.sess.run(tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(img, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            classes = output_dict["detection_classes"]
            scores = output_dict["detection_scores"]
            bboxes = output_dict["detection_boxes"]
            height = img.shape[0]
            width = img.shape[1]
            boxes_out = []
            scores_out = []
            classes_out = []
            class_names_out = []
            for j in range(1,self.num_classes+1):
                dets = []
                for k in range(classes.shape[0]):
                    cId = int(classes[k])
                    class_name = self.category_index[cId]['name']
                    score = scores[k]
                    if cId == j and score > select_threshold:
                        ymin = int(bboxes[k, 0] * height)
                        xmin = int(bboxes[k, 1] * width)
                        ymax = int(bboxes[k, 2] * height)
                        xmax = int(bboxes[k, 3] * width)
                        boxes_out.append([xmin,ymin,xmax,ymax])
                        scores_out.append(score)
                        class_names_out.append(class_name)
                        classes_out.append(cId)
                        dets.append(np.reshape(np.array([xmin,ymin,xmax,ymax,score]),(1,5)))
                if len(dets) == 0:
                    allboxes[j][i] = np.empty([0, 5], dtype=np.float32)
                else:
                    allboxes[j][i] = np.reshape(np.array(dets),(len(dets),5))
            return boxes_out,class_names_out,classes_out,scores_out,all_boxes
        if (os.path.isfile(self.model_path)):
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    output_dict = predict_pb(img,allboxes)
            return output_dict
        else:
            output_dict = predict_pb(img,allboxes)
        return output_dict
if __name__ == '__main__':
    import argparse

    car_paraser = argparse.ArgumentParser("test")
    car_paraser.add_argument("--model_path",
                             default="/home/jade/Desktop/models/plate_models/ssd_mobilenet_v2_10_2",
                             help="path to load model")
    car_paraser.add_argument("--label_path", default="label_map/car_label_map.pbtxt", help="path to labels")
    car_paraser.add_argument("--num_classes", default=4, help="the number of classes")
    car_paraser.add_argument("--gpu_memory_fraction", default=0.8, help="the memory of gpu")
    car_args = car_paraser.parse_args()
    car_model = ObjectModel(car_args)
    img_path = "/home/jade/Data/PLATE_DATASET/12_PLATE/JPEGImages"
    img_list = os.listdir(img_path)
    for img_name in img_list:
        stat_time = time.time()
        img = cv2.imread(ops.join(img_path, img_name))
        car_out_dict = car_model.predict(img)
        img, plate_img = car_model.cut_img(car_out_dict, img, select_threshold=0.2, car_detect=True, )
        print("detect car and plate use {} s".format(time.time() - stat_time))
        cv2.imshow("car", img)
        cv2.waitKey(1000)

