## Credit: Some parts of the program has been taken from OpenCV documentation
#importing required libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf

#function to load TensorFlow graph from a model file
def load_graph(model_file):
  graph = tf.Graph()  #creating a tensorflow computation graph
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())  #parsing binary graph definition
  with graph.as_default():       #setting this graph as default computation graph
    tf.import_graph_def(graph_def)     #importing graph definitions into current graph

  return graph

#function to read and pre-process the image
def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):   # if a PNG image, setting the number of color channels to 3
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):   # if a GIF image, removing the singleton dimension
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):    # if bmp, then decoding a BMP image
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:                                #default: decoding the image as a JPEG with 3 color channels
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)  #converting the image into float32 dtype
  dims_expander = tf.expand_dims(float_caster, 0); #adding batch dimension 
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width]) #resizing the image
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])  #normalizing the image
  sess = tf.Session()
  result = sess.run(normalized)

  return result

#function for loading labels from a file
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())  #appending labels after stripping newline characters
  return label

#main function for image classification
def main(img):
  file_name = img
  model_file = "retrained_graph.pb"
  label_file = "retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  #parsing command-line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  #over-riding default values with command line arguments(if provided)
  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(file_name,                 #reading and pre-processing the image input
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);  # obtaining references to the input and output operations within the graph
  output_operation = graph.get_operation_by_name(output_name);

  #running the image through the model
  with tf.Session(graph=graph) as sess:
    start = time.time() #starting the timer
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()  #recording the end time for measuring performance
  results = np.squeeze(results) #removing dimensions of size 1, making it a 1D Array

  #identifying the top k results
  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)

  for i in top_k:
    return labels[i] #returning the label with highest confidence
