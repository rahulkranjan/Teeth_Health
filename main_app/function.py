import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.tflite')

classes = ['normal', 'decay', 'yellow', 'rough']
#image_preprocessing
def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  resized_img = tf.cast(resized_img, dtype=tf.uint8)
  return resized_img, original_image


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  output = signature_fn(images=image)

  # Get all outputs from the model
  count = int(np.squeeze(output['output_0']))
  scores = np.squeeze(output['output_1'])
  classes = np.squeeze(output['output_2'])
  boxes = np.squeeze(output['output_3'])

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), 	(119,136,153), 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8


def show_result_table(interpreter, image_path, threshold=0.5):
  """Returns a table of class of each teeth detected"""

   # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
    )
  #running an object detection on input image
  results = detect_objects(interpreter, preprocessed_image, threshold)

  #storing the class of each objet detected in a list
  labels = []
  for obj in results:
    class_id = int(obj['class_id'])
    label = "{}".format(classes[class_id])
    labels.append(label)

  #converting the list to a dataframe
  output_table = pd.DataFrame(labels)
  return output_table

def final_result(output_table):

  yellow = np.sum(output_table == 'yellow')
  normal = np.sum(output_table == 'normal')
  rough = np.sum(output_table == 'rough')
  decay = np.sum(output_table == 'decay')
  total = yellow + normal + rough + decay

  smoothness = float(((yellow*0.65) + normal)*100/total)
  sign_of_decay = float((decay + (0.35*yellow))*100/total)
  bad_breath = float(((yellow*0.85) + (decay*0.65))*100/total)
  stain = float(((yellow*0.8)+(decay*0.7))*100/total)
  sensitivity = float(((yellow*0.45)+(rough*0.55)+(decay*0.65))*100/total)

  final_output = """
            Smoothness: {}
            Sign of decay: {}
            Bad Breath: {}
            Stain level: {}
            Sensitivity: {}""".format(smoothness, sign_of_decay, bad_breath, stain, sensitivity)

  return final_output
