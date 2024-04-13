import os,sys
import numpy as np
import pandas as pd
import gc
gc.collect()
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
PROJ_DIRECTORY = "D:\\spacecraft"
DATA_DIRECTORY = PROJ_DIRECTORY + "\\data"
IMAGES_DIRECTORY = DATA_DIRECTORY + "\\1\\images"
import json
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from keras.models import Model, Sequential
from keras.layers import *
from keras.applications.vgg16 import *
from logger import *
from keras.applications import ResNet101

####################################
### create RESNET + FPN backbone ###
#################################### 

def build_resnet_backbone(input_tensor):
    # Load a ResNet101 model pre-trained on ImageNet without the top classification layer
    base_model = ResNet101(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # Extract specific feature maps for FPN
    c2_output = base_model.get_layer("conv2_block3_out").output  # 256-d, 1/4 size : 56, 56, 256
    c3_output = base_model.get_layer("conv3_block4_out").output  # 512-d, 1/8 size : 28, 28, 512
    c4_output = base_model.get_layer("conv4_block23_out").output  # 1024-d, 1/16 size : 14, 14, 1024
    c5_output = base_model.get_layer("conv5_block3_out").output  # 2048-d, 1/32 size : 7. 7. 2048

    return c2_output, c3_output, c4_output, c5_output

def build_fpn(c2, c3, c4, c5, feature_size=16):
    # FPN construction starts from the top layer downwards
    p5 = Conv2D(feature_size, (1, 1), name='fpn_p5')(c5)
    p5_upsampled = UpSampling2D(size=(2, 2), name='fpn_p5_upsampled')(p5)
    p5_output = Conv2D(feature_size, (3, 3), padding='same', name='fpn_p5_output')(p5)
    
    # Add and merge feature maps
    p4 = Conv2D(feature_size, (1, 1), name='fpn_c4p4')(c4)
    p4 = Add(name='fpn_p4add')([p5_upsampled, p4])
    p4_upsampled = UpSampling2D(size=(2, 2), name='fpn_p4_upsampled')(p4)
    p4_output = Conv2D(feature_size, (3, 3), padding='same', name='fpn_p4_output')(p4)
    
    p3 = Conv2D(feature_size, (1, 1), name='fpn_c3p3')(c3)
    p3 = Add(name='fpn_p3add')([p4_upsampled, p3])
    p3_upsampled = UpSampling2D(size=(2, 2), name='fpn_p3_upsampled')(p3)
    p3_output = Conv2D(feature_size, (3, 3), padding='same', name='fpn_p3_output')(p3)
    
    p2 = Conv2D(feature_size, (1, 1), name='fpn_c2p2')(c2)
    p2 = Add(name='fpn_p2add')([p3_upsampled, p2])
    p2_output = Conv2D(feature_size, (3, 3), padding='same', name='fpn_p2_output')(p2)

    # Optionally, add a P6 layer by downsampling P5
    p6_output = Conv2D(feature_size, (3, 3), strides=(2, 2), padding='same', name='fpn_p6')(p5)

    return p2_output, p3_output, p4_output, p5_output, p6_output

def resnet_fpn(input_shape=(None, None, 3)):
    input_tensor = Input(shape=input_shape)
    c2, c3, c4, c5 = build_resnet_backbone(input_tensor)
    p2, p3, p4, p5, p6 = build_fpn(c2, c3, c4, c5)
    return Model(inputs=[input_tensor], outputs=[p2, p3, p4, p5, p6])

####################################
### create FPN + RPN backbone ######
#################################### 

def load_and_process_image(image_path):
    """Load and preprocess an image from a path."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalize to [0,1]
    return image

def generate_anchors(fpn_feature_maps, ratios = [0.5, 1., 2.], scales = [2 ** (x/3) for x in range(3)], feature_size =[56., 28., 14., 7., 4.]):
    anchors_all = []
    for ix, fpn_feature_map in enumerate(fpn_feature_maps):
        base_size = fpn_feature_map.shape[1]
        anchors = []
        for i in range(fpn_feature_map.shape[-1]):
            anchors1 = []
            for scale in scales:
                for ratio in ratios:
                    # Calculate anchor dimensions based on scale and ratio
                    anchor_height = base_size * scale * np.sqrt(ratio) /2
                    anchor_width = base_size * scale / np.sqrt(ratio) /2
                    
                    # Generate anchors across the feature map
                    
                    x = feature_size[ix]
                    y = feature_size[ix]
                    center_x = (x + 0.5) /2
                    center_y = (y + 0.5) /2
                    anchors1.append([
                        int(max(center_x - anchor_width / 2, 0.)),
                        int(max(center_y - anchor_height / 2, 0.)),
                        int(min(center_x + anchor_width / 2, x)),
                        int(min(center_y + anchor_height / 2, x)),
                    ])
            anchors.append(anchors1)
    
        anchors_all.append(anchors)
    anchors_all = np.array(anchors_all)
    return anchors_all

def crop_regions_from_feature_map(feature_map, anchors):
    crops = []
    for i in range(anchors.shape[0]): #5 # Iterate through all the anchors
        crops2 = []
        for j in range(anchors.shape[1]): # 16   # Iterate through each of the 9 bboxes for the anchor
            crops1 = []
            for k in range(anchors.shape[2]):   # 9
                anchor = anchors[i, j, k]
                crop = tf.image.crop_to_bounding_box(tf.expand_dims(feature_map[i][0,:,:,j], -1),
                                                    offset_height=anchor[0],
                                                    offset_width=anchor[1],
                                                    target_height=anchor[2] - anchor[0],
                                                    target_width=anchor[3] - anchor[1])
                crop = tf.squeeze(crop, axis=-1)
                crops1.append(crop)
            crops2.append(crops1)    
        crops.append(crops2)    
    return crops

##########################################
### create tf.dataset for training #######
##########################################

def process_batch(image_batch,image_size = (224,224,3)):
    """
    Process a batch of images through the FPN, generate anchors,
    crop regions from the FPN feature maps, and perform ROI Align.
    """
    # Generate FPN feature maps
    backbone = resnet_fpn(image_size)
    fpn_feature_maps = backbone(image_batch, training=False)

    # Generate anchors based on the FPN feature maps
    anchors = generate_anchors(fpn_feature_maps)


    # Crop regions from the feature maps using the generated anchors
    cropped_regions = crop_regions_from_feature_map(fpn_feature_maps, anchors)
    # Perform ROI Align on the cropped regions
    roiali = roi_align(cropped_regions)
    return roiali

def roi_align(cropped_regions):
    # Normalize proposals from [0, 1] to feature map scale
    pooled_list = []

    for cropped_region in cropped_regions:
        for cropped in cropped_region:
            for crop in cropped:
                crop = tf.expand_dims(crop, axis = -1)
                resized_data = tf.image.resize(crop, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
                pooled_list.append(resized_data)
    finalfpn = tf.concat(pooled_list, axis=-1)
    return finalfpn

def fpn_dataflow(image_paths, batch_size=2):
    """Create a dataset that processes images through the FPN to generate ROIALI."""
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_process_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

##########################################
########### model for bbox ###############
##########################################
def create_bbox_model(input_shape=(224, 224, 720)):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='sigmoid')  # Output layer; using sigmoid to ensure outputs are between 0 and 1
    ])
    return model


def load_and_process_image_bbox(image_path, bbox):
    """Load and preprocess an image from a path."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalize to [0,1]
    print(image.shape)
    return image, bbox

def process_batch_bbox(image_batch, bbox, image_size = (224,224,3)):
    """
    Process a batch of images through the FPN, generate anchors,
    crop regions from the FPN feature maps, and perform ROI Align.
    """
    # Generate FPN feature maps
    backbone = resnet_fpn(image_size)
    fpn_feature_maps = backbone(image_batch, training=False)

    # Generate anchors based on the FPN feature maps
    anchors = generate_anchors(fpn_feature_maps)


    # Crop regions from the feature maps using the generated anchors
    cropped_regions = crop_regions_from_feature_map(fpn_feature_maps, anchors)
    # Perform ROI Align on the cropped regions
    roiali = roi_align(cropped_regions)
    return roiali, bbox

def bbox_dataflow(bbox, batch_size=2):
    bbox = [[i[0]/1280*224,i[1]/1024*224,i[2]/1280*224,i[3]/1024*224] for i in bbox]
    dataset = tf.data.Dataset.from_tensor_slices(bbox)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def bbox_fpn_datasets(image_paths, bboxes, batch_size=2):
    # Assuming bbox_dataflow and fpn_dataflow are correctly defined and return batched datasets
    bbox_dataset = bbox_dataflow(bboxes, batch_size=batch_size)
    image_dataset = fpn_dataflow(image_paths, batch_size=batch_size)
    
    # Combine the two datasets
    combined_dataset = tf.data.Dataset.zip((image_dataset, bbox_dataset))
    
    return combined_dataset

##########################################
####### model for classification #########
##########################################

def create_classification_model(input_shape=(224, 224, 720), num_classes=30):
    model = Sequential([
        # Convolutional base
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Dense layers for classification
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
    ])
    
    return model

def one_hot_encode_labels(label, num_classes):
    return tf.one_hot(label, depth=num_classes)

def classification_dataflow(classifications, num_classes, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices(classifications)
    # Use a lambda to pass num_classes to one_hot_encode_labels
    dataset = dataset.map(lambda label: one_hot_encode_labels(label, num_classes))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def class_fpn_datasets(image_paths, classification, num_classes = 30, batch_size=2):
    class_dataset = classification_dataflow(classification,num_classes, batch_size=batch_size)
    image_dataset = fpn_dataflow(image_paths, batch_size=batch_size)
    
    # Combine the two datasets
    combined_dataset = tf.data.Dataset.zip((image_dataset, class_dataset))
    
    return combined_dataset
    

##########################################
########### model for mask ###############
##########################################
def conv_block(input_tensor, num_filters):
    """A convolutional block."""
    x = layers.Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x

def encoder_block(input_tensor, num_filters):
    """An encoder block (downsampling)"""
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    """A decoder block (upsampling)"""
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
    x = layers.concatenate([x, concat_tensor], axis=-1)
    x = conv_block(x, num_filters)
    return x

def create_mask_fpn_model(input_shape=(224, 224, 720)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1, p1 = encoder_block(inputs, 16)
    c2, p2 = encoder_block(p1, 32)
    c3, p3 = encoder_block(p2, 64)
    c4, p4 = encoder_block(p3, 128)

    # Bottleneck
    b = conv_block(p4, 256)

    # Decoder
    d1 = decoder_block(b, c4, 128)
    d2 = decoder_block(d1, c3, 64)
    d3 = decoder_block(d2, c2, 32)
    d4 = decoder_block(d3, c1, 16)

    # Output
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(d4)  # Assuming binary segmentation; adjust filters for multiclass

    model = Model(inputs, outputs, name="U-Net")
    return model

def load_and_process_mask(image_path):
    """Load and preprocess an image from a path, converting it to grayscale."""
    # Load the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Decode as a three-channel image
    image = tf.image.resize(image, [224, 224])  # Resize to the target size
    
    # Convert the image to grayscale
    image_grayscale = tf.image.rgb_to_grayscale(image)
    
    # Normalize the grayscale image to [0,1]
    image_normalized = image_grayscale / 255.0
    return image_normalized

def mask_dataflow(mask_list, batch_size = 2):
    dataset = tf.data.Dataset.from_tensor_slices(mask_list)
    dataset = dataset.map(load_and_process_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
    
def mask_fpn_dataflow(mask_list, image_paths, batch_size =2):
    mask_dataset = mask_dataflow(mask_list, batch_size=batch_size)
    image_dataset = fpn_dataflow(image_paths, batch_size=batch_size)
    
    # Combine the two datasets
    combined_dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
    
    return combined_dataset
    
    
