import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenetv2 import preprocess_input

import tensorflow as tf
from tensorflow.python.framework import ops

def build_model(dataset):
    return load_model('result/models/' + dataset + '.h5')

def load_image(path, preprocess=True):
    x = image.load_img(path, target_size=(224, 224))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x

def build_guided_model(dataset):
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model(dataset)
    return new_model


def grad_cam(input_model, image, cls, layer_name):
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

def compute_saliency(model, guided_model, img_path, layer_name='block5_conv3'):
    preprocessed_input = load_image(img_path)

    predictions = model.predict(preprocessed_input)

    print('Model prediction:')
    print(predictions)

    cls = np.argmax(predictions)

    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)

    plt.axis('off')
    plt.imshow(load_image(img_path, preprocess=False))
    plt.imshow(gradcam, cmap='jet', alpha=0.5)
    plt.show()
    return gradcam

dataset = 'marque'

model = build_model(dataset)
guided_model = build_guided_model(dataset)

for file in glob.iglob('train/'+ dataset + '/*/*.png'):
    print compute_saliency(model, guided_model, layer_name='Conv_1_bn', img_path=file)
