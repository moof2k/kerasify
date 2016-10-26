import numpy as np
import struct

LAYER_DENSE = 1
LAYER_CONVOLUTION2D = 2
LAYER_FLATTEN = 3
LAYER_ELU = 4
LAYER_ACTIVATION = 5

ACTIVATION_LINEAR = 1
ACTIVATION_RELU = 2
ACTIVATION_SOFTPLUS = 3

def write_floats(file, floats):
    '''
    Writes floats to file in 1024 chunks.. prevents memory explosion
    writing very large arrays to disk when calling struct.pack().
    '''
    step = 1024
    written = 0

    for i in np.arange(0, len(floats), step):
        remaining = min(len(floats) - i, step)
        written += remaining
        file.write(struct.pack('=%sf' % remaining, *floats[i:i+remaining]))

    assert written == len(floats)

def export_model(model, filename):
    with open(filename, 'wb') as f:

        def write_activation(activation):
            if activation == 'linear':
                f.write(struct.pack('I', ACTIVATION_LINEAR))
            elif activation == 'relu':
                f.write(struct.pack('I', ACTIVATION_RELU))
            elif activation == 'softplus':
                f.write(struct.pack('I', ACTIVATION_SOFTPLUS))
            else:
                assert False, "Unsupported activation type: %s" % activation

        num_layers = len(model.layers)
        f.write(struct.pack('I', num_layers))

        for layer in model.layers:
            layer_type = type(layer).__name__

            if layer_type == 'Dense':
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                activation = layer.get_config()['activation']

                f.write(struct.pack('I', LAYER_DENSE))
                f.write(struct.pack('I', weights.shape[0]))
                f.write(struct.pack('I', weights.shape[1]))
                f.write(struct.pack('I', biases.shape[0]))

                weights = weights.flatten()
                biases = biases.flatten()

                write_floats(f, weights)
                write_floats(f, biases)

                write_activation(activation)

            elif layer_type == 'Convolution2D':
                assert layer.border_mode == 'valid', "Only border_mode=valid is implemented"

                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                activation = layer.get_config()['activation']

                # The kernel is accessed in reverse order. To simplify the C side we'll
                # flip the weight matrix for each kernel.
                weights = weights[:,:,::-1,::-1]

                f.write(struct.pack('I', LAYER_CONVOLUTION2D))
                f.write(struct.pack('I', weights.shape[0]))
                f.write(struct.pack('I', weights.shape[1]))
                f.write(struct.pack('I', weights.shape[2]))
                f.write(struct.pack('I', weights.shape[3]))
                f.write(struct.pack('I', biases.shape[0]))

                weights = weights.flatten()
                biases = biases.flatten()

                write_floats(f, weights)
                write_floats(f, biases)

                write_activation(activation)

            elif layer_type == 'Flatten':
                f.write(struct.pack('I', LAYER_FLATTEN))

            elif layer_type == 'ELU':
                f.write(struct.pack('I', LAYER_ELU))
                f.write(struct.pack('f', layer.alpha))

            elif layer_type == 'Activation':
                activation = layer.get_config()['activation']

                f.write(struct.pack('I', LAYER_ACTIVATION))
                write_activation(activation)

                

            else:
                assert False, "Unsupported layer type: %s" % layer_type
