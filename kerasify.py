import numpy as np
import struct

LAYER_DENSE = 1
LAYER_CONVOLUTION2D = 2
LAYER_FLATTEN = 3
LAYER_ELU = 4
LAYER_ACTIVATION = 5
LAYER_MAXPOOLING2D = 6
LAYER_INPUT = 7
LAYER_MERGE = 8

ACTIVATION_LINEAR = 1
ACTIVATION_RELU = 2
ACTIVATION_SOFTPLUS = 3


def write_floats(file, floats):
  """
    Writes floats to file in 1024 chunks. prevents memory explosion
    writing very large arrays to disk when calling struct.pack().
    """
  step = 1024
  written = 0

  for i in np.arange(0, len(floats), step):
    remaining = min(len(floats) - i, step)
    written += remaining
    file.write(struct.pack('=%sf' % remaining, *floats[i:i + remaining]))

  assert written == len(floats)


def write_strings(file, strings):
  """
    Writes strings to file keeping to 4-byte aligned.
    """
  file.write(struct.pack('I', len(strings)))
  for string in strings:
    write_string(file, string)


def write_string(file, string):
  """
    Writes strings to file keeping to 4-byte aligned.
    """
  length = len(string)

  # Round length up to the nearest multiple of 4.
  size = (length + 3 % 4)

  file.write(struct.pack('I', size))
  file.write(struct.pack('%ds' % size, string))


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
        assert False, 'Unsupported activation type: %s' % activation

    # Sequential models hide the Input layer within the first layer's
    # inbound nodes and these do not appear in the layer list.
    layers = []
    layer_map = {}
    for layer in model.layers:
      for node in layer.inbound_nodes:
        for inbound_layer in node.inbound_layers:
          #TODO(hemalshah): Handle dependent layers recursively.
          if inbound_layer.name not in layer_map:
            layer_map[inbound_layer.name] = inbound_layer
            layers.append(inbound_layer)
      layer_map[layer.name] = layer
      layers.append(layer)

    num_layers = len(layers)
    f.write(struct.pack('I', num_layers))

    write_strings(f, model.input_names)
    write_strings(f, model.output_names)

    for layer in layers:
      layer_type = type(layer).__name__

      name = layer.name
      write_string(f, name)

      inbound_layer_names = []
      for node in layer.inbound_nodes:
        for inbound_layer in node.inbound_layers:
          inbound_layer_names.append(inbound_layer.name)
      write_strings(f, inbound_layer_names)

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

      elif layer_type == 'InputLayer':
        f.write(struct.pack('I', LAYER_INPUT))

      elif layer_type == 'Merge':
        assert layer.concat_axis == -1, ('Only concatenation along batch '
                                         'dimensions implemented')
        assert layer.mode == 'concat', 'Only concatenation implemented'
        f.write(struct.pack('I', LAYER_MERGE))

      elif layer_type == 'Convolution2D':
        assert layer.border_mode == 'valid', ('Only border_mode=valid is '
                                              'implemented')

        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        activation = layer.get_config()['activation']

        # The kernel is accessed in reverse order. To simplify the C side we'll
        # flip the weight matrix for each kernel.
        weights = weights[:, :, ::-1, ::-1]

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

      elif layer_type == 'MaxPooling2D':
        assert layer.border_mode == 'valid', ('Only border_mode=valid is '
                                              'implemented')

        pool_size = layer.get_config()['pool_size']

        f.write(struct.pack('I', LAYER_MAXPOOLING2D))
        f.write(struct.pack('I', pool_size[0]))
        f.write(struct.pack('I', pool_size[1]))

      else:
        assert False, 'Unsupported layer type: %s' % layer_type
