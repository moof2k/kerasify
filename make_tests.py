import numpy as np
import pprint

from keras.models import Model
from keras.models import Sequential
from keras.layers import merge, Input
from keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D
from keras.layers.advanced_activations import ELU

from kerasify import export_model

np.set_printoptions(precision=25, threshold=np.nan)

TEST_CASE = """
bool test_%s(double* load_time, double* apply_time)
{
    printf("TEST %s\\n");

    KASSERT(load_time, "Invalid double");
    KASSERT(apply_time, "Invalid double");

    const std::vector<std::string> input_layer_names = {%s};
    const std::vector<std::string> output_layer_names = {%s};

    std::vector<Tensor> in_tensors = {
        %s
    };

    std::vector<Tensor> expected = {
        %s
    };

    KerasTimer load_timer;
    load_timer.Start();

    KerasModel model;
    KASSERT(model.LoadModel("test_%s.model"), "Failed to load model");

    *load_time = load_timer.Stop();

    KerasTimer apply_timer;
    apply_timer.Start();

    std::vector<Tensor> predicted = expected;

    // Build input tensor map.
    TensorMap in;
    for (unsigned int i = 0; i < in_tensors.size(); i++)
    {
        const std::string& input_layer_name = input_layer_names[i];
        in[input_layer_name] = &(in_tensors[i]);
    }

    // Build output tensor map.
    TensorMap out;
    for (unsigned int i = 0; i < predicted.size(); i++)
    {
        const std::string& output_layer_name = output_layer_names[i];
        out[output_layer_name] = &(predicted[i]);
    }
    KASSERT(model.Apply(in, &out), "Failed to apply");

    *apply_time = apply_timer.Stop();

    for (unsigned int i = 0; i < expected.size(); i++)
    {
        Tensor& expect = expected[i];
        Tensor& predict = predicted[i];
        for (int j = 0; j < expect.dims_[0]; j++)
        {
            KASSERT_EQ(expect(j), predict(j), %s);
        }
    }

    return true;
}
"""


def c_array_init(a):
  s = pprint.pformat(a.flatten())
  s = s.replace('[', '{').replace(']', '}').replace('array(', '').replace(
      ')', '').replace(', dtype=float32', '')

  shape = ''

  if a.shape == () or a.shape == (1,):
    s = '{%s}' % s
    shape = '{{1}}'
  elif a.shape:
    shape = repr(a.shape).replace(',)', ')')

  shape = shape.replace('(', '{').replace(')', '}')
  return shape, s


def tensor_map_init(tensor_list, join_str):
  y = ['{%s, %s}' % c_array_init(tensor) for tensor in tensor_list]
  y = join_str.join(y)
  return y


def output_testcase(model, test_x_list, test_y_list, name, eps):
  print 'Processing %s' % name
  assert isinstance(test_x_list, list), 'test_x_list must be a list.'
  assert isinstance(test_y_list, list), 'test_y_list must be a list.'

  model.compile(loss='mean_squared_error', optimizer='adamax')
  model.fit(test_x_list, test_y_list, nb_epoch=1, verbose=False)
  predict_y_list = model.predict(test_x_list)
  if not isinstance(predict_y_list, list):
    predict_y_list = [predict_y_list]

  print model.summary()

  export_model(model, 'test_%s.model' % name)

  with open('test_%s.h' % name, 'w') as f:
    predict_x_list = [test_x[0] for test_x in test_x_list]
    x_map = tensor_map_init(predict_x_list, ',\n        ')
    y_map = tensor_map_init(predict_y_list, ',\n        ')
    input_layer_names = ', '.join(
        ["\"%s\"" % layer_name for layer_name in model.input_names])
    output_layer_names = ', '.join(
        ["\"%s\"" % layer_name for layer_name in model.output_names])
    f.write(TEST_CASE % (name, name, input_layer_names, output_layer_names,
                         x_map, y_map, name, eps))


""" Dense 1x1 """
test_x = np.arange(10)
test_y = test_x * 10 + 1
model = Sequential()
model.add(Dense(1, input_dim=1))

output_testcase(model, [test_x], [test_y], 'dense_1x1', '1e-6')
""" Dense 10x1 """
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential()
model.add(Dense(1, input_dim=10))

output_testcase(model, [test_x], [test_y], 'dense_10x1', '1e-6')
""" Dense 2x2 """
test_x = np.random.rand(10, 2).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'dense_2x2', '1e-6')
""" Dense 10x10 """
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'dense_10x10', '1e-6')
""" Dense 10x10x10 """
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10, 10).astype('f')
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(10))

output_testcase(model, [test_x], [test_y], 'dense_10x10x10', '1e-6')
""" Conv 2x2 """
test_x = np.random.rand(10, 1, 2, 2).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(Convolution2D(1, 2, 2, input_shape=(1, 2, 2)))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'conv_2x2', '1e-6')
""" Conv 3x3 """
test_x = np.random.rand(10, 1, 3, 3).astype('f').astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(Convolution2D(1, 3, 3, input_shape=(1, 3, 3)))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'conv_3x3', '1e-6')
""" Conv 3x3x3 """
test_x = np.random.rand(10, 3, 10, 10).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(Convolution2D(3, 3, 3, input_shape=(3, 10, 10)))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'conv_3x3x3', '1e-6')
""" Activation ELU """
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 1).astype('f')
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(ELU(alpha=0.5))
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'elu_10', '1e-6')
""" Activation relu """
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Activation('relu'))

output_testcase(model, [test_x], [test_y], 'relu_10', '1e-6')
""" Dense relu """
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))

output_testcase(model, [test_x], [test_y], 'dense_relu_10', '1e-6')
""" Conv softplus """
test_x = np.random.rand(10, 1, 2, 2).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(Convolution2D(1, 2, 2, input_shape=(1, 2, 2), activation='softplus'))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'conv_softplus_2x2', '1e-6')
""" Maxpooling2D 1x1"""
test_x = np.random.rand(10, 1, 10, 10).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(MaxPooling2D(pool_size=(1, 1), input_shape=(1, 10, 10)))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'maxpool2d_1x1', '1e-6')
""" Maxpooling2D 2x2"""
test_x = np.random.rand(10, 1, 10, 10).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(1, 10, 10)))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'maxpool2d_2x2', '1e-6')
""" Maxpooling2D 3x2x2"""
test_x = np.random.rand(10, 3, 10, 10).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(3, 10, 10)))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'maxpool2d_3x2x2', '1e-6')
""" Maxpooling2D 3x3x3"""
test_x = np.random.rand(10, 3, 10, 10).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(MaxPooling2D(pool_size=(3, 3), input_shape=(3, 10, 10)))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, [test_x], [test_y], 'maxpool2d_3x3x3', '1e-6')
""" Benchmark """
test_x = np.random.rand(1, 3, 128, 128).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential()
model.add(Convolution2D(16, 7, 7, input_shape=(3, 128, 128), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(ELU())
model.add(Convolution2D(8, 3, 3))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10))

output_testcase(model, [test_x], [test_y], 'benchmark', '1e-3')

###
# Functional Model Support Tests.
###
""" Functional Dense 1x1 """
test_x = np.arange(10)
test_y = test_x * 10 + 1

input = Input(name='in1', shape=(1,))
output = Dense(1, name='out1')(input)
model = Model(input=input, output=output)

output_testcase(model, [test_x], [test_y], 'func_dense_1x1', '1e-6')
""" Functional Conv 2x2 """
test_x = np.random.rand(10, 1, 2, 2).astype('f')
test_y = np.random.rand(10, 1).astype('f')

input0 = Input(name='in0', shape=(1, 2, 2))
conv0 = Convolution2D(1, 2, 2)(input0)
f0 = Flatten()(conv0)
output0 = Dense(1)(f0)

model = Model(input=input0, output=output0)

output_testcase(model, [test_x], [test_y], 'func_conv_2x2', '1e-6')
""" Functional Maxpooling2D 3x3x3"""
test_x = np.random.rand(10, 3, 10, 10).astype('f')
test_y = np.random.rand(10, 1).astype('f')

input0 = Input(name='in0', shape=(3, 10, 10))
conv0 = MaxPooling2D(pool_size=(3, 3))(input0)
f0 = Flatten()(conv0)
output0 = Dense(1)(f0)

model = Model(input=input0, output=output0)

output_testcase(model, [test_x], [test_y], 'func_maxpool2d_3x3x3', '1e-6')
""" Functional Merge 1x1"""
test_x = np.arange(10)
test_y = test_x * 10 + 1

input = Input(name='in1', shape=(1,))
h1 = Dense(1, name='hidden1')(input)
h2 = Dense(1, name='hidden2')(h1)
hA = Dense(1, name='hiddenA')(input)
m = merge([h2, hA], mode='concat', concat_axis=-1)
output = Dense(1, name='out1')(m)
model = Model(input=input, output=output)

output_testcase(model, [test_x], [test_y], 'func_merge_1x1', '1e-6')
""" Functional Dense 1x1 Multi In"""
test_x_list = [np.random.rand(10).astype('f'), np.random.rand(10).astype('f')]
test_y_list = [test_x_list[0] * 10 + 1]

input0 = Input(name='in0', shape=(1,))
input1 = Input(name='in1', shape=(1,))
h1 = Dense(1, name='hidden1')(input0)
h2 = Dense(1, name='hidden2')(h1)

hA = Dense(1, name='hiddenA')(input1)
m = merge([h2, hA], mode='concat', concat_axis=-1)
output0 = Dense(1, name='out0')(m)
model = Model(input=[input0, input1], output=[output0])

output_testcase(model, test_x_list, test_y_list,
                'func_dense_1x1_multi_in', '1e-6')
""" Functional Dense 1x1 Multi Out"""
test_x_list = [np.arange(10)]
test_y_list = [test_x_list[0] * 10 + 1, test_x_list[0] * 5 + 2]

input0 = Input(name='in0', shape=(1,))
h1 = Dense(1, name='hidden1')(input0)
output0 = Dense(1, name='out0')(h1)
output1 = Dense(1, name='out1')(h1)
model = Model(input=[input0], output=[output0, output1])

output_testcase(model, test_x_list, test_y_list,
                'func_dense_1x1_multi_out', '1e-6')
""" Functional Dense 1x1 Multi In Out"""
test_x_list = [np.arange(10), np.arange(10)]
test_y_list = [test_x_list[0] * 10 + 1, test_x_list[1] * 5 + 2]

input0 = Input(name='in0', shape=(1,))
input1 = Input(name='in1', shape=(1,))
h1 = Dense(1, name='hidden1')(input0)
h2 = Dense(1, name='hidden2')(h1)

hA = Dense(1, name='hiddenA')(input1)
m = merge([h2, hA], mode='concat', concat_axis=-1)
output0 = Dense(1, name='out0')(m)
output1 = Dense(1, name='out1')(m)
model = Model(input=[input0, input1], output=[output0, output1])

output_testcase(model, test_x_list, test_y_list,
                'func_dense_1x1_multi_in_out', '1e-6')
""" Functional Conv 2x2 Multi In Out"""
test_x_list = [
    np.random.rand(10, 1, 2, 2).astype('f'),
    np.random.rand(10, 1, 2, 2).astype('f')
]
test_y_list = [
    np.random.rand(10, 1).astype('f'), np.random.rand(10, 1).astype('f')
]

input0 = Input(name='in0', shape=(1, 2, 2))
conv0 = Convolution2D(1, 2, 2)(input0)
f0 = Flatten()(conv0)

input1 = Input(name='in1', shape=(1, 2, 2))
conv1 = Convolution2D(1, 2, 2)(input1)
f1 = Flatten()(conv1)

m = merge([f0, f1], mode='concat', concat_axis=-1)

output0 = Dense(1, name='out0')(m)
output1 = Dense(1, name='out1')(m)

model = Model(input=[input0, input1], output=[output0, output1])

output_testcase(model, test_x_list, test_y_list,
                'func_conv_2x2_multi_in_out', '1e-6')
