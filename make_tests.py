import numpy as np
import pprint

from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Activation
from keras.layers.advanced_activations import ELU

from kerasify import export_model

np.set_printoptions(precision=25, threshold=np.nan)

def c_array(a):
    s = pprint.pformat(a.flatten())
    s = s.replace('[', '{').replace(']', '}').replace('array(', '').replace(')', '').replace(', dtype=float32', '')

    shape = ''

    if a.shape == ():
        s = '{%s}' % s
        shape = '(1)'
    else:
        shape = repr(a.shape).replace(',)', ')')

    return shape, s


TEST_CASE = '''
bool test_%s(double* load_time, double* apply_time)
{
    printf("TEST %s\\n");

    KASSERT(load_time, "Invalid double");
    KASSERT(apply_time, "Invalid double");

    Tensor in%s;
    in.data_ = %s;

    Tensor out%s;
    out.data_ = %s;

    KerasTimer load_timer;
    load_timer.Start();

    KerasModel model;
    KASSERT(model.LoadModel("%s_test.model"), "Failed to load model");

    *load_time = load_timer.Stop();

    KerasTimer apply_timer;
    apply_timer.Start();

    Tensor predict = out;
    KASSERT(model.Apply(&in, &out), "Failed to apply");

    *apply_time = apply_timer.Stop();

    for (int i = 0; i < out.dims_[0]; i++)
    {
        KASSERT_EQ(out(i), predict(i), %s);
    }

    return true;
}
'''

def output_testcase(model, test_x, test_y, name, eps):
    print "Processing %s" % name
    model.compile(loss='mean_squared_error', optimizer='adamax')
    model.fit(test_x, test_y, nb_epoch=1, verbose=False)
    predict_y = model.predict(test_x).astype('f')
    print model.summary()

    export_model(model, '%s_test.model' % name)

    with open('%s_test.h' % name, 'w') as f:
        x_shape, x_data = c_array(test_x[0])
        y_shape, y_data = c_array(predict_y[0])

        f.write(TEST_CASE % (name, name, x_shape, x_data, y_shape, y_data, name, eps))

    


''' Dense 1x1 '''
test_x = np.arange(10)
test_y = test_x * 10 + 1
model = Sequential()
model.add(Dense(1, input_dim=1))

output_testcase(model, test_x, test_y, 'dense_1x1', '1e-6')

''' Dense 10x1 '''
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential()
model.add(Dense(1, input_dim=10))

output_testcase(model, test_x, test_y, 'dense_10x1', '1e-6')

''' Dense 2x2 '''
test_x = np.random.rand(10, 2).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Dense(1))

output_testcase(model, test_x, test_y, 'dense_2x2', '1e-6')

''' Dense 10x10 '''
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(1))

output_testcase(model, test_x, test_y, 'dense_10x10', '1e-6')

''' Dense 10x10x10 '''
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10, 10).astype('f')
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(10))

output_testcase(model, test_x, test_y, 'dense_10x10x10', '1e-6')

''' Conv 2x2 '''
test_x = np.random.rand(10, 1, 2, 2).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(Convolution2D(1, 2, 2, input_shape=(1, 2, 2)))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, test_x, test_y, 'conv_2x2', '1e-6')

''' Conv 3x3 '''
test_x = np.random.rand(10, 1, 3, 3).astype('f').astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(Convolution2D(1, 3, 3, input_shape=(1, 3, 3)))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, test_x, test_y, 'conv_3x3', '1e-6')

''' Conv 3x3x3 '''
test_x = np.random.rand(10, 3, 10, 10).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(Convolution2D(3, 3, 3, input_shape=(3, 10, 10)))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, test_x, test_y, 'conv_3x3x3', '1e-6')

''' Activation ELU '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 1).astype('f')
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(ELU(alpha=0.5))
model.add(Dense(1))

output_testcase(model, test_x, test_y, 'elu_10', '1e-6')

''' Activation relu '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Activation('relu'))

output_testcase(model, test_x, test_y, 'relu_10', '1e-6')

''' Dense relu '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))

output_testcase(model, test_x, test_y, 'dense_relu_10', '1e-6')

''' Conv softplus '''
test_x = np.random.rand(10, 1, 2, 2).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential()
model.add(Convolution2D(1, 2, 2, input_shape=(1, 2, 2), activation='softplus'))
model.add(Flatten())
model.add(Dense(1))

output_testcase(model, test_x, test_y, 'conv_softplus_2x2', '1e-6')

''' Benchmark '''
test_x = np.random.rand(1, 3, 64, 64).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential()
model.add(Convolution2D(16, 7, 7, input_shape=(3, 64, 64), activation='relu'))
model.add(ELU())
model.add(Convolution2D(8, 3, 3))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10))

output_testcase(model, test_x, test_y, 'benchmark', '1e-3')


