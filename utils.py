import theano
from theano import tensor
import theano.tensor as T
import warnings
import six
import pickle

import numpy
import inspect
from collections import OrderedDict
import time
import os
from sklearn.cross_validation import KFold
import numpy as np
import numpy.random as rng

# make prefix-appended name
def _p(pp, name):
        return '%s_%s' % (pp, name)

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in six.iteritems(params):
        tparams[kk].set_value(vv)



# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in six.iteritems(zipped):
        new_params[kk] = vv.get_value()
    return new_params

def create_log_dir(suffix, model_id):
    model_id += suffix + time.strftime('-%y%m%dT%H%M%S')
    model_dir = model_id
    os.makedirs(model_dir)
    return model_dir


def merge_images(img1,img2,cutoff=16):
    i1 = img1.reshape((64,3,32,32))
    i2 = img2.reshape((64,3,32,32))

    new_img = np.concatenate([i1[:,:,:,0:cutoff], i2[:,:,:,cutoff:32]], axis = 3)

    return new_img.reshape((64,32*32*3))

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in six.iteritems(tparams)]

srng = theano.tensor.shared_randomstreams.RandomStreams(42)

def dropout(inp,p=0.5):
    return inp * tensor.cast(srng.binomial(n=1,p=p,size=inp.shape),'float32')

def logit(inp):
    return tensor.log(inp) - tensor.log(1.0 - inp)

def inverse_sigmoid(inp):
    inp = tensor.clip(inp, 0.000001, 0.999999)
    return logit(inp)

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         state_before *
                         trng.binomial(state_before.shape,
                                       p=0.5,
                                       n=1,
                                       dtype=state_before.dtype),
                         state_before * 0.5)
    return proj

def join2(a,b):
    return tensor.concatenate([a,b],axis=1)

def join3(a,b,c):
    return tensor.concatenate([a,b,c],axis=1)

def l2_norm(a):
    return T.sqrt(T.sum(T.sqr(a)))

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in six.iteritems(params):
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in six.iteritems(params):
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def uniform_weight(nin, nout, scale=None):
    if scale is None:
        scale = numpy.sqrt(6. / (nin + nout))

    W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k], )
    output_shape += (concat_size, )
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k], )

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None), )
        indices += (slice(offset, offset + tt.shape[axis]), )
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None), )

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


class Parameters():
    def __init__(self):
        # self.__dict__['tparams'] = dict()
        self.__dict__['tparams'] = OrderedDict()

    def __setattr__(self, name, array):
        tparams = self.__dict__['tparams']
        # if name not in tparams:
        tparams[name] = array

    def __setitem__(self, name, array):
        self.__setattr__(name, array)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __getattr__(self, name):
        tparams = self.__dict__['tparams']
        return tparams[name]

    # def __getattr__(self):
    # return self.get()

    def remove(self, name):
        del self.__dict__['tparams'][name]

    def get(self):
        return self.__dict__['tparams']

    def values(self):
        tparams = self.__dict__['tparams']
        return tparams.values()

    def save(self, filename):
        tparams = self.__dict__['tparams']
        pickle.dump({p: tparams[p] for p in tparams}, open(filename, 'wb'), 2)

    def load(self, filename):
        tparams = self.__dict__['tparams']
        loaded = pickle.load(open(filename, 'rb'))
        for k in loaded:
            tparams[k] = loaded[k]

    def setvalues(self, values):
        tparams = self.__dict__['tparams']
        for p, v in zip(tparams, values):
            tparams[p] = v

    def __enter__(self):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe(
        ).f_back)
        self.__dict__['_env_locals'] = env_locals.keys()

    def __exit__(self, type, value, traceback):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe(
        ).f_back)
        prev_env_locals = self.__dict__['_env_locals']
        del self.__dict__['_env_locals']
        for k in env_locals.keys():
            if k not in prev_env_locals:
                self.__setattr__(k, env_locals[k])
                env_locals[k] = self.__getattr__(k)
        return True

      #here is the information to commit..

if __name__ == "__main__":

    import numpy.random as rng
    import numpy as np
    #x1 = rng.normal(size = (64,3,32,32))
    #x2 = np.zeros(shape = (64,3,32,32))

    #x3 = merge_images(x1,x2)

    #print x3.shape
    p = Parameters()
    #p.tparams = {'a' : theano.shared(np.zeros(shape=(13,)))}

    p.load("derp")
    
    print p.tparams['tparams']['a'].get_value()



