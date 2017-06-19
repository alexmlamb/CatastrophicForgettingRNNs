
import theano
import theano.tensor as T
from nn_layers import param_init_gru, param_init_fflayer, fflayer, gru_layer
import lasagne
from utils import init_tparams
import numpy as np

num_iterations_switch_task = 10000

def init_params():
    params = {}

    params = param_init_gru({}, params, prefix='gru', nin=1, dim=512)

    params = param_init_fflayer({},params,prefix='ff_1',nin=512,nout=512,batch_norm=False)
    params = param_init_fflayer({},params,prefix='ff_2',nin=512,nout=1,batch_norm=False)

    return init_tparams(params)


def create_network(p,inp,num_steps=6):

    loss = 0.0

    last_state = theano.shared(np.zeros((1,512)).astype('float32'))

    for step in range(num_steps-1):

        gru_state = gru_layer(p,state_below=inp[:,step],options={},prefix='gru',mask=None,one_step=True,init_state=last_state)

        last_state = gru_state[0]

        h1 = fflayer(p,state_below=gru_state[0],options={},prefix='ff_1',activ='lambda x: T.nnet.relu(x)')

        pred = fflayer(p,state_below=h1,options={},prefix='ff_2',activ='lambda x: x')
        loss += T.sum((pred - inp[:,step+1])**2)

    return loss

params = init_params()
inp = T.matrix()
attract_to_snapshot = T.scalar()

precisions = {}
snapshots = {}
save_to_snapshots = {}

for param in params.values():
    precisions[param] = theano.shared(np.zeros(shape=param.get_value().shape).astype('float32'))

for param in params.values():
    snapshots[param] = theano.shared(np.zeros(shape=param.get_value().shape).astype('float32'))

for param in params.values():
    save_to_snapshots[snapshots[param]] = param

loss = create_network(params,inp)
loss_ewc = loss

for param in params.values():
    loss_ewc += attract_to_snapshot * T.sum(precisions[param] * T.sqr(param - snapshots[param]))

print "params", params
updates = lasagne.updates.adam(loss_ewc, params.values())

#precision is 1/sigma^2
for param in precisions:
    precision = precisions[param]
    updates[precision] = precision + T.sqr(T.grad(loss,param))

save_snapshot = theano.function(inputs = [], outputs = [], updates = save_to_snapshots)
train_func = theano.function(inputs = [inp,attract_to_snapshot], outputs = [loss], updates = updates)
eval_func = theano.function(inputs = [inp], outputs = [loss])

if __name__ == "__main__":

    for iteration in range(0,2000):
        train_func(np.asarray([[0,1,0,1,0,1]]).astype('float32'),0.0)

    print "eval A", eval_func(np.asarray([[0,1,0,1,0,1]]).astype('float32'))
    print "eval B", eval_func(np.asarray([[2,3,2,3,2,3]]).astype('float32'))

    print "change"

    save_snapshot()

    for iteration in range(0,2000):
        train_func(np.asarray([[2,3,2,3,2,3]]).astype('float32'),10.0)

    print "eval A", eval_func(np.asarray([[0,1,0,1,0,1]]).astype('float32'))
    print "eval B", eval_func(np.asarray([[2,3,2,3,2,3]]).astype('float32'))



