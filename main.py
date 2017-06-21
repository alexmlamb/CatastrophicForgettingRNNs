
import theano
import theano.tensor as T
from nn_layers import param_init_gru, param_init_fflayer, fflayer, gru_layer
import lasagne
from utils import init_tparams
import numpy as np
from data_maker import anbn, cndn, keyvalue

num_iterations_switch_task = 10000

def init_params():
    params = {}

    params = param_init_gru({}, params, prefix='gru', nin=1, dim=512)

    #params = param_init_fflayer({},params,prefix='ff_0',nin=512+1,nout=512,batch_norm=False)
    params = param_init_fflayer({},params,prefix='ff_1',nin=512,nout=512,batch_norm=False)
    params = param_init_fflayer({},params,prefix='ff_2',nin=512,nout=512,batch_norm=False)
    params = param_init_fflayer({},params,prefix='ff_3',nin=512,nout=1,batch_norm=False)

    return init_tparams(params)


def create_network(p,inp,initial_state,num_steps=6):

    last_state = initial_state

    loss = 0.0

    for step in range(num_steps-1):

        gru_state = gru_layer(p,state_below=inp[:,step:step+1],options={},prefix='gru',mask=None,one_step=True,init_state=last_state)

        last_state = gru_state[0]

        h1 = fflayer(p,state_below=last_state,options={},prefix='ff_1',activ='lambda x: T.nnet.relu(x)')
        h2 = fflayer(p,state_below=h1,options={},prefix='ff_2',activ='lambda x: T.nnet.relu(x)')
        pred = fflayer(p,state_below=h2,options={},prefix='ff_3',activ='lambda x: x')
        loss += T.sum((pred - inp[:,step+1:step+2])**2)

    return loss, last_state

params = init_params()
inp = T.matrix()
attract_to_snapshot = T.scalar()
initial_state = T.matrix()
#last_state = T.matrix()

precisions = {}
snapshots = {}
save_to_snapshots = {}

for param in params.values():
    precisions[param] = theano.shared(np.zeros(shape=param.get_value().shape).astype('float32'))

for param in params.values():
    precision = precisions[param]
    snapshots[param] = theano.shared(np.zeros(shape=param.get_value().shape).astype('float32'))
    snapshots[precision] = theano.shared(np.zeros(shape=param.get_value().shape).astype('float32'))

for param in params.values():
    save_to_snapshots[snapshots[param]] = param
    precision = precisions[param]
    save_to_snapshots[snapshots[precision]] = precision

loss, last_state = create_network(params,inp,initial_state)
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
train_func = theano.function(inputs = [inp, attract_to_snapshot, initial_state], outputs = [loss, last_state], updates = updates)
eval_func = theano.function(inputs = [inp,initial_state], outputs = [loss])

if __name__ == "__main__":

    initial_state = np.zeros(shape=(4,512)).astype('float32')
    print "eval A", eval_func(anbn(),initial_state)
    print "eval B", eval_func(cndn(),initial_state)
    print "eval keyval", eval_func(keyvalue(),initial_state)

    for iteration in range(0,20000):
        loss, last_state = train_func(keyvalue(), 0.0, initial_state)
        initial_state = last_state
        #save_snapshot()

    states_snapshot = initial_state

    print "eval A", eval_func(anbn(),initial_state)
    print "eval B", eval_func(cndn(),initial_state)
    print "eval keyval", eval_func(keyvalue(),initial_state)

    print "change"

    save_snapshot()

    for iteration in range(0,10000):
        loss, last_space = train_func(anbn(), 1.0, initial_state)
        initial_state = last_state
        #save_snapshot()

    print "eval A", eval_func(anbn(),initial_state)
    print "eval B", eval_func(cndn(),initial_state)
    print "eval keyval", eval_func(keyvalue(), initial_state)
