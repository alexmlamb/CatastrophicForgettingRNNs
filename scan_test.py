import theano
import theano.tensor as T
import numpy as np
from utils import srng

A = T.ivector("A")

#ns = 10

all_memory = theano.shared(np.zeros(shape=(100,1),dtype=theano.config.floatX))
all_keyholes = theano.shared(np.zeros(shape=(100,1),dtype=theano.config.floatX))
#shared_tensor = srng.normal(size=(25,128))

def onestep(a,memory,keyholes):
    
    new_memory = T.set_subtensor(memory[a],T.cast(srng.normal(size=memory[a].shape),'float32'))
    new_keyholes = T.set_subtensor(keyholes[a],T.cast(srng.normal(size=keyholes[a].shape),'float32'))
    
    

    return a + 1, new_memory, new_keyholes

# Symbolic description of the result
result, updates = theano.scan(fn=onestep,
                              outputs_info=[T.zeros_like(A),all_memory,all_keyholes],
                              non_sequences=[],
                              n_steps=90)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[0]
shared = result[1][-1]

# compiled function that returns A**k
import time
t0 = time.time()
power = theano.function(inputs=[A], outputs=[final_result,shared], updates=updates)
print "time 2 compile", time.time() - t0

print "running"
r = power(range(100))

print r[1]
print r[1].shape




