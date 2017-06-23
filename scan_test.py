import theano
import theano.tensor as T
import numpy as np

k = T.iscalar("k")
A = T.ivector("A")

#ns = 10

shared_tensor = T.zeros(shape=(10,))

def onestep(a,shared_boy):

    shared_boy = T.set_subtensor(shared_boy[a],a)
    return a + 1, shared_boy
    

# Symbolic description of the result
result, updates = theano.scan(fn=onestep,
                              outputs_info=[T.ones_like(A),shared_tensor],
                              non_sequences=[],
                              n_steps=k)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[0]
shared = result[1]

# compiled function that returns A**k
import time
t0 = time.time()
power = theano.function(inputs=[A,k], outputs=[final_result,shared], updates=updates)
print "time 2 compile", time.time() - t0

print "running"
print(power(range(10),2))[1].shape
print(power(range(10),4))[1].shape



