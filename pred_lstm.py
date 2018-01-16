from chainer import  Variable, optimizers
import chainer # 1.16.0
from chainer import serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np
import csv

input_alphabet = '0123456789+= '
output_alphabet = '0123456789 '

def generate(n_data, k):
    a = np.exp(np.random.uniform(np.log(1), np.log(10**k), n_data)).astype(np.int32)
    b = np.exp(np.random.uniform(np.log(1), np.log(10**k), n_data)).astype(np.int32)
    c = a + b
    return a, b, c
def encode_in(a, b, k):
    alphabet = np.array(list(input_alphabet))
    texts = np.array([ '{}+{}='.format(a_, b_).rjust(k, ' ') for a_, b_ in zip(a, b) ])
    return np.array([[alphabet == c for c in s] for s in texts]).astype(np.float32)
def encode_out(c, k):
    texts = np.array([ '{}'.format(c_).ljust(k, ' ') for c_ in c ])
    return np.array([[output_alphabet.index(c) for c in s] for s in texts]).astype(np.int32)

class Model(Chain):
    def __init__(self, unit):
        super(Model, self).__init__(
            l1=L.Linear(len(input_alphabet), unit),
            l2=L.LSTM(unit, unit),
            l3=L.Linear(unit, len(output_alphabet)),
        )
    def forward(self, x, k):
        self.l2.reset_state()
        for i in range(x.shape[1]):
            h = F.relu(self.l1( Variable(x[:, i, :]) ))
            h = self.l2(h)
        result = []
        for i in range(k):
            h = F.relu(h)
            h = self.l2(h)
            result += [ self.l3(h) ]
        return result
 

def init(args):
    global xp
    model = Model(args.unit)
    #if args.gpu is not None:
    #    cuda.get_device(args.gpu).use()
    #    model.to_gpu()
    #    xp = cuda.cupy
    #else:
    #    xp = np
    xp = np
    # optimizer = optimizers.Adam()
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    return model, optimizer



def test(model, optimizer, args):
    k = args.numbersize+2
    testsize = 2000
    test_a, test_b, test_c = generate(testsize, args.numbersize)
    x = xp.array(encode_in(test_a, test_b, 2*args.numbersize+2))
    t = xp.array(encode_out(test_c, k))
    y = model.forward(x, k)
    accu, loss = 0, 0
    for i in range(k):
        ti = Variable(t[:, i])
        accu += F.accuracy(y[i], ti)
        loss += F.softmax_cross_entropy(y[i], ti)
    sum_accu = float(accu.data) * testsize
    sum_loss = float(loss.data) * testsize
    sum_loss_test = sum_loss / testsize
    print('test accuracy: %0.3f'   % (sum_accu / testsize))
    print('test loss: %0.3f' % (sum_loss_test))
    for i in range(3):
        print('test example:\t%d\t+ %d\t= %d\t-> %s' % (test_a[i], test_b[i], test_c[i], ''.join([ output_alphabet[int(y_.data[i].argmax())] for y_ in y ])))
    #sys.stdout.flush()


def main(args):
    model, optimizer = init(args)
    serializers.load_npz(r"model/lstm_model.npz", model)
    test(model, optimizer, args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--unit', type=int, default=200)
    parser.add_argument('--datasize', type=int, default=2000)
    parser.add_argument('--batchsize', type=int, default=1000)
    parser.add_argument('--epochsize', type=int, default=10)
    parser.add_argument('--dataepoch', type=int, default=25  )
    parser.add_argument('--numbersize', type=int, default=5)
    args = parser.parse_args()
    main(args)