#!/usr/bin/env python2
from __future__ import print_function
import numpy as np
import chainer # 1.16.0
from chainer import cuda, Variable, optimizers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import sys
from chainer import serializers
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


def train(model, optimizer, args):
    f = open(r"out/lstm_test.csv", 'a')
    writer = csv.writer(f, lineterminator='\n')
    csvlist = []
    csvlist.append('epoch')
    csvlist.append('train accuracy')
    csvlist.append('train loss')
    csvlist.append('test accuracy')
    csvlist.append('test loss')
    csvlist.append('data refresh')
    writer.writerow(csvlist)
    f.close()

    k = args.numbersize+2
    data_reload_flg = True
    data_reload_cnt = 0
    for epoch in range(args.epochsize):
        f = open(r"out/lstm_test.csv", 'a')
        writer = csv.writer(f, lineterminator='\n')
        csvlist = []
        print(data_reload_cnt)
#        if epoch % args.dataepoch == 0:
        if data_reload_flg == True:
            data_reload_cnt = data_reload_cnt + 1
            print('data_reload : ' + str(data_reload_cnt) + '回目')
            a, b, c = generate(args.datasize, args.numbersize)
            x_train = encode_in(a, b, 2*args.numbersize+2)
            t_train = encode_out(c, k)
            data_reload_flg = False
        print('epoch: %d' % (epoch + 1))
        csvlist.append(epoch + 1)

        sum_accu, sum_loss, sum_loss_train, sum_loss_test = 0, 0, 0, 0
        perm = np.random.permutation(args.datasize)
        for i in range(0, args.datasize, args.batchsize):
            indices = perm[i : i + args.batchsize]
            x = xp.array(x_train[indices])
            t = xp.array(t_train[indices])
 
            y = model.forward(x, k)
            accu, loss = 0, 0
            for i in range(k):
                ti = Variable(t[:, i])
                accu += F.accuracy(y[i], ti)
                loss += F.softmax_cross_entropy(y[i], ti)
            model.zerograds()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
 
            sum_accu += float(accu.data) * len(indices)
            sum_loss += float(loss.data) * len(indices)
 
        sum_loss_train = sum_loss / args.datasize
        print('train accuracy: %0.3f'   % (sum_accu / args.datasize))
        print('train loss: %0.3f' % (sum_loss_train))
        csvlist.append(sum_accu / args.datasize)
        csvlist.append(sum_loss_train)

        for i in range(3):
            j = indices[i]
            print('train example:\t%d\t+ %d\t= %d\t-> %s' % (a[j], b[j], c[j], ''.join([ output_alphabet[int(y_.data[i].argmax())] for y_ in y ])))
        if (epoch + 1) % 10 == 0:
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
            csvlist.append(sum_accu / testsize)
            csvlist.append(sum_loss_test)
            for i in range(3):
                print('test example:\t%d\t+ %d\t= %d\t-> %s' % (test_a[i], test_b[i], test_c[i], ''.join([ output_alphabet[int(y_.data[i].argmax())] for y_ in y ])))
            sys.stdout.flush()
            if sum_loss_test - sum_loss_train > 0.3:
                data_reload_flg = True
        else:
            csvlist.append('')
            csvlist.append('')

        csvlist.append(data_reload_cnt)
        writer.writerow(csvlist)
        f.close()

        if (epoch + 1) % 1000 == 0:
            serializers.save_npz(r"model/lstm_model_" + str(epoch + 1) + ".npz", model)

def main(args):
    model, optimizer = init(args)
    train(model, optimizer, args)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--unit', type=int, default=200)
    parser.add_argument('--datasize', type=int, default=2000)
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--epochsize', type=int, default=100000)
    parser.add_argument('--dataepoch', type=int, default=25)
    parser.add_argument('--numbersize', type=int, default=5)
    args = parser.parse_args()
    main(args)