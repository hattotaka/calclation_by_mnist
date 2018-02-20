from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import pandas as pd
import model as m
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain, serializers

IMG_SIZE = 500
OUT_IMG_SIZE = 50
FONT_WIDTH = int(round(IMG_SIZE / 20, 0))
GAUSSIAN = int(round(IMG_SIZE / 150, 0))

def data_normalization(data):
    tmpmax = 0
    for i in range(len(data)):
        if tmpmax <= data[i]:
            tmpmax = data[i]
    
    tmpmin = tmpmax
    for i in range(len(data)):
        if tmpmin >= data[i]:
            tmpmin = data[i]
    
    data_conv = []
    for i in range(len(data)):
        data_conv.append((data[i] - tmpmin)/(tmpmax - tmpmin) * IMG_SIZE)

    return data_conv

def make_graph_picture(data):
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    w = IMG_SIZE / (len(data) - 1)
    for i in range(len(data) - 1) :
        draw.line((i * w, IMG_SIZE - data[i], (i + 1) * w, IMG_SIZE - data[i + 1]), fill=(255, 255, 255), width=FONT_WIDTH)
    graph_img = img.filter(ImageFilter.GaussianBlur(GAUSSIAN)).resize((OUT_IMG_SIZE, OUT_IMG_SIZE)).convert('L')
    return graph_img


def pic_to_array(graph_img):
    pic_array = []
    width, height = graph_img.size
    for j in range(width):
        for i in range(height):
            pic_array.append(graph_img.getpixel((i,j)) / 255)
    
    #pic_array = np.array(pic_array).reshape((-1, 1, OUT_IMG_SIZE, OUT_IMG_SIZE)).astype(np.float32)
    return pic_array

def main(args):
    df = pd.read_csv('data.csv')
    print('csvファイル読込完了')
    df = df[0:100]
    data = [list(df.drop('label', axis=1).iloc[i].values.flatten()) for i in range(len(df))]
    print('データ変換完了')
    label = np.array([int(df['label'].iloc[i]) for i in range(len(df))])
    print('ラベル変換完了')
    pic_array = np.array([np.array(pic_to_array(make_graph_picture(data_normalization(data[i])))).reshape((-1, 1, OUT_IMG_SIZE, OUT_IMG_SIZE)).astype(np.float32) for i in range(len(data))])
    '''
    pic_arrays = []
    for i in range(len(data)):
        tmpdata = data_normalization(data[i])
        graph_img = make_graph_picture(tmpdata)
        pic_arrays.append(pic_to_array(graph_img))
        if (i + 1) % 100 == 0:
            print('グラフ配列変換：' + str(i + 1) +'件/' + str(len(data) + 1) + '件')
    print('グラフ配列変換：' + str(len(data) + 1) +'件/' + str(len(data) + 1) + '件')
    '''
    print('グラフ配列変換完了')

    x_train = pic_array[0:50]
    t_train = label[0:50]
    x_test = pic_array[50:len(pic_array)]
    t_test = label[50:len(label)]

    model = m.CNN()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    print('学習開始')
    datasize = 30
    #batchsize = 100
    for epoch in range(1,args.epochsize):
        print("epoch: %d" % epoch)
    
        #train
        sum_accu, sum_loss, sum_loss_train, sum_loss_test = 0, 0, 0, 0
        perm = np.random.permutation(datasize)
        for i in range(0, datasize, args.batchsize):
            perm = np.random.permutation(datasize)
            for i in range(0, datasize, args.batchsize):
                indices = perm[i : i + args.batchsize]
                x = np.array(x_train[indices])
                t = np.array(t_train[indices])
                print(111)
                y = model.forward(x)
                print(112)
                accu, loss = 0, 0
                
                for i in range(len(y)):
                    ti = Variable(t[:, i])
                    accu += F.accuracy(y[i], ti)
                    loss += F.softmax_cross_entropy(y[i], ti)
                model.zerograds()
                loss.backward()
                #loss.unchain_backward()
                optimizer.update()
     
                sum_accu += float(accu.data) * len(indices)
                sum_loss += float(loss.data) * len(indices)
     
        sum_accuracy_train = sum_accu / datasize
        sum_loss_train = sum_loss / datasize
        print('train accuracy: %0.3f'   % (sum_accuracy_train))
        print('train loss: %0.3f' % (sum_loss_train))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--datasize', type=int, default=0.7)
    #parser.add_argument('--testdatasize', type=int, default=0.3)
    parser.add_argument('--batchsize', type=int, default=30)
    parser.add_argument('--testbatchsize', type=int, default=100)
    parser.add_argument('--epochsize', type=int, default=2)
    args = parser.parse_args()
    main(args)