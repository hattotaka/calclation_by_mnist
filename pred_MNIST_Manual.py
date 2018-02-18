import mnist_model as mm
from chainer import functions as F
from chainer import links as L
import numpy as np
from chainer import serializers
from PIL import Image, ImageFilter, ImageChops

def get_Centroid(mnist_img):
    imgArray = np.asarray(mnist_img)    
    x = 0
    y = 0
    x_cnt = 0
    y_cnt = 0
    for i in range(20):
        for j in range(20):
            x += (i) * imgArray[i][j]
            y += (j) * imgArray[i][j]        
            x_cnt += imgArray[i][j]
            y_cnt += imgArray[i][j]
    slide_x = int(round(x/x_cnt, 0))
    slide_y = int(round(y/y_cnt, 0))

    return 14 - slide_y, 14 - slide_x

def test(raw_img):
    # create model
    model = L.Classifier(mm.MLP(mm.p.n_units1, mm.p.n_units2, mm.p.n_out))
    serializers.load_npz(r"model/mnist_model.npz", model)
    
    bg = Image.new('RGB', (400, 400), (255, 255, 255))
    diff = ImageChops.difference(raw_img, bg)
    bbox = diff.getbbox()
    cropped_img = raw_img.crop(bbox)
    width, height = cropped_img.size
    s = max(width, height)
    resize_img = Image.new('RGB', (s, s), (255, 255, 255))
    if width >= height:
        resize_img.paste(cropped_img, (0, int((width - height)/2)))        
    else:
        resize_img.paste(cropped_img, (int((height - width)/2), 0))   

    #色をグレースケールへ変換
    gray_img = resize_img.convert("L")

    #色を真っ黒or真っ白へ変換
    black_img = gray_img.point(lambda x: 255 if x == 0 else 0)

    #モザイクをかけ、20px×20pxへリサイズ
    mnist_img = black_img.resize((400, 400)).filter(ImageFilter.GaussianBlur(3))
    mnist_img = mnist_img.point(lambda x: 0 if x == 0 else 255)
    mnist_img = mnist_img.filter(ImageFilter.GaussianBlur(5)).resize((20, 20))

    x, y = get_Centroid(mnist_img)
    #x, y = 4, 4
    canvas = Image.new('RGB', (28, 28), (0, 0, 0))
    canvas.paste(mnist_img, (x, y))
    mnist_img = canvas.convert("L")
    mnist_img.save(r'img\mnist_test_img.bmp')

    a = []
    width, height = mnist_img.size
    for j in range(width):
        for i in range(height):
            a.append(mnist_img.getpixel((i,j)) / 255)
    
    a = np.array(a).astype(np.float32)
    
    '''
    im2 = Image.new('RGBA',mnist_img.size)
    for i in range(28):
        for j in range(28):
            #ピクセルを取得
        #r,g,b = rgb_im.getpixel((x,y))
            #if i == 2:
                #print(a[28 * i + j])
            #平均化
        #g = (r + g + b)/3
        
            g = int(a[28 * i + j] * 255)
    #        g = int(x[28 * i + j] * 255)
            #if i < 6:
                #print(g)
            #print(g)
            #set pixel
            im2.putpixel((j,i),(g,g,g,0))
    
    #im2.show()
    im2.save(r'img\2_mnist2.bmp', 'bmp', quality=100, optimize=True)
    
    
    
    
    #a.append(mnist_img.getpixel((i,j)) for j in range(width) for i in range(height))
    print(a)
    #playground.print_mnist(x)
    '''
    #print(model.predictor(a.reshape(1, 784)))
    pred = F.softmax(model.predictor(a.reshape(1, 784))).data

    #print(pred)
    print("Prediction: ", np.argmax(pred))
    #print("Correct answer: ", y)
    return np.argmax(pred)
