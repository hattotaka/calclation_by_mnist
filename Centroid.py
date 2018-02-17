from PIL import Image, ImageFilter
import numpy as np

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

#画像取得
raw_img = Image.open(r'img\test.bmp', 'r')

#色をグレースケールへ変換
gray_img = raw_img.convert("L")

#色を真っ黒or真っ白へ変換(背景は黒)
black_img = gray_img.point(lambda x: 255 if x < 255 else 255 - x)

#モザイクをかけ、28px×28pxへリサイズ
mnist_img = black_img.filter(ImageFilter.GaussianBlur(10)).resize((20, 20))
#mnist_img.save(r'img\test2.bmp')

canvas = Image.new('RGB', (28, 28), (0, 0, 0))

x, y = get_Centroid(mnist_img)
canvas.paste(mnist_img, (x, y))
#canvas.save(r'img\test3.bmp')