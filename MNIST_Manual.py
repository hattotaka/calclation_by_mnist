import tkinter
from PIL import ImageGrab
import pred_MNIST_Manual as p

class Scribble:
    #文字描画開始時の初期処理
    def on_pressed(self, event):
        self.sx = event.x
        self.sy = event.y
        self.text_out_formula.set('')

    #文字の描画処理
    def on_dragged(self, event):
        self.canvas.create_line(self.sx, self.sy, event.x, event.y, fill = 'black', width = '10', tag="line")
        self.sx = event.x
        self.sy = event.y

    def on_released(self, event):
        img = self.get_pic()
        pred = p.test(img)
        self.text_out_formula.set(pred)
        
    #計算実行
    def get_pic(self):
        #canvasの左上の位置取得
        pos_x= self.canvas.winfo_rootx()
        pos_y = self.canvas.winfo_rooty()

        #cancas内部（縁無し）のサイズ取得
        inner_size_w = int(self.canvas.cget("width"))
        inner_size_h = int(self.canvas.cget("height"))

        #cancas外部（縁込み）のサイズ取得
        outer_size_w = self.canvas.winfo_width()
        outer_size_h = self.canvas.winfo_height()

        #縁の太さ取得
        edge_x = (outer_size_w - inner_size_w)//2
        edge_y = (outer_size_h - inner_size_h)//2

        #canvasの画像取得
        img = ImageGrab.grab((pos_x + edge_x, pos_y + edge_y, pos_x + edge_x + 400 - 2, pos_y + inner_size_h + edge_y))
        return img

    #ペイント領域クリア処理
    def clear_canvas(self):
        self.canvas.delete("line")
        self.text_out_formula.set('')

    #ウィンドウクローズ処理
    def close_window(self):
        self.window.destroy()

    #ウィンドウ作成処理
    def create_window(self):
        window = tkinter.Tk()
        window.title('手書き数字認識')
        window.configure(width=444, height=530, bg='skyblue')
        window.minsize(444, 300)

        #ペイント領域
        self.canvas = tkinter.Canvas(window, bg="white", width=400, height=400)
        self.canvas.place(x=20, y=20)
        self.canvas.bind("<ButtonPress-1>", self.on_pressed)
        self.canvas.bind("<B1-Motion>", self.on_dragged)
        self.canvas.bind("<ButtonRelease-1>", self.on_released)

        #計算式出力用ラベル
        self.text_out_formula = tkinter.StringVar()
        self.label_out_formula = tkinter.Label(window, bg='white', width=10, font=(u'ＭＳ ゴシック', 30), anchor='c', textvariable=self.text_out_formula)
        self.label_out_formula.place(x=20, y=450)

        #ペイント領域クリアボタン
        clear_button = tkinter.Button(window, text="クリア", width=10, command=self.clear_canvas)
        clear_button.place(x=240, y=457)

        #終了ボタン
        exit_button = tkinter.Button(window, text="終了", width=10, command=self.close_window)
        exit_button.place(x=340, y=457)

        return window;

    def __init__(self):
        self.window = self.create_window();

    def run(self):
        self.window.mainloop()

Scribble().run()