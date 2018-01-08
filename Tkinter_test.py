import tkinter
from PIL import ImageGrab
import pred_mnist as pm

class Scribble:
    #文字描画開始時の初期処理
    def on_pressed(self, event):
        self.sx = event.x
        self.sy = event.y
        self.paint_cnt_up = True
        self.cnt = 0
        self.text_out_formula.set('')
        self.text_out_result.set('')

    #文字の描画処理
    def on_dragged(self, event):
        if self.paint_cnt_up == True:
            self.paint_cnt = self.paint_cnt + 1
            self.paint_cnt_up = False

        self.cnt = self.cnt +1

#        if self.sy != event.y:
#            self.canvas.create_oval(event.x, event.y, event.x, event.y, outline = self.color.get(), width = 14, tag="line" + str(self.paint_cnt))

        if self.cnt == 3:
            self.canvas.create_line(self.sx, self.sy, event.x, event.y, fill = self.color.get(), width = self.width.get(), tag="line" + str(self.paint_cnt))

            self.cnt = 0
            self.sx = event.x
            self.sy = event.y

    #計算実行
    def calc(self):
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
        #im = ImageGrab.grab((pos_x + edge_x, pos_y + edge_y, pos_x + inner_size_w + edge_x, pos_y + inner_size_h + edge_y))
        im1 = ImageGrab.grab((pos_x + edge_x, pos_y + edge_y, pos_x + edge_x + 400 - 2, pos_y + inner_size_h + edge_y))
        im2 = ImageGrab.grab((pos_x + edge_x + 402, pos_y + edge_y, pos_x + edge_x + 800 - 2, pos_y + inner_size_h + edge_y))
        im3 = ImageGrab.grab((pos_x + edge_x + 802, pos_y + edge_y, pos_x + edge_x + 1200 - 2, pos_y + inner_size_h + edge_y))
        im1.save(r'img\1_mnist.bmp', 'bmp', quality=100, optimize=True)
        im2.save(r'img\2_mnist.bmp', 'bmp', quality=100, optimize=True)
        im3.save(r'img\3_mnist.bmp', 'bmp', quality=100, optimize=True)
        pred1 = pm.test(im1)
        pred2 = pm.test(im2)
        pred3 = pm.test(im3)

        out_formula = str(pred1) + str(pred2) + str(pred3)
        self.text_out_formula.set(out_formula)
        self.text_out_result.set('7')
        #im.show()

    #ペイント領域を一つ前の状態へ戻す処理
    def undo(self):
        if self.paint_cnt != 0:
            self.canvas.delete("line" + str(self.paint_cnt))
            self.paint_cnt = self.paint_cnt - 1
        self.text_out_formula.set('')
        self.text_out_result.set('')

    #ペイント領域クリア処理
    def clear_canvas(self):
        for i in range(self.paint_cnt):
            self.canvas.delete("line" + str(i + 1))
        self.text_out_formula.set('')
        self.text_out_result.set('')

    #ウィンドウクローズ処理
    def close_window(self):
        self.window.destroy()

    #ウィンドウ作成処理
    def create_window(self):
        self.paint_cnt = 0
        window = tkinter.Tk()
        window.title('手書き計算機')
        window.configure(width=1240, height=800, bg='skyblue')
        window.minsize(1240, 300)

        #ペイント領域
        self.canvas = tkinter.Canvas(window, bg="white", width=1200, height=500)
        self.canvas.place(x=20, y=20)
        self.canvas.bind("<ButtonPress-1>", self.on_pressed)
        self.canvas.bind("<B1-Motion>", self.on_dragged)
        self.canvas.create_line(400, 0, 400, 500, fill = 'black', width = 1, tag="separator")
        self.canvas.create_line(800, 0, 800, 500, fill = 'black', width = 1, tag="separator")
        self.canvas.create_line(1200, 0, 1200, 500, fill = 'black', width = 1, tag="separator")

        #ラベル（計算式）
        self.label_formula = tkinter.Label(window, text='計算式', bg='skyblue', width=10, font=(u'ＭＳ ゴシック', 20), anchor='w')
        self.label_formula.place(x=100, y=550)

        #計算式出力用ラベル
        self.text_out_formula = tkinter.StringVar()
        self.label_out_formula = tkinter.Label(window, bg='white', width=20, font=(u'ＭＳ ゴシック', 30), anchor='w', textvariable=self.text_out_formula)
        self.label_out_formula.place(x=100, y=590)

        #ラベル（計算結果）
        self.label2 = tkinter.Label(window, text='計算結果', bg='skyblue', width=10, font=(u'ＭＳ ゴシック', 20), anchor='w')
        self.label2.place(x=100, y=660)

        #計算結果出力用ラベル
        self.text_out_result = tkinter.StringVar()
        self.label_out_result = tkinter.Label(window, bg='white', width=20, font=(u'ＭＳ ゴシック', 30), anchor='w', textvariable=self.text_out_result)
        self.label_out_result.place(x=100, y=700)

        #ラベル（ペイントカラー）
        self.label_formula = tkinter.Label(window, text='文字色', bg='skyblue', width=10, font=(u'ＭＳ ゴシック', 20), anchor='w')
        self.label_formula.place(x=580, y=550)

        #ペイントカラー選択
        COLORS = ["black", "red", "blue", "green"]
        self.color = tkinter.StringVar()
        self.color.set(COLORS[0])
        b = tkinter.OptionMenu(window, self.color, *COLORS)
        b.place(x=590, y=600)

        #ラベル（字の太さ）
        self.label_formula = tkinter.Label(window, text='文字サイズ', bg='skyblue', width=10, font=(u'ＭＳ ゴシック', 20), anchor='w')
        self.label_formula.place(x=700, y=550)

        #字の太さ
        self.width = tkinter.Scale(window, from_=5, to=20, orient=tkinter.HORIZONTAL)
        self.width.set(10)
        self.width.place(x=720, y=600)

        #計算ボタン
        calc_button = tkinter.Button(window, text="計算", width=15, command=self.calc)
        calc_button.place(x=570, y=712)

        #ペイント領域を一つ前の状態へ戻すボタン
        undo_button = tkinter.Button(window, text="戻る", width=15, command=self.undo)
        undo_button.place(x=720, y=712)

        #ペイント領域クリアボタン
        clear_button = tkinter.Button(window, text="クリア", width=15, command=self.clear_canvas)
        clear_button.place(x=870, y=712)

        #終了ボタン
        exit_button = tkinter.Button(window, text="終了", width=15, command=self.close_window)
        exit_button.place(x=1020, y=712)

        return window;

    def __init__(self):
        self.window = self.create_window();

    def run(self):
        self.window.mainloop()

Scribble().run()
