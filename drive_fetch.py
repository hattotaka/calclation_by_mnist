"""Drive 上の JSON を GAS 経由で受け取る試作（同期コンフリクト対応の局所検証）。

ランチャー本体には組み込まない**単体のスクリプト**。会社端末の適当なローカル
フォルダに置き、標準ライブラリだけで動かす（pip 依存なし）。

流れ:
  1. 127.0.0.1 の空きポートで受け口（HTTP サーバ）を立てる
  2. GAS の Webアプリを既定のブラウザ（Chrome）で開く
     ── 引数は 対象ファイルの絶対パス / 受け口のポート / nonce
  3. GAS が絶対パスからファイルを特定し、中身を返す HTML を出す
  4. その HTML の JavaScript が受け口へ POST する
  5. nonce が一致した 1 回だけ受け取り、コンソールへ出す
  6. 前面のウィンドウが GAS の画面なら Ctrl+W を送ってタブを閉じる

Google とのやり取り（認証）はすべて Chrome が行う。Python が触るのは
127.0.0.1 に届いた送信だけ。

タブを閉じる件: GAS の画面を JavaScript から閉じることはできない（スクリプトが
開いたタブではない）ので、キー操作で閉じる。誤って別のアプリを閉じないよう、
**前面のウィンドウのタイトルが GAS の画面のタイトル（`Code.gs` の `setTitle`）で
始まるときだけ**送る。Chrome のウィンドウのタイトルは「表示中のタブのタイトル
- Google Chrome」なので、一致すれば閉じるのはそのタブ。新しいウィンドウに
タブ 1 つだけで開いていれば、ウィンドウごと閉じる。

実行前に書き換えるのは下の 2 つの変数だけ:
  WEBAPP_URL  … デプロイした Webアプリの URL（末尾 /exec）
  TARGET_PATH … 手元（Google ドライブ）の対象ファイルの絶対パス
"""
from __future__ import annotations

import ctypes
import json
import queue
import secrets
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ------------------------------------------------------------------ #
# 実行前に書き換える
# ------------------------------------------------------------------ #
WEBAPP_URL = "https://script.google.com/a/macros/example.co.jp/s/XXXXXXXXXXXXXXXX/exec"
TARGET_PATH = r"G:\マイドライブ\aaa\bbb\secrets.json"

# 受け取りを待つ秒数。初回は承認画面・127.0.0.1 へのアクセス許可で人の操作が
# 挟まるので長めに取る。
TIMEOUT_SECONDS = 180

# GAS の画面のタイトル（Code.gs の setTitle と同じ文字列にする）
PAGE_TITLE = "WDL 同期確認"

# 受け取ったあと、前面が GAS の画面になるのを待つ秒数。過ぎたら閉じずに諦める。
CLOSE_WAIT_SECONDS = 2.0


def _make_handler(nonce: str, results: "queue.Queue[dict]"):
    """nonce を照合して 1 回だけ結果を積むハンドラを作る。"""

    class Handler(BaseHTTPRequestHandler):
        def _cors(self) -> None:
            # GAS の画面は googleusercontent.com から来る。no-cors の送信なら
            # 本来不要だが、Chrome が事前確認（OPTIONS）を送ってきた場合に備える。
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
            try:
                data = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                print("[受け口] JSON として読めない送信を無視しました", file=sys.stderr)
                return
            if not isinstance(data, dict) or data.get("nonce") != nonce:
                print("[受け口] nonce が一致しない送信を無視しました", file=sys.stderr)
                return
            results.put(data)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass    # アクセスログは出さない（コンソールを結果だけにする）

    return Handler


def build_url(webapp_url: str, target_path: str, port: int, nonce: str) -> str:
    """Webアプリを開く URL（絶対パス・ポート・nonce を引数に載せる）。"""
    query = urllib.parse.urlencode({"path": target_path, "port": port, "nonce": nonce})
    sep = "&" if "?" in webapp_url else "?"
    return webapp_url + sep + query


def start_receiver(nonce: str) -> tuple[ThreadingHTTPServer, "queue.Queue[dict]"]:
    """127.0.0.1 の空きポートで受け口を立てる（裏のスレッドで動かす）。"""
    results: "queue.Queue[dict]" = queue.Queue()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(nonce, results))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, results


def fetch(webapp_url: str, target_path: str, timeout: float) -> dict | None:
    """Webアプリを開いて結果を待つ。時間内に届かなければ None。"""
    nonce = secrets.token_urlsafe(16)
    server, results = start_receiver(nonce)
    try:
        port = server.server_address[1]
        url = build_url(webapp_url, target_path, port, nonce)
        print("受け口: 127.0.0.1:%d" % port)
        print("開く URL: %s" % url)
        webbrowser.open(url)
        try:
            return results.get(timeout=timeout)
        except queue.Empty:
            return None
    finally:
        server.shutdown()
        server.server_close()


def foreground_title() -> str:
    """前面のウィンドウのタイトル（取れなければ空文字）。"""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def send_ctrl_w() -> None:
    """Ctrl+W を送る（押す → 離す）。"""
    VK_CONTROL, VK_W, KEYEVENTF_KEYUP = 0x11, 0x57, 0x0002
    kbd = ctypes.windll.user32.keybd_event
    kbd(VK_CONTROL, 0, 0, 0)
    kbd(VK_W, 0, 0, 0)
    kbd(VK_W, 0, KEYEVENTF_KEYUP, 0)
    kbd(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def close_page_tab(title: str, wait: float) -> bool:
    """前面が GAS の画面なら Ctrl+W で閉じる。閉じる操作を送ったら True。

    前面のタイトルが title で始まるときだけ送る。wait 秒のあいだ 0.1 秒ごとに見て、
    一度も一致しなければ何もしない（別のアプリを閉じないことを優先する）。
    """
    deadline = time.monotonic() + wait
    while True:
        current = foreground_title()
        if current.startswith(title):
            send_ctrl_w()
            return True
        if time.monotonic() >= deadline:
            print("[タブ] 前面が GAS の画面ではないので閉じませんでした（前面: %r）" % current)
            return False
        time.sleep(0.1)


def main() -> int:
    result = fetch(WEBAPP_URL, TARGET_PATH, TIMEOUT_SECONDS)
    if result is None:
        print("タイムアウト: %d 秒以内に GAS から届きませんでした" % TIMEOUT_SECONDS)
        return 1
    # エラーのときも GAS の画面は出ているので閉じる
    if close_page_tab(PAGE_TITLE, CLOSE_WAIT_SECONDS):
        print("[タブ] GAS の画面に Ctrl+W を送りました")
    if result.get("error"):
        print("GAS がエラーを返しました: %s" % result["error"])
        return 1
    print("ファイル ID: %s" % result.get("id"))
    print("----- 中身 -----")
    print(result.get("content", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
