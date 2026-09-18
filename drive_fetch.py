"""Drive 上の JSON を GAS 経由で受け取る試作（同期コンフリクト対応の局所検証）。

ランチャー本体には組み込まない**単体のスクリプト**。会社端末の適当なローカル
フォルダに置き、標準ライブラリだけで動かす（pip 依存なし）。

流れ:
  1. 127.0.0.1 の空きポートで受け口（HTTP サーバ）を立てる
  2. GAS の Webアプリを既定のブラウザ（Chrome）の**新規ウィンドウ**で開く
     ── 引数は 対象ファイルの絶対パス / 受け口のポート / nonce
  3. 開く前後の Chrome のウィンドウを比べて、**新しく出てきたウィンドウ（hwnd）**を
     見つけ、すぐ最小化する（MINIMIZE_MODE）
  4. GAS が絶対パスからファイルを特定し、中身を返す HTML を出す
  5. その HTML の JavaScript が受け口へ POST する
  6. nonce が一致した 1 回だけ受け取り、コンソールへ出す
  7. 新しく出てきたウィンドウのうち、タイトルが GAS の画面のものへ WM_CLOSE を送って閉じる
     （前面かどうかは問わない）。見つからなければ従来の Ctrl+W に退避する。
     閉じる前に、見つけた時点の大きさへ戻す（CLOSE_MODE。Chrome が閉じたウィンドウの
     大きさを覚えて次に使うため）

Google とのやり取り（認証）はすべて Chrome が行う。Python が触るのは
127.0.0.1 に届いた送信だけ。

ウィンドウを見つける件: 起動した chrome.exe は、Chrome がすでに動いていれば
「新しいウィンドウを開いて」と本体へ伝えてすぐ終わるので、プロセスからは
ウィンドウに辿り着けない。そこで**開く直前の Chrome のウィンドウの一覧**を取っておき、
あとから増えたものを「今回開いたウィンドウ」とみなす。
閉じるときは**タイトルが GAS の画面のタイトル（`Code.gs` の `setTitle`）で始まる**
ことも確かめる ── Chrome のウィンドウのタイトルは「表示中のタブのタイトル
- Google Chrome」なので、一致すればそのウィンドウの表示中のタブは GAS の画面。
WM_CLOSE はタブではなく**ウィンドウごと**閉じる（新規ウィンドウにタブ 1 つなので同じこと）。

実行前に書き換えるのは下の 2 つの変数だけ:
  WEBAPP_URL  … デプロイした Webアプリの URL（末尾 /exec）
  TARGET_PATH … 手元（Google ドライブ）の対象ファイルの絶対パス
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import secrets
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
import winreg
from ctypes import wintypes
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

# 受け取ったあと、前面が GAS の画面になるのを待つ秒数（Ctrl+W に退避したとき）。
CLOSE_WAIT_SECONDS = 2.0

# 開いたウィンドウをどう最小化するか（検証で切り替える）
#   "after" … 見つけ次第最小化する（一瞬は画面に出る）
#   "start" … 起動時に「最小化で開いて」と頼む（STARTUPINFO）＋ 見つけ次第最小化も併用。
#             Chrome が頼みを聞いたかは、見つけた時点で最小化済みだったかでコンソールに出る
#   "none"  … 最小化しない（hwnd で閉じる部分だけを試す）
MINIMIZE_MODE = "start"

# 閉じる前にウィンドウの大きさをどう戻すか（検証で切り替える）
#   Chrome は閉じたウィンドウの「元に戻したときの大きさ」を覚え、次の新規ウィンドウに使う。
#   描き始めのうちに最小化すると、その大きさが小さいまま保存されることがある（会社で観察）。
#   戻す先は「見つけた時点（最小化する前）の大きさ」。
#   "restore"   … 透明（α=0）にしてから、焦点を奪わずに元の大きさで表示 → 少し待って閉じる
#   "placement" … 最小化したまま「元に戻したときの大きさ」だけを書き換えて閉じる（画面に出ない）
#   "none"      … 何もせず閉じる（前回の版と同じ）
CLOSE_MODE = "restore"

# "restore" で表示してから閉じるまで待つ秒数（Chrome が大きさを受け取る間）
RESTORE_SETTLE_SECONDS = 0.3

# 開いてから新しいウィンドウを探し続ける秒数（Chrome が起動していない状態からも含む）
WINDOW_SEARCH_SECONDS = 15.0

# ウィンドウを探す／結果を待つ間隔（秒）。短いほど画面に出ている時間が短い
POLL_SECONDS = 0.02


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


# 実行ファイル名（小文字）→ 新規ウィンドウを開くフラグ
# （ランチャー本体の workdesk/services/browser.py と同じ表。試作は単体で動かすので写してある）
_NEW_WINDOW_FLAG = {
    "chrome.exe": "--new-window",
    "msedge.exe": "--new-window",
    "brave.exe": "--new-window",
    "vivaldi.exe": "--new-window",
    "opera.exe": "--new-window",
    "firefox.exe": "-new-window",
    "librewolf.exe": "-new-window",
}


def _default_browser_exe() -> str | None:
    """既定ブラウザ（https）の実行ファイルのパス。取れなければ None。"""
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\Shell\Associations"
            r"\UrlAssociations\https\UserChoice",
        ) as key:
            progid, _ = winreg.QueryValueEx(key, "ProgId")
        with winreg.OpenKey(
            winreg.HKEY_CLASSES_ROOT, rf"{progid}\shell\open\command"
        ) as key:
            command, _ = winreg.QueryValueEx(key, None)
        parts = shlex.split(str(command), posix=False)
    except (OSError, ValueError):
        return None
    if not parts:
        return None
    exe = parts[0].strip('"')
    return exe if os.path.isfile(exe) else None


def open_in_new_window(url: str, *, minimized: bool = False) -> bool:
    """既定ブラウザの新規ウィンドウで URL を開く。

    既存のウィンドウのタブで開くと、そのウィンドウ（後ろにいた／最小化していた）が
    前面に出てきて、タブを閉じたあとも前面に残る。新規ウィンドウならタブと一緒に
    ウィンドウごと消えるので、開く前の状態に戻る。
    minimized なら STARTUPINFO で「最小化で開いて」と頼む（Chrome が聞くとは限らない）。
    新規ウィンドウで開けたら True。無理なら既存ウィンドウのタブで開いて False。
    """
    exe = _default_browser_exe()
    flag = _NEW_WINDOW_FLAG.get(os.path.basename(exe).lower()) if exe else None
    if flag:
        startupinfo = None
        if minimized:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = SW_SHOWMINNOACTIVE
        try:
            subprocess.Popen([exe, flag, url], close_fds=True, startupinfo=startupinfo)  # noqa: S603
            return True
        except OSError:
            pass
    webbrowser.open(url)
    return False


# ------------------------------------------------------------------ #
# ウィンドウの操作（user32）
# ------------------------------------------------------------------ #
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
_user32.EnumWindows.restype = wintypes.BOOL
_user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.GetWindowLongW.restype = wintypes.LONG
_user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
_user32.GetWindow.restype = wintypes.HWND
_user32.IsWindow.argtypes = [wintypes.HWND]
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.IsIconic.argtypes = [wintypes.HWND]
_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.PostMessageW.restype = wintypes.BOOL
_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

CHROME_WINDOW_CLASS = "Chrome_WidgetWin_1"
GWL_STYLE = -16
GW_OWNER = 4
WS_MINIMIZEBOX = 0x00020000
SW_MINIMIZE = 6
SW_SHOWMINNOACTIVE = 7
WM_CLOSE = 0x0010
SW_SHOWNOACTIVATE = 4
SW_SHOWMAXIMIZED = 3
WPF_RESTORETOMAXIMIZED = 0x2
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
LWA_ALPHA = 0x2


class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.UINT),
        ("flags", wintypes.UINT),
        ("showCmd", wintypes.UINT),
        ("ptMinPosition", wintypes.POINT),
        ("ptMaxPosition", wintypes.POINT),
        ("rcNormalPosition", wintypes.RECT),
    ]


_user32.GetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]
_user32.SetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]
_user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
_user32.SetWindowLongW.restype = wintypes.LONG
_user32.SetLayeredWindowAttributes.argtypes = [
    wintypes.HWND, wintypes.COLORREF, wintypes.BYTE, wintypes.DWORD]


def get_placement(hwnd: int) -> WINDOWPLACEMENT:
    """ウィンドウの配置（rcNormalPosition＝元に戻したときの位置と大きさ）。"""
    wp = WINDOWPLACEMENT()
    wp.length = ctypes.sizeof(WINDOWPLACEMENT)
    _user32.GetWindowPlacement(hwnd, ctypes.byref(wp))
    return wp


def describe(wp: WINDOWPLACEMENT) -> str:
    """配置をコンソール用の短い文字列にする。"""
    r = wp.rcNormalPosition
    return "元の大きさ %dx%d（%d,%d）/ showCmd=%d" % (
        r.right - r.left, r.bottom - r.top, r.left, r.top, wp.showCmd)


def make_transparent(hwnd: int) -> None:
    """ウィンドウを透明（α=0）にする。閉じる直前にだけ使う。"""
    ex = _user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    _user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED)
    _user32.SetLayeredWindowAttributes(hwnd, 0, 0, LWA_ALPHA)


def window_title(hwnd: int) -> str:
    """ウィンドウのタイトル（取れなければ空文字）。"""
    length = _user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def process_exe_name(hwnd: int) -> str:
    """ウィンドウを持つプロセスの実行ファイル名（小文字。取れなければ空文字）。"""
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if not _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return ""
        return os.path.basename(buf.value).lower()
    finally:
        _kernel32.CloseHandle(handle)


def chrome_windows() -> set[int]:
    """Chrome の「ブラウザのウィンドウ」の hwnd の集合。

    同じクラス名はメニュー・ツールチップ・候補の吹き出しにも、Electron 製のアプリ
    （Teams・VS Code 等）にも使われるので、表示中・持ち主なし・最小化ボタンあり・
    **既定ブラウザと同じ実行ファイルのプロセス**のものに絞る。
    最小化されているウィンドウも含む（IsWindowVisible は最小化でも真）。
    """
    exe = _default_browser_exe()
    browser = os.path.basename(exe).lower() if exe else "chrome.exe"
    found: set[int] = set()

    def _each(hwnd, _lparam):
        cls = ctypes.create_unicode_buffer(64)
        _user32.GetClassNameW(hwnd, cls, 64)
        if (cls.value == CHROME_WINDOW_CLASS
                and _user32.IsWindowVisible(hwnd)
                and not _user32.GetWindow(hwnd, GW_OWNER)
                and _user32.GetWindowLongW(hwnd, GWL_STYLE) & WS_MINIMIZEBOX
                and process_exe_name(hwnd) == browser):
            found.add(hwnd)
        return True

    _user32.EnumWindows(_WNDENUMPROC(_each), 0)
    return found


def find_opened_window(before: set[int], title: str) -> int | None:
    """開く前に無かった Chrome のウィンドウを 1 つ選ぶ。まだ決められなければ None。

    新しいウィンドウが 1 つだけならそれ（ページが読み込まれる前＝タイトルが
    まだ付いていなくても選べるので、早く最小化できる）。
    Chrome を起動していない状態から開くと、前回のウィンドウの復元などで
    複数増えることがあるので、そのときはタイトルが一致するものが出るまで待つ。
    """
    new = chrome_windows() - before
    if len(new) == 1:
        return next(iter(new))
    for hwnd in new:
        if window_title(hwnd).startswith(title):
            return hwnd
    return None


def close_opened_window(before: set[int], title: str,
                        saved: dict[int, WINDOWPLACEMENT]) -> int | None:
    """開く前に無かった Chrome のウィンドウのうち、タイトルが一致するものへ WM_CLOSE を送る。

    送った hwnd を返す。見つからなければ None（何も閉じない）。
    前面かどうかは問わない ── 最小化していても、別のウィンドウを前面に出していても閉じられる。
    閉じる前に、saved（見つけた時点＝最小化する前の配置）の大きさへ戻す（CLOSE_MODE）。
    """
    for hwnd in chrome_windows() - before:
        if window_title(hwnd).startswith(title):
            restore_size(hwnd, saved.get(hwnd))
            _user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            return hwnd
    return None


def restore_size(hwnd: int, original: WINDOWPLACEMENT | None) -> None:
    """閉じる前に「元に戻したときの大きさ」を original へ戻す（CLOSE_MODE で方法を選ぶ）。"""
    now = get_placement(hwnd)
    print("[窓] 閉じる前: %s" % describe(now))
    if CLOSE_MODE == "none" or original is None:
        return
    wp = get_placement(hwnd)
    wp.rcNormalPosition = original.rcNormalPosition
    if CLOSE_MODE == "restore":
        make_transparent(hwnd)
        if original.showCmd == SW_SHOWMAXIMIZED:
            # 最大化で開いていたなら最大化で閉じる（Chrome は閉じたときの実際の状態を保存する
            # ので、最小化や通常のまま閉じると次から最大化で開かなくなる）。焦点は移る
            wp.showCmd = SW_SHOWMAXIMIZED
        else:
            wp.showCmd = SW_SHOWNOACTIVATE     # 元の大きさで表示するが焦点は奪わない
        _user32.SetWindowPlacement(hwnd, ctypes.byref(wp))
        time.sleep(RESTORE_SETTLE_SECONDS)
    else:  # "placement"
        wp.showCmd = SW_SHOWMINNOACTIVE    # 最小化のまま、元に戻したときの大きさだけ書き換える
        if original.showCmd == SW_SHOWMAXIMIZED:
            wp.flags |= WPF_RESTORETOMAXIMIZED   # 「元に戻すと最大化」も引き継ぐ
        _user32.SetWindowPlacement(hwnd, ctypes.byref(wp))
    print("[窓] 大きさを戻しました（%s）: %s" % (CLOSE_MODE, describe(get_placement(hwnd))))


def start_receiver(nonce: str) -> tuple[ThreadingHTTPServer, "queue.Queue[dict]"]:
    """127.0.0.1 の空きポートで受け口を立てる（裏のスレッドで動かす）。"""
    results: "queue.Queue[dict]" = queue.Queue()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(nonce, results))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, results


def fetch(webapp_url: str, target_path: str, timeout: float
          ) -> tuple[dict | None, set[int], dict[int, WINDOWPLACEMENT]]:
    """Webアプリを開いて結果を待つ。

    (結果, 開く前の Chrome のウィンドウの集合, 見つけた時点の配置) を返す。
    結果は時間内に届かなければ None。
    待っている間に、新しく出てきたウィンドウを探して最小化する。
    """
    saved: dict[int, WINDOWPLACEMENT] = {}
    nonce = secrets.token_urlsafe(16)
    server, results = start_receiver(nonce)
    try:
        port = server.server_address[1]
        url = build_url(webapp_url, target_path, port, nonce)
        print("受け口: 127.0.0.1:%d" % port)
        print("開く URL: %s" % url)
        before = chrome_windows()
        for h in before:    # 参考: 元からある Chrome のウィンドウの大きさ
            print("[窓] 元からある Chrome: %s" % describe(get_placement(h)))
        started = time.monotonic()
        if open_in_new_window(url, minimized=(MINIMIZE_MODE == "start")):
            print("新規ウィンドウで開きました（最小化: %s）" % MINIMIZE_MODE)
        else:
            print("新規ウィンドウで開けなかったので、既存ウィンドウのタブで開きました")

        searching = True
        while True:
            now = time.monotonic()
            if searching:
                hwnd = find_opened_window(before, PAGE_TITLE)
                if hwnd:
                    searching = False
                    iconic = bool(_user32.IsIconic(hwnd))
                    saved[hwnd] = get_placement(hwnd)
                    print("[窓] 見つけました: hwnd=0x%X / %.2f 秒後 / 最小化済み=%s / タイトル=%r"
                          % (hwnd, now - started, iconic, window_title(hwnd)))
                    print("[窓] 見つけた時点: %s" % describe(saved[hwnd]))
                    if MINIMIZE_MODE != "none" and not iconic:
                        # SW_MINIMIZE は次のウィンドウを前面にする（SW_SHOWMINNOACTIVE だと
                        # 見えない Chrome に入力の焦点が残りうる）
                        _user32.ShowWindow(hwnd, SW_MINIMIZE)
                        print("[窓] 最小化しました")
                elif now - started >= WINDOW_SEARCH_SECONDS:
                    searching = False
                    print("[窓] %.0f 秒以内に新しいウィンドウが見つかりませんでした" % WINDOW_SEARCH_SECONDS)
            try:
                return results.get(timeout=POLL_SECONDS), before, saved
            except queue.Empty:
                if now - started >= timeout:
                    return None, before, saved
    finally:
        server.shutdown()
        server.server_close()


def foreground_title() -> str:
    """前面のウィンドウのタイトル（取れなければ空文字）。"""
    hwnd = _user32.GetForegroundWindow()
    return window_title(hwnd) if hwnd else ""


def send_ctrl_w() -> None:
    """Ctrl+W を送る（押す → 離す）。"""
    VK_CONTROL, VK_W, KEYEVENTF_KEYUP = 0x11, 0x57, 0x0002
    kbd = _user32.keybd_event
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
    result, before, saved = fetch(WEBAPP_URL, TARGET_PATH, TIMEOUT_SECONDS)
    if result is None:
        print("タイムアウト: %d 秒以内に GAS から届きませんでした" % TIMEOUT_SECONDS)
        return 1
    # エラーのときも GAS の画面は出ているので閉じる
    hwnd = close_opened_window(before, PAGE_TITLE, saved)
    if hwnd:
        print("[窓] hwnd=0x%X に WM_CLOSE を送りました" % hwnd)
    else:
        print("[窓] 閉じる対象のウィンドウが見つからないので、Ctrl+W に退避します")
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
