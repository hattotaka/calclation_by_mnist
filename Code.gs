/**
 * Drive 上の JSON を 127.0.0.1 の受け口へ返す試作（drive_fetch.py と対）。
 *
 * デプロイ: 種類＝ウェブアプリ / 次のユーザーとして実行＝ウェブアプリにアクセスしているユーザー /
 *           アクセスできるユーザー＝（会社名）内の全員
 *
 * 引数:
 *   path  … 手元の絶対パス（例: G:\マイドライブ\aaa\bbb\secrets.json）
 *   port  … Python の受け口のポート
 *   nonce … Python が照合する使い捨ての印（そのまま送り返す）
 */

// 絶対パスの中で「ここからがマイドライブ」を示すフォルダ名（表示言語で変わる）
var MY_DRIVE_NAMES = ['マイドライブ', 'My Drive'];

function doGet(e) {
  var p = e.parameter;
  if (!/^\d{1,5}$/.test(p.port || '')) {
    return HtmlService.createHtmlOutput('invalid port');
  }

  var payload;
  try {
    var file = resolveByPath_(p.path || '');
    payload = {
      id: file.getId(),
      name: file.getName(),
      content: file.getBlob().getDataAsString('UTF-8')
    };
  } catch (err) {
    payload = { error: String(err && err.message || err) };
  }
  payload.nonce = p.nonce || '';

  var target = 'http://127.0.0.1:' + p.port + '/';
  // 中身に "</script>" があっても壊れないよう、< をエスケープして埋め込む
  var body = JSON.stringify(JSON.stringify(payload)).replace(/</g, '\\u003c');

  var html =
    '<p id="s">送信中…</p>' +
    '<script>' +
    'var s = document.getElementById("s");' +
    'fetch(' + JSON.stringify(target) + ', {' +
    '  method: "POST", mode: "no-cors",' +
    '  headers: { "Content-Type": "text/plain" },' +
    '  body: ' + body +
    '}).then(function () { s.textContent = "完了しました。このタブは閉じてかまいません。"; })' +
    '  .catch(function (e) { s.textContent = "送信に失敗しました: " + e; });' +
    '</script>';

  return HtmlService.createHtmlOutput(html).setTitle('WDL 同期確認');
}

/**
 * 絶対パスからファイルを特定する。
 * 「マイドライブ」より後ろを、ルートフォルダから名前で 1 段ずつたどる。
 * 0 件・複数件はエラーにする（どれかを勝手に選ばない）。
 */
function resolveByPath_(path) {
  var parts = path.split(/[\\\/]+/).filter(function (x) { return x !== ''; });
  var start = -1;
  for (var i = 0; i < parts.length; i++) {
    if (MY_DRIVE_NAMES.indexOf(parts[i]) >= 0) { start = i + 1; break; }
  }
  if (start < 0) throw new Error('パスに「マイドライブ」が見つかりません: ' + path);
  var rel = parts.slice(start);
  if (rel.length === 0) throw new Error('ファイル名がありません: ' + path);

  var folder = DriveApp.getRootFolder();
  for (var j = 0; j < rel.length - 1; j++) {
    folder = single_(folder.getFoldersByName(rel[j]), 'フォルダ', rel[j]);
  }
  return single_(folder.getFilesByName(rel[rel.length - 1]), 'ファイル', rel[rel.length - 1]);
}

/** ゴミ箱を除いて 1 件だけのときに返す。 */
function single_(iter, kind, name) {
  var found = [];
  while (iter.hasNext()) {
    var x = iter.next();
    if (!x.isTrashed()) found.push(x);
  }
  if (found.length === 0) throw new Error(kind + 'が見つかりません: ' + name);
  if (found.length > 1) throw new Error(kind + 'が ' + found.length + ' 件あります: ' + name);
  return found[0];
}
