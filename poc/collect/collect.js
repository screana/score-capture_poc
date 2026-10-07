// Bing画像検索から採点画面のサムネイルを集めるブラウザ用スニペット。
// bing.com を開いたタブのコンソール（または自動操作のJS実行）に貼って使う。
//
//   await __list("精密採点Ai Heart 結果")                 // 候補一覧（最大70件、重複除去）
//   JSON.stringify(await __grab2("精密採点Ai Heart 結果", 0, 10, 300))
//
// __grab2 は、サムネイルを幅 W px の JPEG に再圧縮して base64 で返す。
// クラウド環境からは画像サイトに直接アクセスできないので、この戻り値を
// ツールの結果ファイル経由で受け取り、decode.py で画像に戻していた。
// ローカル環境なら、普通にダウンロードして保存するだけでよい。
// 1回の受け渡しは約25万文字が上限だったので、W=300 で10枚ずつに分けていた。

window.__lists = window.__lists || {};

window.__list = async (q, pages = 2) => {
  if (window.__lists[q]) return window.__lists[q];
  let all = [];
  for (let p = 0; p < pages; p++) {
    const html = await fetch("/images/async?q=" + encodeURIComponent(q) + "&first=" + p * 35 + "&count=35&mmasync=1").then(r => r.text());
    const doc = new DOMParser().parseFromString(html, "text/html");
    all = all.concat([...doc.querySelectorAll("a.iusc")]
      .map(a => { try { return JSON.parse(a.getAttribute("m")); } catch (e) { return null; } })
      .filter(Boolean));
  }
  const seen = new Set();
  all = all.filter(m => {
    const id = (m.turl.match(/id=([^&]+)/) || [])[1];
    if (!id || seen.has(id)) return false;
    seen.add(id); m.id = id; return true;
  });
  window.__lists[q] = all;
  return all;
};

window.__grab2 = async (q, start, count, W = 360, quality = 0.72) => {
  const items = (await window.__list(q)).slice(start, start + count);
  const out = [];
  await Promise.all(items.map(async m => {
    try {
      const b = await fetch("https://th.bing.com/th/id/" + m.id + "?w=640&rs=1&pid=1.7").then(r => r.blob());
      const bmp = await createImageBitmap(b);
      const c = document.createElement("canvas");
      const s = W / bmp.width; c.width = W; c.height = Math.round(bmp.height * s);
      c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
      out.push({ id: m.id, t: (m.t || "").slice(0, 80), purl: m.purl, murl: m.murl, ow: bmp.width, oh: bmp.height,
                 b64: c.toDataURL("image/jpeg", quality).split(",")[1] });
    } catch (e) { out.push({ id: m.id, err: String(e) }); }
  }));
  return out;
};
