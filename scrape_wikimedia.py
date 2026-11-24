import requests, time
from pathlib import Path

CATEGORIES = {
    
    "Manipuri":    ["Category:Manipuri", "Category:Manipuri dancers"],
}

SESSION = requests.Session()
# Wikimedia's policy: descriptive UA with contact info. Use a browser-like one:
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
})

def get_category_images(category, limit=1000):
    urls = []
    continue_token = {}
    while len(urls) < limit:
        params = {
            "action": "query", "format": "json", "formatversion": 2,
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 1024,
        }
        params.update(continue_token)
        resp = SESSION.get("https://commons.wikimedia.org/w/api.php",
                           params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  API returned {resp.status_code}, retrying in 5s...")
            time.sleep(5)
            continue
        try:
            data = resp.json()
        except Exception:
            print(f"  Non-JSON response: {resp.text[:200]!r}")
            time.sleep(5)
            continue
        pages = data.get("query", {}).get("pages", [])
        for p in pages:
            info = p.get("imageinfo", [])
            if info:
                thumb = info[0].get("thumburl") or info[0].get("url")
                if thumb:
                    urls.append(thumb)
        if "continue" not in data:
            break
        continue_token = data["continue"]
        time.sleep(1)
    return urls[:limit]

def download(url, path):
    try:
        img = SESSION.get(url, timeout=20)
        if img.status_code == 200 and len(img.content) > 10000:
            path.write_bytes(img.content)
            return True
    except Exception as e:
        print(f"    download error: {e}")
    return False

for cls, categories in CATEGORIES.items():
    folder = Path("data/raw") / cls
    folder.mkdir(parents=True, exist_ok=True)
    count = len([f for f in folder.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png",".webp"}])
    print(f"{cls}: starting at {count}")
    seen = set()
    for cat in categories:
        print(f"  fetching {cat} ...")
        urls = get_category_images(cat, limit=1000)
        print(f"  found {len(urls)} images")
        for u in urls:
            if count >= 1100:
                break
            if u in seen:
                continue
            seen.add(u)
            ext = Path(u.split("?")[0]).suffix.lower() or ".jpg"
            if ext not in {".jpg",".jpeg",".png",".webp"}:
                ext = ".jpg"
            if download(u, folder / f"wiki_{count}{ext}"):
                count += 1
        time.sleep(1)
        if count >= 1100:
            break
    print(f"{cls}: DONE at {count}")
