import requests, time
from pathlib import Path

QUERIES = {
    "Manipuri": ["manipuri dance", "manipuri", "manipuri dancer"],
}

def download_image(url, save_path):
    try:
        img = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if img.status_code == 200 and len(img.content) > 10000:
            save_path.write_bytes(img.content)
            return True
        else:
            print(f"    skipped: status {img.status_code}, size {len(img.content)}")
    except Exception as e:
        print(f"    ERROR: {e}")
    return False

def scrape(query, save_dir, target, existing):
    downloaded = existing
    for page in range(1, 30):
        if downloaded >= target:
            break
        resp = requests.get("https://api.openverse.org/v1/images/",
            params={"q": query, "page": page, "page_size": 50}, timeout=30)
        results = resp.json().get("results", [])
        print(f"  [{query}] page {page}: {len(results)} results")
        if not results:
            break
        for item in results:
            if downloaded >= target:
                break
            # try full image first, then thumbnail fallback
            url = item.get("url") or item.get("thumbnail")
            if not url:
                continue
            ext = Path(url.split("?")[0]).suffix.lower() or ".jpg"
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"
            fname = save_dir / f"{query.replace(' ','_')}_{downloaded}{ext}"
            if download_image(url, fname):
                downloaded += 1
        time.sleep(1)
    return downloaded

for cls, queries in QUERIES.items():
    folder = Path("data/raw") / cls
    folder.mkdir(parents=True, exist_ok=True)
    current = len([f for f in folder.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png",".webp"}])
    print(f"{cls}: starting at {current}")
    for q in queries:
        if current >= 500:
            break
        current = scrape(q, folder, 500, current)
        print(f"{cls}: now at {current}")
    print(f"{cls}: DONE at {current}")
