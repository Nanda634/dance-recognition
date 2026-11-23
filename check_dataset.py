from pathlib import Path

CLASSES = ["Bharatanatyam", "Kathak", "Kuchipudi", "Odissi",
           "Kathakali", "Manipuri", "Mohiniyattam", "Sattriya"]

RAW = Path("data/raw")
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

print(f"{'Class':<15}{'Count':>6}{'Needed':>8}  Status")
print("-" * 45)
total = 0
for cls in CLASSES:
    folder = RAW / cls
    if not folder.exists():
        print(f"{cls:<15}{0:>6}{1000:>8}  ❌ folder missing")
        continue
    n = len([f for f in folder.iterdir() if f.suffix.lower() in VALID_EXT])
    total += n
    status = "✅ done" if n >= 1000 else "⚠️ short"
    print(f"{cls:<15}{n:>6}{max(0, 1000 - n):>8}  {status}")
print("-" * 45)
print(f"{'TOTAL':<15}{total:>6}")
