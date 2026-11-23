import requests

resp = requests.get("https://api.openverse.org/v1/images/",
    params={"q": "kuchipudi", "page_size": 5}, timeout=30)
print("Status code:", resp.status_code)
print(resp.text[:500])
