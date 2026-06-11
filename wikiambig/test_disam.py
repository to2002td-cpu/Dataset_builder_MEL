"""
Explore all Wikipedia disambiguation subcategories.
Crawls Category:Disambiguation_categories, counts members for each subcategory,
and outputs a sorted CSV + pretty table.
"""

import requests
import time
import csv
from collections import defaultdict

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "disambig-explorer/1.0 (research)"})

API = "https://en.wikipedia.org/w/api.php"


def get_subcategories(category: str, depth: int = 0, max_depth: int = 2) -> list[dict]:
    """Recursively fetch subcategories of a category up to max_depth."""
    results = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "subcat",
        "cmlimit": 500,
        "format": "json",
    }

    while True:
        r = SESSION.get(API, params=params, timeout=30)
        data = r.json()
        members = data["query"]["categorymembers"]

        for m in members:
            name = m["title"].replace("Category:", "")
            results.append({"name": name, "depth": depth})
            # Recurse one level deeper if needed
            if depth < max_depth and "disambiguation" in name.lower():
                sub = get_subcategories(name, depth + 1, max_depth)
                results.extend(sub)

        if "continue" not in data:
            break
        params["cmcontinue"] = data["continue"]["cmcontinue"]
        time.sleep(0.1)

    return results


def count_members(category: str) -> int:
    """Count the number of pages (not subcats) in a category."""
    params = {
        "action": "query",
        "prop": "categoryinfo",
        "titles": f"Category:{category}",
        "format": "json",
    }
    r = SESSION.get(API, params=params, timeout=30)
    data = r.json()
    pages = data["query"]["pages"]
    for page in pages.values():
        info = page.get("categoryinfo", {})
        return info.get("pages", 0)
    return 0


def main():
    print("Fetching subcategories of Category:Disambiguation_categories...")
    raw = get_subcategories("Disambiguation_categories", depth=0, max_depth=1)

    # Deduplicate by name
    seen = {}
    for item in raw:
        name = item["name"]
        if name not in seen:
            seen[name] = item

    cats = list(seen.values())
    print(f"Found {len(cats)} unique subcategories. Counting members...")

    results = []
    for i, cat in enumerate(cats):
        name = cat["name"]
        count = count_members(name)
        results.append({"category": name, "pages": count, "depth": cat["depth"]})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cats)} done...")
        time.sleep(0.05)

    # Sort by page count descending
    results.sort(key=lambda x: x["pages"], reverse=True)

    # Save CSV
    csv_path = "/mnt/user-data/outputs/disambiguation_categories.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "pages", "depth"])
        writer.writeheader()
        writer.writerows(results)

    # Print top 60 as table
    print(f"\n{'Rank':<5} {'Pages':>7}  Category")
    print("-" * 80)
    for i, row in enumerate(results[:60], 1):
        print(f"{i:<5} {row['pages']:>7}  {row['category']}")

    print(f"\nFull results saved to {csv_path}")
    print(f"Total categories: {len(results)}")


if __name__ == "__main__":
    main()