#!/usr/bin/env python
"""Build the WebShop Lucene search-index documents from a products JSON file.

WebShop's engine picks the index directory from num_products (1000 -> indexes_1k);
this writes the matching `search_engine/resources_1k/documents.jsonl` collection,
which is then indexed with pyserini at image-build time. Mirrors the upstream
`search_engine/convert_product_file_format.py`, but reads the small 1000-product
file (public) and writes the 1k collection directly.

Usage:
    build_index.py <items_shuffle_1000.json> <resources_out_dir>
"""
import json
import sys


def main() -> None:
    data_file, out_dir = sys.argv[1], sys.argv[2]
    with open(data_file) as f:
        all_products = json.load(f)
    print(f"Loaded {len(all_products)} products from {data_file}")

    docs = []
    for p in all_products:
        option_texts = []
        options = p.get("options", p.get("customization_options", {}))
        if options:
            for name, contents in options.items():
                if contents is None:
                    continue
                if isinstance(contents, list):
                    if contents and isinstance(contents[0], dict):
                        text = ", ".join(o.get("value", str(o)) for o in contents)
                    else:
                        text = ", ".join(str(o) for o in contents)
                else:
                    text = str(contents)
                option_texts.append(f"{name}: {text}")
        option_text = ", and ".join(option_texts)

        title = p.get("Title", p.get("name", ""))
        description = p.get("Description", p.get("full_description", ""))
        bullets = p.get("BulletPoints", p.get("small_description", [""]))
        bullet_text = (bullets[0] if bullets else "") if isinstance(bullets, list) else bullets

        docs.append({
            "id": p["asin"],
            "contents": " ".join([title, description, bullet_text, option_text]).lower(),
            "product": p,
        })

    import os

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "documents.jsonl")
    with open(out_file, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    print(f"Wrote {len(docs)} documents to {out_file}")


if __name__ == "__main__":
    main()
