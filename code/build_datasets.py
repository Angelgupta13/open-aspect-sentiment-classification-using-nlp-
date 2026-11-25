import xml.etree.ElementTree as ET
import csv, json, html

RESTAURANT_ENTITY_WORD = {
    "AMBIENCE": "ambience", "DRINKS": "drinks", "FOOD": "food",
    "LOCATION": "location", "RESTAURANT": "restaurant", "SERVICE": "service",
}
LAPTOP_ENTITY_WORD = {
    "BATTERY": "battery", "COMPANY": "company", "CPU": "processor",
    "DISPLAY": "screen", "FANS_COOLING": "fan", "GRAPHICS": "graphics card",
    "HARDWARE": "hardware", "HARD_DISC": "hard drive", "KEYBOARD": "keyboard",
    "LAPTOP": "laptop", "MEMORY": "memory", "MOTHERBOARD": "motherboard",
    "MOUSE": "mouse", "MULTIMEDIA_DEVICES": "multimedia", "OPTICAL_DRIVES": "optical drive",
    "OS": "operating system", "PORTS": "ports", "POWER_SUPPLY": "power supply",
    "SHIPPING": "shipping", "SOFTWARE": "software", "SUPPORT": "support", "WARRANTY": "warranty",
}

def parse(path):
    tree = ET.parse(path)
    rows = []
    for sent in tree.getroot().iter("sentence"):
        text_el = sent.find("text")
        if text_el is None or text_el.text is None:
            continue
        text = text_el.text
        ops = sent.find("Opinions")
        if ops is None:
            continue
        for op in ops.findall("Opinion"):
            polarity = op.get("polarity")
            if polarity == "conflict" or polarity is None:
                continue
            category = op.get("category")
            target = op.get("target")
            rows.append({
                "sentence_id": sent.get("id"),
                "text": text,
                "category": category,
                "target": target,
                "polarity": polarity,
            })
    return rows

def build(domain, entity_word_map, train_path, test_path, outdir):
    for split, path in [("train", train_path), ("test", test_path)]:
        rows = parse(path)
        term_rows, cat_rows = [], []
        for r in rows:
            entity = r["category"].split("#")[0]
            word = entity_word_map.get(entity, entity.lower().replace("_", " "))
            word_in_text = word.lower() in r["text"].lower()
            is_explicit = r["target"] not in (None, "NULL")
            aspect_term = r["target"] if is_explicit else word
            term_rows.append({
                "sentence_id": r["sentence_id"], "text": r["text"],
                "aspect": aspect_term, "polarity": r["polarity"],
                "explicit": is_explicit, "category": r["category"],
            })
            cat_rows.append({
                "sentence_id": r["sentence_id"], "text": r["text"],
                "category": r["category"], "entity": entity, "aspect_word": word,
                "polarity": r["polarity"], "word_in_text": word_in_text,
            })
        with open(f"{outdir}/{domain}_2016_{split}_term.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(term_rows[0].keys())); w.writeheader(); w.writerows(term_rows)
        with open(f"{outdir}/{domain}_2016_{split}_category.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys())); w.writeheader(); w.writerows(cat_rows)
        n_explicit = sum(1 for r in term_rows if r["explicit"])
        pol_counts = {}
        for r in term_rows:
            pol_counts[r["polarity"]] = pol_counts.get(r["polarity"], 0) + 1
        print(f"{domain} {split}: {len(rows)} opinions, {n_explicit} explicit-target ({n_explicit/len(rows):.1%}), polarity={pol_counts}")

build("restaurant", RESTAURANT_ENTITY_WORD, "restaurant_train.xml", "restaurant_test.xml", "data")
build("laptop", LAPTOP_ENTITY_WORD, "laptop_train.xml", "laptop_test.xml", "data")
