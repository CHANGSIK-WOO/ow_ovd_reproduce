# tools/encode_attributes_from_json.py
# in config file, att_embeddings = 'data/VOC2007/SOWOD/att_t_1_embeddings.pth'
# in CLI, python tools/encode_attributes_from_json.py \
#   --in_json unique_attributes.json \
#   --out_pth data/VOC2007/SOWOD/att_t_1_embeddings.pth

import json, torch, argparse
from transformers import AutoTokenizer, CLIPTextModelWithProjection

CATEGORIES = ["Shape","Color","Texture","Size","Context",
              "Features","Appearance","Behavior","Environment","Material"]

def load_attributes(path):
    """Category JSON: { "Shape":[..], "Color":[..], ... }"""
    data = json.load(open(path, "r", encoding="utf-8"))
    pairs = []
    for cat, att_list in data.items():
        for att in att_list:
            att = str(att).strip()
            if att:
                pairs.append((cat, att))

    # dedup
    seen, uniq = set(), []
    for cat, att in pairs:
        key = f"{cat.lower()}::{att.lower()}"
        if key not in seen:
            seen.add(key)
            uniq.append((cat, att))
    return uniq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attribute embeddings from category JSON")
    parser.add_argument("--in_json", type=str, default="unique_attributes.json",
                        help="Input attribute JSON file (category JSON format)")
    parser.add_argument("--out_pth", type=str, default="data/VOC2007/SOWOD/att_t_1_embeddings.pth",
                        help="Output torch file path")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP text encoder model name")
    parser.add_argument("--pattern", type=str, default="{category} is {attr}",
                        help="Prompt pattern, can use {category} and {attr}")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    txt_model = CLIPTextModelWithProjection.from_pretrained(args.model_name).eval().to(args.device)

    pairs = load_attributes(args.in_json)
    texts = [args.pattern.format(category=cat, attr=a) for cat, a in pairs]

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i+args.batch_size]
            enc = tokenizer(text=batch, return_tensors="pt", padding=True, truncation=True).to(args.device)
            out = txt_model(**enc)
            all_embs.append(out.text_embeds.float().cpu())

    emb = torch.cat(all_embs, dim=0) if all_embs else torch.empty(0, 512)

    torch.save({"att_text": texts,
                "att_category": [cat for cat,_ in pairs],
                "att_embedding": emb}, args.out_pth)
    print(f"{len(texts)} prompts -> {tuple(emb.shape)} saved to {args.out_pth}")




   






    

    



            


