# tools/gen_class_embeds.py
# in config file, embedding_path = 'data/VOC2007/SOWOD/t1_gt_embeddings.npy'
# in CLI, python tools/gen_class_embeds.py --task t1, --normalize

import os, torch, argparse, numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModelWithProjection

CLASS_NAMES = {
    "t1" : ["airplane","bicycle","bird","boat","bus","car",
            "cat","cow","dog","horse","motorcycle","sheep",
            "train","elephant","bear","zebra","giraffe","truck","person"],
    "t2" : ["traffic light","fire hydrant","stop sign","parking meter","bench","chair","dining table",
            "potted plant","backpack","umbrella","handbag","tie","suitcase","microwave","oven","toaster",
            "sink","refrigerator","bed","toilet","couch"],
    "t3" : ["frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake"],
    "t4" : ["laptop","mouse","remote","keyboard","cell phone","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush","wine glass","cup","fork","knife","spoon","bowl","tv","bottle"]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate class embeddings for a given task split")
    parser.add_argument("--task_name", type=str, choices=list(CLASS_NAMES.keys()), required=True,
                        help="Task split (t1, t2, t3, t4)")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP text encoder model name")
    parser.add_argument("--pattern", type=str, default="a photo of a {classname}",
                        help="Prompt pattern for class names")
    parser.add_argument("--normalize", action="store_true", help="L2 normalize embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    txt_model = CLIPTextModelWithProjection.from_pretrained(args.model_name).eval().to(args.device)

    prompts = [args.pattern.format(classname=c) for c in CLASS_NAMES[args.task_name]]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(args.device) for k, v in enc.items()}

    with torch.no_grad():
        embeds = txt_model(**enc).text_embeds  # (N, D)
        if args.normalize:
            embeds = F.normalize(embeds, dim=-1)
        arr = embeds.detach().cpu().numpy().astype("float32")

    out_path = f"data/VOC2007/SOWOD/{args.task_name}_gt_embeddings.npy"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)
    print(arr.shape, "-> saved to", out_path)

