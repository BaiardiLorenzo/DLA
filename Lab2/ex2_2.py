"""Confronto fra generazione greedy e sampling con GPT-2 preaddestrato."""
import torch

from common.runtime import parser, positive_int, setup, write_json


def main():
    p = parser(__doc__, training=False, checkpoint=False)
    p.add_argument("--model", default="openai-community/gpt2")
    p.add_argument("--prompt", default="Nel mezzo del cammin di nostra vita")
    p.add_argument("--max-new-tokens", type=positive_int, default=150)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0, 1.2])
    args = p.parse_args()
    if any(value <= 0 for value in args.temperatures) or not args.prompt:
        p.error("Il prompt non può essere vuoto e le temperature devono essere positive")
    device, out = setup(args, "lab2", "ex2_2")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    inputs = {key: value.to(device) for key, value in tokenizer(args.prompt, return_tensors="pt").items()}
    if inputs["input_ids"].shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
        p.error("Prompt e generazione superano il contesto massimo di GPT-2")
    results = {}
    with torch.no_grad():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        results["greedy"] = tokenizer.decode(generated[0], skip_special_tokens=True)
        for temperature in args.temperatures:
            generated = model.generate(**inputs, do_sample=True, temperature=temperature,
                                       max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            results[f"temperature_{temperature}"] = tokenizer.decode(generated[0], skip_special_tokens=True)
    write_json(out / "generation.json", results)
    (out / "generation.txt").write_text("\n\n".join(f"{key}\n{text}" for key, text in results.items()), encoding="utf-8")


if __name__ == "__main__":
    main()
