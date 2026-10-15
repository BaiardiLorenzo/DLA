"""Esplorazione della tokenizzazione subword di GPT-2."""
from common.runtime import parser, setup, write_json


def main():
    p = parser(__doc__, training=False, checkpoint=False)
    p.add_argument("--model", default="openai-community/gpt2")
    p.add_argument("--text", default="Nel mezzo del cammin di nostra vita")
    args = p.parse_args()
    _, out = setup(args, "lab2", "ex2_1")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = tokenizer(args.text, return_tensors="pt")["input_ids"][0].tolist()
    result = dict(text=args.text, characters=len(args.text), words=len(args.text.split()),
                  tokens=len(encoded), ids=encoded, pieces=tokenizer.convert_ids_to_tokens(encoded),
                  decoded=tokenizer.decode(encoded))
    write_json(out / "tokenization.json", result)
    print(result)


if __name__ == "__main__":
    main()
