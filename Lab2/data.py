import re

import torch


def clean_dante(path):
    text = path.read_text(encoding="utf-8-sig")
    start = re.search(r"^Inferno:\s*Canto I\s*$", text, flags=re.MULTILINE)
    if not start:
        raise ValueError("Intestazione 'Inferno: Canto I' non trovata nel corpus")
    text = text[start.start():]
    footer = re.search(r"(?im)^\s*(?:\*\*\*\s*)?(?:end of (?:the )?project gutenberg|end of the project gutenberg|end of this project gutenberg)", text)
    return text[:footer.start()].strip() if footer else text.strip()


def character_batch(tokens, batch_size, block_size, generator):
    if len(tokens) <= block_size:
        raise ValueError("Split troppo breve per il contesto richiesto")
    starts = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
    x = torch.stack([tokens[i:i + block_size] for i in starts])
    y = torch.stack([tokens[i + 1:i + block_size + 1] for i in starts])
    return x, y


class TextCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, examples):
        batch = self.tokenizer([item["text"] for item in examples], truncation=True, padding=True,
                               max_length=self.max_length, return_tensors="pt")
        # DistilBERT non usa token_type_ids.
        batch = {key: value for key, value in batch.items() if key in ("input_ids", "attention_mask")}
        batch["labels"] = torch.tensor([item["label"] for item in examples])
        return batch


def multiple_choice_example(item, dataset):
    if dataset == "arc":
        labels = item["choices"]["label"]
        answer = str(item["answerKey"])
        if answer not in labels:
            raise ValueError(f"answerKey {answer} non presente nelle scelte per {item['id']}")
        return item["question"], item["choices"]["text"], labels.index(answer)
    return item["sent1"], [item["sent2"] + " " + item[f"ending{i}"] for i in range(4)], int(item["label"])


class ChoiceCollator:
    def __init__(self, tokenizer, dataset, max_length):
        self.tokenizer, self.dataset, self.max_length = tokenizer, dataset, max_length

    def __call__(self, examples):
        processed = [multiple_choice_example(item, self.dataset) for item in examples]
        choices = max(len(item[1]) for item in processed)
        questions, answers, masks, labels = [], [], [], []
        for question, options, label in processed:
            questions.extend([question] * choices)
            answers.extend(options + [""] * (choices - len(options)))
            masks.append([True] * len(options) + [False] * (choices - len(options)))
            labels.append(label)
        encoded = self.tokenizer(questions, answers, truncation=True, padding=True,
                                 max_length=self.max_length, return_tensors="pt")
        batch = {key: value.reshape(len(examples), choices, -1) for key, value in encoded.items()}
        batch["choice_mask"] = torch.tensor(masks, dtype=torch.bool)
        batch["labels"] = torch.tensor(labels)
        return batch
