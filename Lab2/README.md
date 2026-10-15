# Lab 2 — modelli linguistici

[Consegne originali](CONSEGNE.md). Tutti i modelli e dataset vengono scaricati solo durante l'esecuzione; la cache è in `data/`. Non serve un account di tracking.

| Consegna | Comando | Risultati |
|---|---|---|
| 1 GPT a caratteri su Dante | `python -m lab2.ex1` | Loss train/validation, checkpoint, testo generato |
| 2.1 Tokenizzazione GPT-2 | `python -m lab2.ex2_1` | Caratteri, parole, token, ID e decode |
| 2.2 Generazione GPT-2 | `python -m lab2.ex2_2` | Greedy e sampling a più temperature |
| 3.1 Classificazione irony | `python -m lab2.ex3_1 --task irony` | Accuracy e macro-F1 |
| 3.1 Classificazione emoji | `python -m lab2.ex3_1 --task emoji` | Accuracy e macro-F1 |
| 3.2 Multiple-choice ARC-Easy | `python -m lab2.ex3_2 --dataset arc` | Loss, accuracy e macro-F1 |
| 3.2 Multiple-choice SWAG | `python -m lab2.ex3_2 --dataset swag` | Loss, accuracy e macro-F1 |
| 3.3 Retrieval SciFact | `python -m lab2.ex3_3` | Recall@1/5/10, MRR, nDCG@10, esempi di ranking |

Gli esercizi sono indipendenti. Usa `--output-dir` per assegnare directory nuove a ogni esecuzione. `--checkpoint` negli esercizi di training carica il modello per generazione o valutazione finale, senza riprendere il training.

## Scelte delle soluzioni

**GPT Dante:** il corpus locale è `archive/Lab2/commedia.txt`; il preprocessing rimuove intestazione e footer Gutenberg da una copia. Il testo originale resta intatto. Il vocabolario di caratteri è condiviso; train e validation sono due parti contigue 90%/10%, senza finestre che attraversano il confine. Embedding da 192, sei head, quattro blocchi pre-LayerNorm, dropout 0.1 e contesto 128. Le epoche sono gruppi di `--steps-per-epoch` minibatch casuali, non passaggi esaustivi sul corpus. La validation usa finestre fisse ottenute con un generatore separato. `--corpus`, `--block-size`, `--width`, `--heads`, `--layers`, `--prompt`, `--generate` e `--temperature` sono configurabili; il checkpoint contiene anche il vocabolario.

**GPT-2:** `openai-community/gpt2` preaddestrato, senza fine-tuning. Il codice conserva attention mask e usa EOS come padding. Il limite di contesto è controllato prima della generazione. GPT-2 non è specializzato in Dante o in italiano: confronta qualitativamente ripetizioni, coerenza e temperatura, senza aspettarti lo stesso stile del modello addestrato sul corpus.

**TweetEval:** `cardiffnlp/tweet_eval`, configurazioni irony ed emoji, con split ufficiali. DistilBERT è congelato nella baseline e resta in modalità eval, così il dropout non altera casualmente le feature. La testa è addestrata con cross-entropy. `--fine-tune` sblocca il backbone; il learning rate predefinito passa da 1e-3 a 2e-5 per questa modalità. Usa due esecuzioni separate per il confronto. Il checkpoint migliore è scelto dalla loss di validation e il test è valutato solo al termine.

**Multiple-choice:** `google-bert/bert-base-uncased` sia per tokenizer sia per encoder. Un punteggio scalare per ciascuna coppia domanda/risposta, seguito da cross-entropy fra opzioni. ARC traduce answerKey cercandola nelle etichette fornite; le opzioni mancanti sono mascherate e non possono essere scelte. SWAG concatena sent2 alla possibile conclusione e usa sent1 come contesto. Per ARC vengono usati train/validation/test ufficiali; per SWAG il training è diviso 90%/10% e la validation ufficiale resta l'holdout finale, perché il test pubblico non contiene le risposte. L'encoder è fine-tuned di default; `--frozen` offre la baseline con sola testa.

**SciFact:** [corpus e query](https://huggingface.co/datasets/BeIR/scifact) e [qrels](https://huggingface.co/datasets/BeIR/scifact-qrels). Le query dei qrels train sono divise 90%/10%; tutti i positivi di una query restano nello stesso split. Encoder DistilBERT condiviso, media dei token con attention mask e normalizzazione L2. Loss contrastiva con positivi nel batch e un negativo esplicito per query; documenti rilevanti condivisi fra query non vengono usati come falsi negativi. La validation sceglie il checkpoint per nDCG@10; i qrels test restano separati. Il ranking considera **l'intero corpus**, con similarità a blocchi, senza librerie di indicizzazione aggiuntive. `splits.json`, `test_queries.csv` ed esempi top-5 rendono leggibile l'analisi. Le metriche usano rilevanza binaria, coerente con SciFact.

## Commento dei risultati

Per Dante e GPT-2 leggi i testi generati e commenta stile, coerenza e ripetizioni. Per classificazione confronta accuracy e macro-F1, soprattutto nello sbilanciamento delle classi. Per multiple-choice descrivi gli errori senza equiparare le scelte A/B/C/D a categorie semantiche. Per retrieval analizza sia le metriche sia i documenti recuperati. Le metriche delle esecuzioni reali e i loro limiti sono in [RISULTATI.md](../RISULTATI.md).
