# Deep Learning Applications — soluzioni Python

Quattro piccoli progetti Python che risolvono **tutti gli esercizi**, comprese le alternative opzionali, dei notebook del corso. Le consegne originali sono in `lab*/CONSEGNE.md`; notebook, immagini, video e checkpoint storici sono conservati in `archive/`.

Il codice è stato riscritto e verificato con controlli funzionali, training ed evaluation reali. I risultati misurati e lo stato delle esecuzioni sono in [RISULTATI.md](RISULTATI.md); gli esperimenti ancora in corso sono indicati esplicitamente. La convergenza dipende dal modello e dall'ambiente e non viene dedotta dalla sola assenza di errori.

## Preparare Conda DLA su Windows

Da PowerShell, nella directory di questa repository:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
. .\activate.ps1
```

Lo script crea `DLA` se manca, oppure ne aggiorna Python e le dipendenze. Usa il Python di base soltanto per invocare Conda e non installa pacchetti in `base`. Non elimina pacchetti estranei già presenti in `DLA`. Per un'installazione Conda in un'altra posizione:

```powershell
.\setup.ps1 -CondaBasePython 'D:\Miniconda3\python.exe'
. .\activate.ps1 -CondaBasePython 'D:\Miniconda3\python.exe'
```

`activate.ps1` carica il modulo PowerShell ufficiale e attiva `DLA`, usando `python -m conda` anche quando il launcher `conda.exe` è inutilizzabile. Il primo punto nel comando è necessario per mantenere l'attivazione nella sessione corrente. Se Conda funziona già normalmente puoi usare direttamente `conda activate DLA`. Se PowerShell blocca gli script, abilitali soltanto nella sessione corrente con `Set-ExecutionPolicy -Scope Process Bypass`.

Python 3.12 è scelto per compatibilità. Le dipendenze dirette sono fissate alle release stabili disponibili al **14 ottobre 2026**: PyTorch 2.14.1, torchvision 0.29.1, Transformers 5.19.0, Datasets 5.1.0, Gymnasium 1.4.0, oltre alle librerie numeriche e grafiche. L'ambiente `DLA` è stato installato: attivazione, forward/backward CUDA e `pip check` sono riusciti.

La build PyTorch è **CUDA 13.0** per la RTX 5070 Ti rilevata sul PC. Il runtime CUDA è distribuito nei wheel; lo script non installa il toolkit o modifica il driver. Box2D 2.3.10 viene installato dal wheel precompilato: non servono compilazioni locali o SWIG. Riferimenti: [installazione PyTorch](https://pytorch.org/get-started/locally/), [indice CUDA 13.0](https://download.pytorch.org/whl/cu130/torch/), [Box2D](https://pypi.org/project/Box2D/), [Gymnasium](https://pypi.org/project/gymnasium/).

`requirements-gpu.txt` va installato prima di `requirements.txt`: `setup.ps1` gestisce l'ordine. Le versioni effettivamente risolte sono esportate in `outputs/environment/`. Le dipendenze indirette restano affidate al resolver; il freeze dell'installazione reale ne registra le versioni. Lo script usa solo conda-forge, senza richiedere i termini dei canali Anaconda.

## Eseguire gli esercizi

Lancia i comandi **dalla radice della repository**, con `DLA` attivo. `--help` descrive i parametri di ciascun modulo.

| Laboratorio | Attività | Istruzioni |
|---|---|---|
| 1 | MNIST/CIFAR-10, residuali, gradienti, FCN, Grad-CAM | [lab1/README.md](lab1/README.md) |
| 2 | GPT Dante, GPT-2, TweetEval, ARC/SWAG, SciFact | [lab2/README.md](lab2/README.md) |
| 3 | REINFORCE, baseline, DQN e PPO | [lab3/README.md](lab3/README.md) |
| 4 | OOD, ROC/PR, FGSM, adversarial training, ODIN, JARN | [lab4/README.md](lab4/README.md) |

Per iniziare:

```powershell
python -m lab1.ex1_1
```

I README indicano l'ordine e checkpoint riutilizzabili. Non c'è un comando che avvia automaticamente tutti i training: scegli esplicitamente gli esercizi, alcuni richiedono molte ore, soprattutto CarRacing e i confronti fra più modelli.

## Dati e risultati locali

- `common/`: seed, device, percorsi, CSV/JSON e grafici.
- `lab1/` … `lab4/`: implementazioni e comandi degli esercizi, senza dipendere dall'ordine delle celle.
- `data/`: dataset e cache Hugging Face; i download avvengono soltanto quando lanci un esercizio.
- `outputs/<lab>/<esercizio>/<esecuzione>/`: configurazione, metriche, grafici, checkpoint, testi e video.
- `archive/`: materiali storici, con risultati riferiti esclusivamente alle vecchie implementazioni.

Non viene usato W&B né TensorBoard. Il monitoraggio è codice locale: CSV per epoche/episodi, JSON per riepiloghi, PNG per grafici, `.pt` per pesi, TXT per generazione e MP4 per video RL opzionali. `data/`, `outputs/` e cache non entrano in Git.

CIFAR usa i file torchvision se già presenti; altrimenti scarica la copia pubblica `uoft-cs` da Hugging Face, dopo che il server originale è risultato non raggiungibile durante le esecuzioni. Le immagini vengono conservate anche come RGB uint8 in una cache NumPy per accelerare gli epoch, senza cambiare split o etichette.

`run_experiments.py` esegue le code di training/evaluation con log e registri persistenti. `--group cpu` esegue CartPole/LunarLander; `--group gpu` gli altri laboratori e CarRacing; `--group vision` permette di ripetere solo CNN, gradienti e Grad-CAM. Le code `--group racing_dqn` e `--group racing_ppo` permettono esecuzioni CarRacing indipendenti; PPO raccoglie transizioni da otto ambienti paralleli per default. Ogni job ha una directory distinta; quelli già completati sono saltati. Non avviare contemporaneamente gruppi che contengono lo stesso job ancora in corso. Sono previste esecuzioni estese, anche di diverse ore, e non il resume degli optimizer.

Ogni esecuzione ha un identificatore distinto. `--output-dir` permette di scegliere un percorso esplicito, che deve essere nuovo. `--checkpoint` carica i pesi per l'analisi prevista dal modulo; **non riprende** lo stato degli optimizer o del replay buffer. I checkpoint storici non sono compatibili automaticamente con i nuovi modelli.

Per completare soltanto i CarRacing rimasti interrotti, su Windows usa `powershell -ExecutionPolicy Bypass -File .\finish_remaining.ps1`. Il processo locale continua dai pesi salvati e dai relativi contatori, salta i job completati, esegue le evaluation finali e aggiorna automaticamente `RISULTATI.md` ogni due minuti e a ogni conclusione. Non richiede Codex aperto né chiamate a modelli; log e stato sono in `outputs/experiments_20261014/background.log` e `background_state.json`. Lascia il PC acceso; il processo impedisce la sospensione automatica durante il lavoro. Il launcher evita duplicati e può essere rilanciato se il PC viene spento.

`--continue-run <directory_run>` per DQN/PPO usa `recovery.pt`, oppure il precedente `best.pt`, e conserva la cronologia fino al checkpoint caricato. I budget `--episodes` e `--total-steps` sono totali comprensivi del prefisso recuperato. Optimizer, replay e simulatori vengono reinizializzati: è una continuazione dai pesi, non una ripresa esatta. I nuovi run salvano anche `recovery.pt` durante il training.

`--device auto` sceglie CUDA se disponibile, altrimenti CPU; `--device cuda` segnala l'indisponibilità della GPU. Il seed rende gli split e le inizializzazioni riproducibili, ma non promette identità numerica fra hardware e versioni diversi. Su Windows i DataLoader usano `num_workers=0` per evitare dipendenze dal multiprocessing.

## Errori originali corretti

- GPT incompleto sostituito da un transformer causale con obiettivo next-token.
- Separazione train/validation/test, senza usare il test per scegliere il modello o gli iperparametri ODIN.
- Tokenizer e backbone coerenti; risposte multiple ARC mascherate quando mancanti.
- Retrieval implementato con training e ranking sul corpus completo.
- FCN ottenuta convertendo i pesi dei layer densi, anziché addestrare una segmentazione diversa dalla consegna.
- Grad-CAM calcolato dal logit della classe e non dalla cross-entropy.
- RL con valutazione separata, checkpoint del valore corretti e bootstrap distinto per terminali e TimeLimit.
- ODIN senza etichette reali, con score continui e perturbazione che aumenta la confidenza.
- FGSM nello spazio dei pixel, con clipping valido; JARN con Jacobiani differenziabili e obiettivo minimax separato.

Per le consegne che chiedono di spiegare i risultati, i README indicano quali grafici e metriche interpretare. I risultati misurati e i limiti delle esecuzioni sono in [RISULTATI.md](RISULTATI.md); `python report_experiments.py` aggiorna le tabelle dai file prodotti. `run_experiments.py --group nlp_extra` completa il confronto con fine-tuning per emoji.

## Versionare la riorganizzazione su Windows

Le nuove cartelle `lab1`–`lab4` hanno nomi minuscoli. Per registrare anche il cambio di maiuscole nei README precedentemente tracciati su Windows, quando vorrai salvare le modifiche in Git usa `git -c core.ignorecase=false add -A`. Nessun commit è stato creato durante la riscrittura.
