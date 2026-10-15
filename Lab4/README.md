# Lab 4 — OOD e adversarial learning

[Consegne originali](CONSEGNE.md). Input in [0,1], normalizzazione CIFAR interna al modello e analisi salvata in CSV/JSON/PNG. Nessun checkpoint dei notebook è richiesto.

| Consegna | Comando |
|---|---|
| Attività introduttiva, rumore/immagini reali ed epoche | `python -m lab4.intro` |
| 1.1 Pipeline OOD e istogrammi | `python -m lab4.ex1_1` |
| 1.2 AUROC e curve PR ID/OOD | `python -m lab4.ex1_2 --checkpoint outputs/lab4/base/training/best.pt` |
| 2.1 FGSM ed effetto di epsilon | `python -m lab4.ex2_1 --checkpoint outputs/lab4/base/training/best.pt` |
| 2.2 Adversarial training online | `python -m lab4.ex2_2 --checkpoint outputs/lab4/base/training/best.pt` |
| 3.1 ODIN | `python -m lab4.ex3_1 --checkpoint outputs/lab4/base/training/best.pt` |
| 3.2 JARN | `python -m lab4.ex3_2 --checkpoint outputs/lab4/base/training/best.pt` |
| 3.3 FGSM mirato | `python -m lab4.ex3_3 --checkpoint outputs/lab4/base/training/best.pt --target 0` |

## Ordine suggerito

```powershell
python -m lab4.intro --output-dir outputs/lab4/intro
python -m lab4.ex1_1 --output-dir outputs/lab4/base
python -m lab4.ex1_2 --checkpoint outputs/lab4/base/training/best.pt
python -m lab4.ex2_1 --checkpoint outputs/lab4/base/training/best.pt
python -m lab4.ex2_2 --checkpoint outputs/lab4/base/training/best.pt
python -m lab4.ex3_1 --checkpoint outputs/lab4/base/training/best.pt
python -m lab4.ex3_2 --checkpoint outputs/lab4/base/training/best.pt
python -m lab4.ex3_3 --checkpoint outputs/lab4/base/training/best.pt --target 0
```

Intro ed esercizio 1.1 sono autonomi e addestrano il modello base; per evitare un secondo training puoi passare `--checkpoint outputs/lab4/intro/training/best.pt` all'esercizio 1.1 e usare quel checkpoint nei comandi successivi. L'esercizio 1.2 riusa la stessa pipeline e produce le metriche threshold-free senza training se riceve un checkpoint.

## Dati, scoring e split

CIFAR-10 è ID. Il training ufficiale è diviso 90%/10%; il test ufficiale è separato. La validation ID serve a scegliere i checkpoint e i parametri ODIN. `--ood cifar100` usa una lista esplicita di classi distinte dalle dieci categorie CIFAR-10, definita in `data.py`; train CIFAR-100 per validation OOD e test CIFAR-100 per test OOD. `--ood fake` usa due insiemi FakeData con offset diversi e non sovrapposti. `--ood-size` limita solo OOD, non i dati di training ID. L'intro studia entrambe le sorgenti e salva checkpoint alle `--snapshot-epochs`, limitate al numero di epoche eseguito.

Max logit e maximum softmax probability sono **score ID**: maggiore significa più in-distribution. Lo score OOD è il negativo. I riepiloghi distinguono AUROC ID, area trapezoidale sotto PR per ID/OOD e average precision, che non è la stessa integrazione. Le AUPR dipendono dalla proporzione ID/OOD: i JSON registrano anche i conteggi. Gli score restano continui; non sono soglie binarie usate impropriamente come probabilità.

Le classi CIFAR-100 scelte sono nominalmente distinte; questo non prova una separazione semantica perfetta, né garantisce che il modello sia poco confidente su OOD. Confronta istogrammi e curve e descrivi i limiti dello scoring.

## Attacchi e training robusto

FGSM non mirato aggiunge epsilon·sign(gradiente della loss); FGSM mirato lo sottrae usando la loss della classe bersaglio. Epsilon è sempre nello spazio pixel [0,1], prima della normalizzazione, e il risultato è clippato a [0,1]. `--epsilons` controlla gli attacchi e `--examples` il numero di immagini salvate. Il successo è misurato solo sugli esempi originariamente corretti; per il mirato sono esclusi quelli già appartenenti al target. Se non ci sono esempi eleggibili il tasso resta non definito. Sono salvati anche accuratezza clean, accuratezza sotto attacco e massima perturbazione effettiva.

L'adversarial training parte dal checkpoint clean e genera FGSM a ogni minibatch con il modello corrente. La loss combina 50% clean e 50% avversario; `--train-epsilon` è indipendente dalla griglia di valutazione. JARN parte dallo stesso tipo di checkpoint e allena classificatore, adattatore e discriminatore. Questi confronti usano come baseline il checkpoint iniziale: il modello robusto ha ricevuto training aggiuntivo; non attribuire ogni differenza esclusivamente alla regolarizzazione senza un controllo clean con durata equivalente.

Per produrre il controllo equivalente usa `python -m lab4.ex2_2 --checkpoint <checkpoint_clean> --mode clean --epochs <stesso_budget>`. `run_experiments.py --group ood_extra` esegue i controlli di cinque e dieci epoche degli esperimenti documentati. Il report confronta clean, FGSM e OOD anche a uguale budget aggiuntivo.

**ODIN:** le pseudo-etichette sono quelle predette, mai quelle del dataset OOD. Cross-entropy sui logit divisi per T; perturbazione sottratta per aumentare la confidenza. Griglia T/epsilon su validation ID e OOD, scelta per AUROC ID e valutazione dei parametri selezionati sul test. Il confronto MSP usa lo stesso test. [Paper ODIN](https://arxiv.org/abs/1706.02690).

**JARN:** l'attività 2.2 richiede adversarial augmentation; l'affermazione nella consegna 3.2 che questo sia già Jacobian regularization è un'incongruenza. Qui JARN implementa il metodo del [paper](https://arxiv.org/html/1912.10185v2): gradiente per esempio della cross-entropy rispetto all'input, adattatore Conv1×1+tanh, discriminatore immagini reali/Jacobiani adattati, obiettivo minimax e gradienti di ordine superiore (`create_graph=True`). Il discriminatore viene aggiornato con Jacobiani detached; classificatore e adattatore vengono aggiornati attraverso il Jacobiano, con il discriminatore congelato. Immagini e adattatore sono confrontati nello stesso dominio [-1,1]. `--jarn-weight` controlla il coefficiente e `--noise-epsilon` il rumore uniforme durante il training. È più costoso del training clean e i risultati richiedono misurazione, senza garanzie di robustezza.

## Commento dei risultati

Commenta separatamente accuratezza ID clean, accuratezza sotto attacco, successo mirato e qualità OOD. Confronta epsilon, tipo di OOD ed epoca, considerando anche il degrado sulle immagini normali. Le figure mostrano esempi e curve ma non sostituiscono il riepilogo numerico. I risultati reali e lo stato dei training sono in [RISULTATI.md](../RISULTATI.md).
