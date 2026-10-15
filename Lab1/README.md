# Lab 1 — reti per immagini

[Consegne originali](CONSEGNE.md). Tutte le attività sono implementate. Le esecuzioni reali sono documentate in [RISULTATI.md](../RISULTATI.md); puoi ripeterle dalla radice, in `DLA`.

| Consegna | Comando | Output principale |
|---|---|---|
| 1.1 MLP baseline e confronto di profondità | `python -m lab1.ex1_1` | Loss/accuracy MNIST per profondità e confronto |
| 1.2 CNN plain/residuali di profondità crescente | `python -m lab1.ex1_2` | CIFAR-10, confronto per profondità e skip |
| 2.1 Quantificare il beneficio delle residuali | `python -m lab1.ex2_1` | CSV gradienti per strato/epoca, grafici e curve |
| 2.2 Fully-convolutionalizzare un classificatore | `python -m lab1.ex2_2` | Classificatore MNIST, FCN, canvas e localizzazioni |
| 2.3 Spiegare le predizioni tramite Grad-CAM | `python -m lab1.ex2_3 --checkpoint outputs/lab1/cnn/residual_2/best.pt` | Immagini e heatmap per classe |

## Ordine suggerito

```powershell
python -m lab1.ex1_1 --output-dir outputs/lab1/mlp
python -m lab1.ex1_2 --output-dir outputs/lab1/cnn
python -m lab1.ex2_1 --output-dir outputs/lab1/gradients
python -m lab1.ex2_2 --output-dir outputs/lab1/fcn
python -m lab1.ex2_3 --checkpoint outputs/lab1/cnn/residual_2/best.pt --output-dir outputs/lab1/gradcam
```

Per ripetere i comandi usa nuove directory di output. La FCN è autonoma: addestra un classificatore MNIST prima della conversione. Per riutilizzare quel classificatore: `python -m lab1.ex2_2 --checkpoint outputs/lab1/fcn/classifier/best.pt`.

## Metodo e interpretazione

MNIST e CIFAR-10 mantengono il test ufficiale separato; il training ufficiale è diviso 90%/10% per training e validation. I confronti riportano training e validation, senza usare il test per selezionare architetture. Stesso seed, split, batch e optimizer per ciascun modello.

L'MLP usa width 64 e profondità 2/4/8/16. Le CNN hanno tre stadi da 32/64/128 canali e 1/2/4 blocchi per stadio; ogni blocco ha due convoluzioni. Versioni plain e residuali hanno gli stessi pesi e cambiano soltanto la somma dello skip. I CSV riportano anche il numero di convoluzioni. Puoi cambiare `--depths`, `--width`, `--blocks`, `--epochs`, `--batch-size` e `--lr`.

L'esercizio 2.1 esegue un confronto dedicato e raccoglie la media del gradiente assoluto delle matrici dei pesi, dopo backward e prima dell'optimizer, mediata sui minibatch di ogni epoca. `gradients.csv` identifica ogni strato; i grafici mostrano prima e ultima epoca su scala logaritmica. Confronta questi valori insieme a loss e accuracy: un gradiente maggiore non dimostra da solo un modello migliore.

La FCN riutilizza le convoluzioni e converte i due Linear in Conv2d 5×5 e 1×1, copiando pesi e bias. Su una patch 28×28 restituisce la stessa classificazione del modello originale; su immagini grandi produce una griglia con stride 4 e campo ricettivo 26. `class_maps.pt` salva griglia, canvas e mappe interpolate alla risoluzione di ogni pixel; l'interpolazione non aggiunge una predizione indipendente per pixel. I bounding box sono finestre di classificazione, con filtro di completezza del tratto, centratura del baricentro e non-maximum suppression. Questi filtri usano solo l'immagine, non le posizioni reali delle cifre. Non è una segmentazione supervisionata e non ha una classe background appresa: cifre vicine o molto decentrate possono limitarla. `detections.json` conserva ground truth, predizioni e precision/recall a IoU 0.5 sul singolo canvas sintetico, da non confondere con un benchmark di detection completo. `--confidence` controlla il filtro; `--digits` il numero di cifre. Il generatore del canvas ha un seed separato dal training; `--canvas-run` permette di riusare esattamente input e ground truth di un'esecuzione precedente.

Per valutare i classificatori sul test ufficiale: `python -m lab1.evaluate --checkpoint outputs/lab1/cnn`. Una directory valuta tutti i `best.pt` contenuti; un file valuta solo quel modello.

Grad-CAM usa il logit della classe predetta; `--class-id` permette di studiarne un'altra. Il gradiente viene mediato sulle dimensioni spaziali, le feature pesate vengono sommate e passate attraverso ReLU. Le mappe nulle restano nulle; gli hook sono rimossi anche in caso di errore.

## Commento dei risultati

Dopo l'esecuzione confronta loss di training, accuracy di validation e gradienti fra profondità e skip. Descrivi i casi in cui reti più profonde degradano o migliorano senza imporre una conclusione ai dati. Per FCN e Grad-CAM commenta gli esempi corretti e gli errori visibili. I risultati reali della riscrittura sono in [RISULTATI.md](../RISULTATI.md); i risultati archiviati restano distinti.
