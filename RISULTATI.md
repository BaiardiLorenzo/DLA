# Training e valutazione — risultati reali

Aggiornamento locale: 2026-10-15T21:19:33.

Esperimenti completati: **46/46**. Le righe in corso o in attesa non sono risultati completati.

Ambiente conda `DLA`, Python 3.12, RTX 5070 Ti 16 GB, PyTorch CUDA 13.0. Gli agenti per osservazioni vettoriali usano CPU con un thread; reti per immagini/linguaggio e CarRacing usano CUDA. Seed 42; validation distinta dal test. Un solo seed non misura la variabilità fra training.

CIFAR-10/100: copia pubblica `uoft-cs` su Hugging Face, stessa suddivisione 50.000/10.000, usata dopo timeout del server originale. Cache RGB uint8 per evitare decodifiche ripetute.

## Stato delle esecuzioni

| Esperimento | Stato | Secondi | Log e risultati |
|---|---|---:|---|
| [cartpole_reinforce](outputs/experiments_20261014/cartpole_reinforce/20261014_212421/run.log) | completed | 7.15 | outputs/experiments_20261014/cartpole_reinforce/20261014_212421 |
| [cartpole_baselines](outputs/experiments_20261014/cartpole_baselines/20261014_212428/run.log) | completed | 143.6 | outputs/experiments_20261014/cartpole_baselines/20261014_212428 |
| [cartpole_value_eval](outputs/experiments_20261014/cartpole_value_eval/20261014_212652/run.log) | completed | 8.84 | outputs/experiments_20261014/cartpole_value_eval/20261014_212652 |
| [lunar_reinforce](outputs/experiments_20261014/lunar_reinforce/20261014_212700/run.log) | completed | 637.72 | outputs/experiments_20261014/lunar_reinforce/20261014_212700 |
| [lunar_value_eval](outputs/experiments_20261014/lunar_value_eval/20261014_213738/run.log) | completed | 24.12 | outputs/experiments_20261014/lunar_value_eval/20261014_213738 |
| [cartpole_dqn](outputs/experiments_20261014/cartpole_dqn/20261014_213802/run.log) | completed | 35.23 | outputs/experiments_20261014/cartpole_dqn/20261014_213802 |
| [cartpole_dqn_eval](outputs/experiments_20261014/cartpole_dqn_eval/20261014_213837/run.log) | completed | 5.79 | outputs/experiments_20261014/cartpole_dqn_eval/20261014_213837 |
| [lunar_dqn](outputs/experiments_20261014/lunar_dqn/20261014_213843/run.log) | completed | 163.55 | outputs/experiments_20261014/lunar_dqn/20261014_213843 |
| [lunar_dqn_eval](outputs/experiments_20261014/lunar_dqn_eval/20261014_214127/run.log) | completed | 28.16 | outputs/experiments_20261014/lunar_dqn_eval/20261014_214127 |
| [mlp](outputs/experiments_20261014/mlp/20261014_212409/run.log) | completed | 587.56 | outputs/experiments_20261014/mlp/20261014_212409 |
| [mlp_eval](outputs/experiments_20261014/mlp_eval/20261014_213356/run.log) | completed | 9.16 | outputs/experiments_20261014/mlp_eval/20261014_213356 |
| [cnn](outputs/experiments_20261014/cnn/20261014_214116/run.log) | completed | 2220.12 | outputs/experiments_20261014/cnn/20261014_214116 |
| [cnn_eval](outputs/experiments_20261014/cnn_eval/20261014_221816/run.log) | completed | 30.49 | outputs/experiments_20261014/cnn_eval/20261014_221816 |
| [gradients](outputs/experiments_20261014/gradients/20261014_221847/run.log) | completed | 386.91 | outputs/experiments_20261014/gradients/20261014_221847 |
| [gradients_eval](outputs/experiments_20261014/gradients_eval/20261014_222514/run.log) | completed | 21.47 | outputs/experiments_20261014/gradients_eval/20261014_222514 |
| [fcn](outputs/experiments_20261014/fcn/20261014_213455/run.log) | completed | 51.33 | outputs/experiments_20261014/fcn/20261014_213455 |
| [fcn_eval](outputs/experiments_20261014/fcn_eval/20261014_213546/run.log) | completed | 4.65 | outputs/experiments_20261014/fcn_eval/20261014_213546 |
| [gradcam](outputs/experiments_20261014/gradcam/20261014_222535/run.log) | completed | 10.49 | outputs/experiments_20261014/gradcam/20261014_222535 |
| [dante](outputs/experiments_20261014/dante/20261014_213551/run.log) | completed | 53.02 | outputs/experiments_20261014/dante/20261014_213551 |
| [gpt2_tokens](outputs/experiments_20261014/gpt2_tokens/20261014_213644/run.log) | completed | 12.65 | outputs/experiments_20261014/gpt2_tokens/20261014_213644 |
| [gpt2_generation](outputs/experiments_20261014/gpt2_generation/20261014_213657/run.log) | completed | 96.02 | outputs/experiments_20261014/gpt2_generation/20261014_213657 |
| [irony](outputs/experiments_20261014/irony/20261014_213833/run.log) | completed | 72.06 | outputs/experiments_20261014/irony/20261014_213833 |
| [irony_finetune](outputs/experiments_20261014/irony_finetune/20261014_213945/run.log) | completed | 32.71 | outputs/experiments_20261014/irony_finetune/20261014_213945 |
| [emoji](outputs/experiments_20261014/emoji/20261014_214017/run.log) | completed | 111.67 | outputs/experiments_20261014/emoji/20261014_214017 |
| [arc](outputs/experiments_20261014/arc/20261014_214209/run.log) | completed | 183.26 | outputs/experiments_20261014/arc/20261014_214209 |
| [swag](outputs/experiments_20261014/swag/20261014_214512/run.log) | completed | 3266.9 | outputs/experiments_20261014/swag/20261014_214512 |
| [scifact](outputs/experiments_20261014/scifact/20261014_223939/run.log) | completed | 116.75 | outputs/experiments_20261014/scifact/20261014_223939 |
| [ood_intro](outputs/experiments_20261014/ood_intro/20261014_224136/run.log) | completed | 65.87 | outputs/experiments_20261014/ood_intro/20261014_224136 |
| [ood_pipeline](outputs/experiments_20261014/ood_pipeline/20261014_224242/run.log) | completed | 13.17 | outputs/experiments_20261014/ood_pipeline/20261014_224242 |
| [ood_metrics](outputs/experiments_20261014/ood_metrics/20261014_224255/run.log) | completed | 17.48 | outputs/experiments_20261014/ood_metrics/20261014_224255 |
| [fgsm](outputs/experiments_20261014/fgsm/20261014_224313/run.log) | completed | 13.67 | outputs/experiments_20261014/fgsm/20261014_224313 |
| [adversarial](outputs/experiments_20261014/adversarial/20261014_224326/run.log) | completed | 94.19 | outputs/experiments_20261014/adversarial/20261014_224326 |
| [odin](outputs/experiments_20261014/odin/20261014_224500/run.log) | completed | 26.94 | outputs/experiments_20261014/odin/20261014_224500 |
| [jarn](outputs/experiments_20261014/jarn/20261014_224527/run.log) | completed | 87.45 | outputs/experiments_20261014/jarn/20261014_224527 |
| [targeted](outputs/experiments_20261014/targeted/20261014_224655/run.log) | completed | 18.59 | outputs/experiments_20261014/targeted/20261014_224655 |
| [racing_dqn](outputs/experiments_20261014/racing_dqn/20261015_202834_b49887/run.log) | completed | 2548.61 | outputs/experiments_20261014/racing_dqn/20261015_202834_b49887 |
| [racing_dqn_eval](outputs/experiments_20261014/racing_dqn_eval/20261015_211105_d4a123/run.log) | completed | 157.3 | outputs/experiments_20261014/racing_dqn_eval/20261015_211105_d4a123 |
| [racing_ppo](outputs/experiments_20261014/racing_ppo/20261015_202834_616eb0/run.log) | completed | 2950.31 | outputs/experiments_20261014/racing_ppo/20261015_202834_616eb0 |
| [racing_ppo_eval](outputs/experiments_20261014/racing_ppo_eval/20261015_211747_f62d7f/run.log) | completed | 105.22 | outputs/experiments_20261014/racing_ppo_eval/20261015_211747_f62d7f |
| [cartpole_none_eval](outputs/experiments_20261014/cartpole_none_eval/run.log) | completed | 3.82 | outputs/experiments_20261014/cartpole_none_eval |
| [cartpole_standardized_eval](outputs/experiments_20261014/cartpole_standardized_eval/run.log) | completed | 4.9 | outputs/experiments_20261014/cartpole_standardized_eval |
| [lunar_none_eval](outputs/experiments_20261014/lunar_none_eval/run.log) | completed | 6.52 | outputs/experiments_20261014/lunar_none_eval |
| [lunar_standardized_eval](outputs/experiments_20261014/lunar_standardized_eval/run.log) | completed | 9.07 | outputs/experiments_20261014/lunar_standardized_eval |
| [emoji_finetune](outputs/experiments_20261014/emoji_finetune/20261014_224313/run.log) | completed | 260.97 | outputs/experiments_20261014/emoji_finetune/20261014_224313 |
| [clean_control_10](outputs/experiments_20261014/clean_control_10/20261014_231219/run.log) | completed | 55.65 | outputs/experiments_20261014/clean_control_10/20261014_231219 |
| [clean_control_5](outputs/experiments_20261014/clean_control_5/20261014_231315/run.log) | completed | 45.01 | outputs/experiments_20261014/clean_control_5/20261014_231315 |

Coordinatore locale in background: **completed**, PID 27476. Non usa Codex o API di modelli per seguire i training. [Log](outputs/experiments_20261014/background.log), [stato](outputs/experiments_20261014/background_state.json). Il report viene aggiornato ogni due minuti e alla conclusione dei job.

| CarRacing | Episodio | Transizioni | Update |
|---|---:|---:|---:|
| racing_dqn | 500 | 124942 |  |
| racing_ppo | 4108 | 1000000 | 977 |

racing_dqn: continuazione dai pesi salvati all’episodio 250, transizione 62500. Optimizer, replay e simulatori reinizializzati; non è una ripresa identica dello stato interrotto. La cronologia include il prefisso fino al checkpoint caricato.


racing_ppo: continuazione dai pesi salvati all’episodio 2119, transizione 512000. Optimizer, replay e simulatori reinizializzati; non è una ripresa identica dello stato interrotto. La cronologia include il prefisso fino al checkpoint caricato.


## Classificazione immagini: test ufficiale

| Modello | Esempi test | Accuracy | Cross-entropy |
|---|---:|---:|---:|
| mlp_eval/depth_16 | 10000 | 95.89% | 0.1745 |
| mlp_eval/depth_2 | 10000 | 97.71% | 0.0782 |
| mlp_eval/depth_4 | 10000 | 97.18% | 0.0958 |
| mlp_eval/depth_8 | 10000 | 97.12% | 0.1119 |
| cnn_eval/plain_1 | 10000 | 80.60% | 0.5606 |
| cnn_eval/plain_2 | 10000 | 83.47% | 0.6196 |
| cnn_eval/plain_4 | 10000 | 78.65% | 0.6573 |
| cnn_eval/residual_1 | 10000 | 80.69% | 0.5874 |
| cnn_eval/residual_2 | 10000 | 81.30% | 0.5887 |
| cnn_eval/residual_4 | 10000 | 80.86% | 0.6197 |
| gradients_eval/plain_2 | 10000 | 74.89% | 0.7469 |
| gradients_eval/plain_4 | 10000 | 64.01% | 1.0396 |
| gradients_eval/residual_2 | 10000 | 77.58% | 0.6378 |
| gradients_eval/residual_4 | 10000 | 78.56% | 0.6611 |
| fcn_eval/classifier | 10000 | 99.04% | 0.0303 |

Magnitudini dei gradienti: media di |gradiente| sui pesi convoluzionali e sui batch dell’epoca. Il CSV completo e i grafici contengono tutti gli strati; qui si mostrano prima e ultima convoluzione.

| Modello | Epoca | Prima convoluzione | Ultima convoluzione |
|---|---:|---:|---:|
| plain_2 | 1 | 0.0299504 | 0.000525312 |
| plain_2 | 5 | 0.0276021 | 0.000225791 |
| plain_4 | 1 | 0.0238009 | 0.000313856 |
| plain_4 | 5 | 0.0311998 | 0.000171008 |
| residual_2 | 1 | 0.0333708 | 0.000587483 |
| residual_2 | 5 | 0.0294801 | 0.000203676 |
| residual_4 | 1 | 0.0279445 | 0.000321853 |
| residual_4 | 5 | 0.0273554 | 6.5394e-05 |

Le skip connection aggiungono un percorso identità al gradiente. Le magnitudini osservate dipendono anche da normalizzazione e scala dei pesi: una magnitudine maggiore da sola non dimostra migliore apprendimento. Leggere queste misure insieme alle curve di loss e accuracy e al confronto plain/residual a profondità equivalente.

Nel confronto di cinque epoche, la configurazione profonda passa da 64,01% (plain_4) a 78,56% (residual_4) sul test: 14,55 punti percentuali. Dopo venti epoche il divario delle stesse profondità è più piccolo, 78,65% contro 80,86%. Questo è coerente con un apprendimento iniziale più efficace per le skip connection nel modello profondo, senza un vantaggio uniforme a ogni profondità.

[Griglia Grad-CAM](outputs/experiments_20261014/gradcam_reviewed/gradcam.png): la mappa usa il logit della classe predetta. Gli esempi mostrano regioni utili al punteggio, anche quando la classe è errata; non sono una segmentazione né una prova causale. Per esempio, l’aereo classificato come nave concentra parte della risposta nella zona superiore dell’immagine.


FCN sul canvas iniziale di nove cifre: 9 detections, precision 100.00%, recall 100.00% a IoU 0.5, dopo il filtro di completezza/centratura. Prima del filtro erano 29 detections. Stessi pesi e stesso canvas; un singolo esempio sintetico non misura la generalizzazione della detection.

## Modelli linguistici

| Esperimento | Accuracy | Macro-F1 | Loss |
|---|---:|---:|---:|
| irony | 63.14% | 0.6266 | 0.6448 |
| irony_finetune | 63.78% | 0.6378 | 0.6543 |
| emoji | 29.32% | 0.1363 | 2.3514 |
| emoji_finetune | 36.06% | 0.2277 | 2.1272 |
| arc | 55.72% | 0.4456 | 1.1028 |
| swag | 78.53% | 0.7853 | 0.5559 |

SciFact sul corpus completo: recall_at_1=0.5222, recall_at_5=0.7418, recall_at_10=0.7982, mrr=0.6443, ndcg_at_10=0.6733.

GPT Dante, ultima epoca: train loss 1.3711, validation loss 1.5367. [Testo generato](outputs/experiments_20261014/dante/20261014_213551/generated.txt). Il testo riproduce punteggiatura e struttura dei versi, ma contiene parole inventate e incoerenze: non è un modello semantico di Dante.

## Agenti RL: valutazione su seed separati

| Agente | Episodi evaluation | Reward medio | Deviazione standard | Lunghezza media |
|---|---:|---:|---:|---:|
| cartpole_none_eval | 100 | 32.0500 | 4.7610 | 32.0500 |
| cartpole_standardized_eval | 100 | 500.0000 | 0.0000 | 500.0000 |
| cartpole_value_eval | 100 | 500.0000 | 0.0000 | 500.0000 |
| cartpole_dqn_eval | 100 | 468.4700 | 31.4240 | 468.4700 |
| lunar_none_eval | 100 | 72.2063 | 114.6922 | 377.0900 |
| lunar_standardized_eval | 100 | 153.2919 | 59.6362 | 789.1700 |
| lunar_value_eval | 100 | 178.3941 | 72.7739 | 496.5800 |
| lunar_dqn_eval | 100 | 213.8758 | 64.5161 | 489.4000 |
| racing_dqn_eval | 20 | 531.7955 | 140.3671 | 250.0000 |
| racing_ppo_eval | 20 | 870.7559 | 22.7359 | 249.6000 |

Completare un training non equivale a risolvere un ambiente: confrontare le ricompense con il criterio della consegna. Il checkpoint migliore è scelto sulla valutazione periodica, mentre questa tabella usa seed aggiuntivi.

## OOD e attacchi

Gli score maggiori indicano ID; per OOD si usa lo score negato. AUPR indica l’area trapezoidale della curva precision-recall. Il riferimento casuale per AUPR dipende dalla proporzione della classe positiva: 10.000 ID/2.800 CIFAR-100 corrispondono a 0,7813 per ID e 0,2188 per OOD; 10.000 ID/5.000 FakeData a 0,6667 e 0,3333. I valori fra sorgenti diverse non sono direttamente confrontabili.

| Esperimento / score | N ID / OOD | AUROC ID | AUPR ID | AUPR OOD |
|---|---:|---:|---:|---:|
| ood_pipeline/max_logit | 10000/2800 | 0.7509 | 0.9138 | 0.4222 |
| ood_pipeline/msp | 10000/2800 | 0.7022 | 0.8965 | 0.3560 |
| ood_metrics/max_logit | 10000/5000 | 0.2409 | 0.5968 | 0.2244 |
| ood_metrics/msp | 10000/5000 | 0.1850 | 0.5234 | 0.2110 |
| odin/test | 10000/2800 | 0.7461 | 0.9108 | 0.4176 |
| adversarial/baseline | 10000/2800 | 0.7022 | 0.8965 | 0.3560 |
| adversarial/adversarial | 10000/2800 | 0.6791 | 0.8889 | 0.3281 |
| jarn/baseline | 10000/2800 | 0.7022 | 0.8965 | 0.3560 |
| jarn/jarn | 10000/2800 | 0.7016 | 0.8989 | 0.3502 |

OOD alle diverse epoche, score orientati sempre verso ID:

| Sorgente | Epoca | Score | AUROC ID |
|---|---:|---|---:|
| fake | 1 | max_logit | 0.1821 |
| fake | 1 | msp | 0.3282 |
| fake | 5 | max_logit | 0.1100 |
| fake | 5 | msp | 0.0949 |
| fake | 10 | max_logit | 0.2409 |
| fake | 10 | msp | 0.1850 |
| cifar100 | 1 | max_logit | 0.6569 |
| cifar100 | 1 | msp | 0.6015 |
| cifar100 | 5 | max_logit | 0.7225 |
| cifar100 | 5 | msp | 0.6695 |
| cifar100 | 10 | max_logit | 0.7509 |
| cifar100 | 10 | msp | 0.7022 |

In questo training la separazione da CIFAR-100 migliora con le epoche e max-logit supera MSP. FakeData produce invece AUROC sotto 0,5: le immagini casuali ricevono spesso confidenza maggiore degli esempi ID. Gli score non vengono invertiti per far apparire migliore il risultato; rumore sintetico e immagini reali evidenziano comportamenti diversi.

| Attacco / modello | Epsilon | Accuracy sotto attacco | Successo sugli eleggibili |
|---|---:|---:|---:|
| fgsm | 0.0 | 72.97% | 0.00% |
| fgsm | 0.01 | 27.36% | 62.51% |
| fgsm | 0.03 | 2.42% | 96.68% |
| fgsm | 0.1 | 0.29% | 99.62% |
| targeted | 0.0 | 72.97% | 0.00% |
| targeted | 0.01 | 57.70% | 16.02% |
| targeted | 0.03 | 23.01% | 47.44% |
| targeted | 0.1 | 12.41% | 43.84% |
| adversarial/adversarial | 0.0 | 70.08% | 0.00% |
| adversarial/adversarial | 0.01 | 46.45% | 33.80% |
| adversarial/adversarial | 0.03 | 39.72% | 45.70% |
| adversarial/adversarial | 0.1 | 18.88% | 74.14% |
| adversarial/baseline | 0.0 | 72.97% | 0.00% |
| adversarial/baseline | 0.01 | 27.36% | 62.51% |
| adversarial/baseline | 0.03 | 2.42% | 96.68% |
| adversarial/baseline | 0.1 | 0.29% | 99.62% |
| jarn/baseline | 0.0 | 72.97% | 0.00% |
| jarn/baseline | 0.01 | 27.36% | 62.51% |
| jarn/baseline | 0.03 | 2.42% | 96.68% |
| jarn/baseline | 0.1 | 0.29% | 99.62% |
| jarn/jarn | 0.0 | 74.16% | 0.00% |
| jarn/jarn | 0.01 | 25.01% | 66.28% |
| jarn/jarn | 0.03 | 2.19% | 97.05% |
| jarn/jarn | 0.1 | 0.34% | 99.58% |

## Lettura degli esperimenti

MNIST: confrontare le profondità usando il test ufficiale; una rete più profonda non migliora automaticamente l’accuracy. CIFAR-10: plain e residual condividono split, seed, durata e batch size; il vantaggio delle skip connection va letto per profondità, insieme alle magnitudini dei gradienti, senza generalizzare da un solo training.

TweetEval: il fine-tuning va confrontato con l’encoder congelato attraverso accuracy e macro-F1; macro-F1 evidenzia la difficoltà delle classi rare di emoji. SWAG usa la validation ufficiale come holdout finale, poiché il test pubblico non fornisce le etichette; la selezione del checkpoint usa una parte separata del training.

Nelle esecuzioni del 14 ottobre 2026, CartPole: le baseline standardizzata e di valore raggiungono reward medio 500 sui 100 episodi aggiuntivi. REINFORCE senza baseline resta debole; DQN ottiene 468,47. LunarLander: DQN ottiene 213,88, mentre le tre configurazioni REINFORCE restano sotto 200. La lunghezza riportata per CarRacing conta decisioni dopo il frame skipping.

La robustezza qui misurata riguarda FGSM, con epsilon nello spazio dei pixel; non dimostra robustezza contro attacchi iterativi. Adversarial training, JARN e ODIN sono confronti di questa configurazione, non garanzie di miglioramento generale.

Adversarial training: accuracy FGSM a epsilon 0,03 2.42% → 39.72%; clean 72.97% → 70.08%; AUROC OOD 0.7022 → 0.6791.

JARN: clean 72.97% → 74.16%; accuracy FGSM a epsilon 0,03 2.42% → 2.19%. La regolarizzazione non produce un guadagno di robustezza in questo esperimento.

Controlli con uguale budget aggiuntivo: tutti partono dallo stesso checkpoint clean, con lo stesso split e selezione sulla loss di validation. Le epoche indicano il budget aggiuntivo massimo, non l’epoca del checkpoint migliore.

| Training aggiuntivo | Epoche | Clean accuracy | Accuracy FGSM eps=0,03 | AUROC OOD |
|---|---:|---:|---:|---:|
| clean_control_10 | 10 | 74.28% | 2.02% | 0.6984 |
| adversarial | 10 | 70.08% | 39.72% | 0.6791 |
| clean_control_5 | 5 | 74.28% | 2.02% | 0.6984 |
| jarn | 5 | 74.16% | 2.19% | 0.7016 |

ODIN seleziona T=10 ed epsilon=0.004 sulla validation. Nel test di questa esecuzione migliora MSP (AUROC 0,7461 contro 0,7022), ma non supera max-logit (0,7509). La ricerca degli iperparametri non garantisce il miglior risultato fra tutti gli score.

## Controlli funzionali

Forward/backward CUDA, equivalenza FCN, causalità GPT, encoder congelato, maschere multiple-choice, embedding normalizzati, aggiornamenti REINFORCE/DQN/PPO, FGSM/ODIN, doppia backpropagation JARN e ambienti Gymnasium: controlli reali superati. Dettagli in [tensor_checks.json](outputs/environment/tensor_checks.json) e [integration_checks.json](outputs/environment/integration_checks.json). PPO parallelo: training reale di 2.048 transizioni e verifica numerica di GAE/reset/bootstrap, [ppo_parallel_checks.json](outputs/environment/ppo_parallel_checks.json). Lo script di attivazione e `pip check` hanno completato senza errori.

## Riproduzione

Le configurazioni e i comandi esatti sono nei registri `status_*.json` e nei `config.json` di ogni esecuzione. Per proseguire o ripetere soltanto i job non riusciti usare `python run_experiments.py --group cpu`, `--group gpu`, `--group vision` o `--group nlp_extra`; i job completati sono saltati. CarRacing può usare le code indipendenti `--group racing_dqn` e `--group racing_ppo`. I controlli clean del laboratorio 4 sono riproducibili con `--group ood_extra`. Aggiornare questo documento con `python report_experiments.py`. `finish_remaining.py` completa soltanto i quattro job CarRacing mancanti e aggiorna questo report autonomamente. I fallimenti iniziali dovuti al download restano nei vecchi log, senza cancellare le evidenze.
