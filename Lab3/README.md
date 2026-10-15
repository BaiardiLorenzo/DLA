# Lab 3 — reinforcement learning

[Consegne originali](CONSEGNE.md). Algoritmi PyTorch espliciti, nessuna libreria di agenti pronta e nessun tracking esterno. Gli esperimenti richiedono esecuzioni reali: il codice non dimostra da solo che un ambiente sia risolto.

| Consegna | Comando |
|---|---|
| 1 REINFORCE CartPole e valutazione separata | `python -m lab3.ex1` |
| 2 Confronto baseline none/standardized/value | `python -m lab3.ex2` |
| 3.1 REINFORCE LunarLander con tutte le baseline | `python -m lab3.ex3_1` |
| 3.2 DQN CartPole | `python -m lab3.ex3_2 --env CartPole-v1` |
| 3.2 DQN LunarLander | `python -m lab3.ex3_2 --env LunarLander-v3` |
| 3.3 CarRacing con DQN | `python -m lab3.ex3_3 --algorithm dqn` |
| 3.3 CarRacing con PPO | `python -m lab3.ex3_3 --algorithm ppo` |

## Esecuzioni e checkpoint

```powershell
python -m lab3.ex2 --output-dir outputs/lab3/cartpole_baselines
python -m lab3.ex3_2 --env LunarLander-v3 --output-dir outputs/lab3/lunar_dqn
python -m lab3.ex3_3 --algorithm ppo --output-dir outputs/lab3/racing_ppo
python -m lab3.ex3_3 --checkpoint outputs/lab3/racing_ppo/best.pt --video --eval-episodes 10
```

`--checkpoint` esegue solo una valutazione dell'agente salvato; `--video` registra il primo episodio. Non è un resume del training. Il checkpoint include ambiente, preprocessing e reti; per REINFORCE/PPO include la rete del valore quando presente. `last.pt` è l'ultimo agente e `best.pt` quello con ricompensa di valutazione migliore.

Per DQN/PPO `--continue-run <run_interrotto>` continua il training dai pesi di `recovery.pt` (o `best.pt` nei run precedenti), con optimizer e replay nuovi. Recupera i contatori e solo la cronologia precedente al checkpoint: le iterazioni successive prive di pesi salvati non sono attribuite al modello recuperato. Il budget richiesto è complessivo. `finish_remaining.ps1` completa i soli CarRacing mancanti in background e aggiorna il report senza usare Codex.

`--episodes` controlla REINFORCE/DQN. PPO usa `--total-steps`, contato **dopo frame skipping**, e `--rollout-steps`: entrambi indicano transizioni complessive, sommate fra tutti gli ambienti. CarRacing PPO usa per default `--num-envs 8`; `--num-envs 1` seleziona il loop sequenziale. I due budget devono essere divisibili per il numero di ambienti. I CSV riportano gli episodi di training e le valutazioni periodiche, con media della ricompensa totale non scontata, deviazione standard e lunghezza media. CarRacing misura la lunghezza in azioni del wrapper, non in frame nativi.

## Dettagli degli algoritmi

REINFORCE usa ritorni scontati e una distribuzione categorica parametrizzata dai logit. Le tre baseline sono: nessuna; standardizzazione con deviazione standard protetta contro zero; rete V(s) addestrata sui ritorni, con vantaggio separato dal grafo del valore. La rete del valore può fare bootstrap alle interruzioni TimeLimit; REINFORCE senza rete usa il ritorno della traiettoria episodica osservata. `--baselines` permette confronti selettivi.

DQN usa replay uniforme, rete target inizializzata dalla rete online, loss Huber, clipping dei gradienti e aggiornamento soft con `--tau`. L'esplorazione epsilon-greedy decade linearmente in `--epsilon-steps`. Il bootstrap viene annullato su `terminated`, mantenuto su `truncated`; il reset avviene su entrambi. `--capacity`, `--warmup`, `--update-every` e `--batch-size` controllano il replay. CarRacing conserva frame uint8 sul processore: capacità predefinita 10.000, circa 565 MB per le due osservazioni per transizione, oltre all'overhead.

PPO conserva log-probabilità e valori della policy prima dell'aggiornamento. GAE fa bootstrap sullo stato finale dei TimeLimit e interrompe la ricorrenza a ogni reset. Loss clipped, regressione del valore ed entropia; vantaggi normalizzati e minibatch rimescolati per più epoche. Il buffer è on-policy e viene svuotato dopo ogni update. `--gae-lambda`, `--clip`, `--entropy`, `--minibatch` e `--update-epochs` sono configurabili.

Con più ambienti PPO usa [AsyncVectorEnv](https://gymnasium.farama.org/api/vector/async_vector_env/) e processi Windows `spawn`. Il reset automatico è disabilitato: il codice calcola il bootstrap sullo stato finale e poi resetta soltanto gli ambienti terminati tramite maschera, come documentato da [Farama](https://farama.org/Vector-Autoreset-Mode). GAE non passa fra ambienti concatenati nel minibatch. La valutazione resta su un singolo ambiente distinto e non alimenta i rollout.

Per CarRacing si usa **continuous=False**. Crop della regione di gioco, grayscale 84×84, quattro frame impilati e ripetizione dell'azione per quattro frame; i reward intermedi vengono sommati e il frame skipping si ferma appena l'episodio finisce. DQN e PPO usano una CNN con due convoluzioni; la divisione per 255 avviene nella rete.

Ogni `--eval-every` episodi (REINFORCE/DQN) o rollout aggiornati (PPO), l'agente viene eseguito per `--eval-episodes` episodi in un **secondo ambiente**, con azioni greedy e seed dedicati. Il confronto periodico usa lo stesso insieme di seed fra modelli. La valutazione non inserisce transizioni nei buffer e non cambia l'esplorazione. L'analisi con checkpoint usa un ulteriore insieme di seed.

## Migrazione degli ambienti e analisi

Gli ID correnti sono `CartPole-v1`, `LunarLander-v3`, `CarRacing-v3`. LunarLander v3 corregge reset e nondeterminismo; CarRacing aggiorna la semantica di fine pista e terminazione. Non confrontare direttamente i vecchi video/score v2 con quelli nuovi senza considerare queste differenze. Riferimenti: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/), [CarRacing](https://gymnasium.farama.org/environments/box2d/car_racing/).

Confronta velocità di apprendimento, stabilità e ricompense su episodi di valutazione; la media mobile del training non sostituisce questa misura. Puoi ripetere gli esperimenti con seed diversi. Per dichiarare un ambiente risolto considera una valutazione sufficientemente ampia e il criterio della consegna, senza dedurlo soltanto da un singolo episodio favorevole. Training e valutazioni reali sono documentati in [RISULTATI.md](../RISULTATI.md), compresi gli agenti che non raggiungono la soglia di soluzione.
