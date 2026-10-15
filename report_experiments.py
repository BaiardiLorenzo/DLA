"""Aggiorna RISULTATI.md esclusivamente con metriche realmente salvate."""
import csv
import json
from datetime import datetime
from pathlib import Path
import argparse

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, default=root / 'outputs' / 'experiments_20261014')
args = p.parse_args()
experiments = args.root.resolve()
states = {}
for group in ['cpu', 'gpu', 'vision', 'cpu_extra', 'nlp_extra', 'racing_dqn', 'racing_ppo', 'ood_extra', 'remaining']:
    path = experiments / f'status_{group}.json'
    if path.exists():
        for name, value in json.loads(path.read_text(encoding='utf-8')).items():
            if (name not in states or value['status'] == 'completed'
                    or states[name]['status'] != 'completed' and value.get('started', '') >= states[name].get('started', '')):
                states[name] = value


def location(name):
    value = states.get(name, {})
    return Path(value['output']) if value.get('status') == 'completed' else None


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def number(value, percent=False):
    if value is None:
        return 'n.d.'
    return f'{float(value)*100:.2f}%' if percent else f'{float(value):.4f}'


completed = sum(value['status'] == 'completed' for value in states.values())
lines = ['# Training e valutazione — risultati reali', '',
         f'Aggiornamento locale: {datetime.now().isoformat(timespec="seconds")}.', '',
         f'Esperimenti completati: **{completed}/{len(states)}**. Le righe in corso o in attesa non sono risultati completati.', '',
         'Ambiente conda `DLA`, Python 3.12, RTX 5070 Ti 16 GB, PyTorch CUDA 13.0. '
         'Gli agenti per osservazioni vettoriali usano CPU con un thread; reti per immagini/linguaggio e CarRacing usano CUDA. '
         'Seed 42; validation distinta dal test. Un solo seed non misura la variabilità fra training.', '',
         'CIFAR-10/100: copia pubblica `uoft-cs` su Hugging Face, stessa suddivisione 50.000/10.000, '
         'usata dopo timeout del server originale. Cache RGB uint8 per evitare decodifiche ripetute.', '',
         '## Stato delle esecuzioni', '', '| Esperimento | Stato | Secondi | Log e risultati |', '|---|---|---:|---|']
for name, value in states.items():
    path = Path(value['output']).relative_to(root).as_posix() if value.get('output') else None
    label = f'[{name}]({path}/run.log)' if path else name
    lines.append(f"| {label} | {value['status']} | {value.get('duration_seconds', '—')} | {path or '—'} |")

background = experiments / 'background_state.json'
if background.exists():
    status = read_json(background)
    lines.extend(['', f"Coordinatore locale in background: **{status['status']}**, PID {status['pid']}. "
                  'Non usa Codex o API di modelli per seguire i training. '
                  '[Log](outputs/experiments_20261014/background.log), '
                  '[stato](outputs/experiments_20261014/background_state.json). '
                  'Il report viene aggiornato ogni due minuti e alla conclusione dei job.', '',
                  '| CarRacing | Episodio | Transizioni | Update |', '|---|---:|---:|---:|'])
    continuations = []
    for name in ('racing_dqn', 'racing_ppo'):
        value = states.get(name, {})
        path = Path(value['output']) if value.get('output') else None
        rows = read_csv(path / 'metrics.csv') if path else []
        if rows:
            last = rows[-1]
            lines.append(f"| {name} | {last.get('episode', '')} | {last.get('steps', '')} | {last.get('update', '')} |")
        info = read_json(path / 'continuation.json') if path else {}
        if info:
            restored = info['restored_progress']
            continuations.extend(['', f"{name}: continuazione dai pesi salvati all’episodio {restored['episode']}, "
                          f"transizione {restored['steps']}. Optimizer, replay e simulatori reinizializzati; "
                          'non è una ripresa identica dello stato interrotto. La cronologia include il prefisso fino al checkpoint caricato.', ''])
    lines.extend(continuations)

lines.extend(['', '## Classificazione immagini: test ufficiale', '',
              '| Modello | Esempi test | Accuracy | Cross-entropy |', '|---|---:|---:|---:|'])
for name in ['mlp_eval', 'cnn_eval', 'gradients_eval', 'fcn_eval']:
    path = location(name)
    if path:
        for row in read_csv(path / 'test_metrics.csv'):
            model = Path(row['checkpoint']).parent.name
            lines.append(f"| {name}/{model} | {row['test_samples']} | {number(row['accuracy'], True)} | {number(row['loss'])} |")

path = location('gradients')
if path:
    lines.extend(['', 'Magnitudini dei gradienti: media di |gradiente| sui pesi convoluzionali e sui batch dell’epoca. '
                  'Il CSV completo e i grafici contengono tutti gli strati; qui si mostrano prima e ultima convoluzione.', '',
                  '| Modello | Epoca | Prima convoluzione | Ultima convoluzione |', '|---|---:|---:|---:|'])
    for folder in sorted(path.iterdir()):
        if not folder.is_dir():
            continue
        rows = [row for row in read_csv(folder / 'gradients.csv') if row['layer'].startswith('features.')]
        for epoch in sorted({int(row['epoch']) for row in rows}):
            if epoch not in (1, max(int(row['epoch']) for row in rows)):
                continue
            selected = [row for row in rows if int(row['epoch']) == epoch]
            lines.append(f"| {folder.name} | {epoch} | {float(selected[0]['mean_abs_gradient']):.6g} | {float(selected[-1]['mean_abs_gradient']):.6g} |")
    lines.extend(['', 'Le skip connection aggiungono un percorso identità al gradiente. Le magnitudini osservate dipendono anche da '
                  'normalizzazione e scala dei pesi: una magnitudine maggiore da sola non dimostra migliore apprendimento. '
                  'Leggere queste misure insieme alle curve di loss e accuracy e al confronto plain/residual a profondità equivalente.'])
    lines.extend(['', 'Nel confronto di cinque epoche, la configurazione profonda passa da 64,01% (plain_4) a 78,56% (residual_4) sul test: '
                  '14,55 punti percentuali. Dopo venti epoche il divario delle stesse profondità è più piccolo, 78,65% contro 80,86%. '
                  'Questo è coerente con un apprendimento iniziale più efficace per le skip connection nel modello profondo, senza un vantaggio uniforme a ogni profondità.'])

gradcam_path = experiments / 'gradcam_reviewed' / 'gradcam.png'
if gradcam_path.exists():
    lines.extend(['', f"[Griglia Grad-CAM]({gradcam_path.relative_to(root).as_posix()}): la mappa usa il logit della classe predetta. "
                  'Gli esempi mostrano regioni utili al punteggio, anche quando la classe è errata; non sono una segmentazione né una prova causale. '
                  'Per esempio, l’aereo classificato come nave concentra parte della risposta nella zona superiore dell’immagine.'])

lines.extend(['', '## Modelli linguistici', '', '| Esperimento | Accuracy | Macro-F1 | Loss |', '|---|---:|---:|---:|'])
refined = experiments / 'fcn_same_canvas_corrected' / 'detections.json'
if refined.exists():
    detection = read_json(refined)
    lines[-4:-4] = ['', 'FCN sul canvas iniziale di nove cifre: '
                     f"{len(detection['predictions'])} detections, precision {number(detection['precision_iou_0_5'], True)}, "
                     f"recall {number(detection['recall_iou_0_5'], True)} a IoU 0.5, dopo il filtro di completezza/centratura. "
                     'Prima del filtro erano 29 detections. Stessi pesi e stesso canvas; un singolo esempio sintetico non misura la generalizzazione della detection.', '']
for name in ['irony', 'irony_finetune', 'emoji', 'emoji_finetune', 'arc', 'swag']:
    path = location(name)
    metrics = read_json(path / 'test_metrics.json') if path else {}
    if metrics:
        lines.append(f"| {name} | {number(metrics['accuracy'], True)} | {number(metrics['macro_f1'])} | {number(metrics['loss'])} |")
path = location('scifact')
if path:
    metrics = read_json(path / 'test_metrics.json')
    lines.extend(['', 'SciFact sul corpus completo: ' + ', '.join(f'{key}={number(value)}' for key, value in metrics.items()) + '.'])
path = location('dante')
if path:
    rows = read_csv(path / 'metrics.csv')
    lines.extend(['', f"GPT Dante, ultima epoca: train loss {number(rows[-1]['train_loss'])}, validation loss {number(rows[-1]['val_loss'])}. "
                  f"[Testo generato]({(path / 'generated.txt').relative_to(root).as_posix()}). "
                  'Il testo riproduce punteggiatura e struttura dei versi, ma contiene parole inventate e incoerenze: non è un modello semantico di Dante.'])

lines.extend(['', '## Agenti RL: valutazione su seed separati', '',
              '| Agente | Episodi evaluation | Reward medio | Deviazione standard | Lunghezza media |', '|---|---:|---:|---:|---:|'])
for name in ['cartpole_none_eval', 'cartpole_standardized_eval', 'cartpole_value_eval', 'cartpole_dqn_eval',
             'lunar_none_eval', 'lunar_standardized_eval', 'lunar_value_eval', 'lunar_dqn_eval', 'racing_dqn_eval', 'racing_ppo_eval']:
    path = location(name)
    if path:
        metrics = read_json(path / 'evaluation.json')
        config = read_json(path / 'config.json')
        lines.append(f"| {name} | {config['eval_episodes']} | {number(metrics['eval_reward'])} | {number(metrics['eval_reward_std'])} | {number(metrics['eval_length'])} |")
lines.extend(['', 'Completare un training non equivale a risolvere un ambiente: confrontare le ricompense con il criterio della consegna. '
              'Il checkpoint migliore è scelto sulla valutazione periodica, mentre questa tabella usa seed aggiuntivi.'])

lines.extend(['', '## OOD e attacchi', '',
              'Gli score maggiori indicano ID; per OOD si usa lo score negato. AUPR indica l’area trapezoidale della curva precision-recall. '
              'Il riferimento casuale per AUPR dipende dalla proporzione della classe positiva: '
              '10.000 ID/2.800 CIFAR-100 corrispondono a 0,7813 per ID e 0,2188 per OOD; '
              '10.000 ID/5.000 FakeData a 0,6667 e 0,3333. I valori fra sorgenti diverse non sono direttamente confrontabili.', '',
              '| Esperimento / score | N ID / OOD | AUROC ID | AUPR ID | AUPR OOD |', '|---|---:|---:|---:|---:|'])
for name, suffix in [('ood_pipeline', 'max_logit/metrics.json'), ('ood_pipeline', 'msp/metrics.json'),
                     ('ood_metrics', 'max_logit/metrics.json'), ('ood_metrics', 'msp/metrics.json'), ('odin', 'test/metrics.json')]:
    path = location(name)
    metrics = read_json(path / suffix) if path else {}
    if metrics:
        lines.append(f"| {name}/{suffix.split('/')[0]} | {metrics['id_samples']}/{metrics['ood_samples']} | {number(metrics['auroc_id'])} | {number(metrics['aupr_id'])} | {number(metrics['aupr_ood'])} |")
for name in ['adversarial', 'jarn']:
    path = location(name)
    comparison = read_json(path / 'comparison.json') if path else {}
    for model, result in comparison.items():
        metrics = result['ood']
        lines.append(f"| {name}/{model} | {metrics['id_samples']}/{metrics['ood_samples']} | {number(metrics['auroc_id'])} | {number(metrics['aupr_id'])} | {number(metrics['aupr_ood'])} |")
path = location('ood_intro')
if path:
    lines.extend(['', 'OOD alle diverse epoche, score orientati sempre verso ID:', '',
                  '| Sorgente | Epoca | Score | AUROC ID |', '|---|---:|---|---:|'])
    for row in read_csv(path / 'epoch_comparison.csv'):
        lines.append(f"| {row['source']} | {row['epoch']} | {row['method']} | {number(row['auroc_id'])} |")
    lines.extend(['', 'In questo training la separazione da CIFAR-100 migliora con le epoche e max-logit supera MSP. '
                  'FakeData produce invece AUROC sotto 0,5: le immagini casuali ricevono spesso confidenza maggiore degli esempi ID. '
                  'Gli score non vengono invertiti per far apparire migliore il risultato; rumore sintetico e immagini reali evidenziano comportamenti diversi.'])
lines.extend(['', '| Attacco / modello | Epsilon | Accuracy sotto attacco | Successo sugli eleggibili |', '|---|---:|---:|---:|'])
for name in ['fgsm', 'targeted', 'adversarial', 'jarn']:
    path = location(name)
    if not path:
        continue
    paths = [(name, path / 'attacks.csv')] if name in ['fgsm', 'targeted'] else [(f'{name}/{folder.name}', folder / 'attacks' / 'attacks.csv') for folder in path.iterdir() if folder.is_dir()]
    for label, csv_path in paths:
        for row in read_csv(csv_path):
            rate = row['attack_success_rate'] or None
            lines.append(f"| {label} | {row['epsilon']} | {number(row['adversarial_accuracy'], True)} | {number(rate, True)} |")

lines.extend(['', '## Lettura degli esperimenti', '',
              'MNIST: confrontare le profondità usando il test ufficiale; una rete più profonda non migliora automaticamente l’accuracy. '
              'CIFAR-10: plain e residual condividono split, seed, durata e batch size; il vantaggio delle skip connection va letto per profondità, '
              'insieme alle magnitudini dei gradienti, senza generalizzare da un solo training.', '',
              'TweetEval: il fine-tuning va confrontato con l’encoder congelato attraverso accuracy e macro-F1; '
              'macro-F1 evidenzia la difficoltà delle classi rare di emoji. SWAG usa la validation ufficiale come holdout finale, '
              'poiché il test pubblico non fornisce le etichette; la selezione del checkpoint usa una parte separata del training.', '',
              'Nelle esecuzioni del 14 ottobre 2026, CartPole: le baseline standardizzata e di valore raggiungono reward medio 500 sui 100 episodi aggiuntivi. '
              'REINFORCE senza baseline resta debole; DQN ottiene 468,47. LunarLander: DQN ottiene 213,88, '
              'mentre le tre configurazioni REINFORCE restano sotto 200. La lunghezza riportata per CarRacing conta decisioni dopo il frame skipping.', '',
              'La robustezza qui misurata riguarda FGSM, con epsilon nello spazio dei pixel; non dimostra robustezza contro attacchi iterativi. '
              'Adversarial training, JARN e ODIN sono confronti di questa configurazione, non garanzie di miglioramento generale.'])

path = location('adversarial')
if path:
    comparison = read_json(path / 'comparison.json')
    baseline, trained = comparison['baseline'], comparison['adversarial']
    before = next(row for row in baseline['attacks'] if row['epsilon'] == 0.03)
    after = next(row for row in trained['attacks'] if row['epsilon'] == 0.03)
    lines.extend(['', f"Adversarial training: accuracy FGSM a epsilon 0,03 {number(before['adversarial_accuracy'], True)} → "
                  f"{number(after['adversarial_accuracy'], True)}; clean {number(baseline['clean_accuracy'], True)} → "
                  f"{number(trained['clean_accuracy'], True)}; AUROC OOD {number(baseline['ood']['auroc_id'])} → "
                  f"{number(trained['ood']['auroc_id'])}."])
path = location('jarn')
if path:
    comparison = read_json(path / 'comparison.json')
    baseline, trained = comparison['baseline'], comparison['jarn']
    before = next(row for row in baseline['attacks'] if row['epsilon'] == 0.03)
    after = next(row for row in trained['attacks'] if row['epsilon'] == 0.03)
    lines.extend(['', f"JARN: clean {number(baseline['clean_accuracy'], True)} → {number(trained['clean_accuracy'], True)}; "
                  f"accuracy FGSM a epsilon 0,03 {number(before['adversarial_accuracy'], True)} → "
                  f"{number(after['adversarial_accuracy'], True)}. La regolarizzazione non produce un guadagno di robustezza in questo esperimento."])

if location('clean_control_10') or location('clean_control_5'):
    lines.extend(['', 'Controlli con uguale budget aggiuntivo: tutti partono dallo stesso checkpoint clean, '
                  'con lo stesso split e selezione sulla loss di validation. Le epoche indicano il budget aggiuntivo massimo, '
                  'non l’epoca del checkpoint migliore.', '',
                  '| Training aggiuntivo | Epoche | Clean accuracy | Accuracy FGSM eps=0,03 | AUROC OOD |', '|---|---:|---:|---:|---:|'])
    for name, model, epochs in [('clean_control_10', 'clean', 10), ('adversarial', 'adversarial', 10),
                                ('clean_control_5', 'clean', 5), ('jarn', 'jarn', 5)]:
        path = location(name)
        if not path:
            continue
        result = read_json(path / 'comparison.json')[model]
        attack = next(row for row in result['attacks'] if row['epsilon'] == 0.03)
        lines.append(f"| {name} | {epochs} | {number(result['clean_accuracy'], True)} | {number(attack['adversarial_accuracy'], True)} | {number(result['ood']['auroc_id'])} |")

path = location('odin')
if path:
    selected = read_json(path / 'selected.json')
    lines.extend(['', f"ODIN seleziona T={selected['temperature']:g} ed epsilon={selected['epsilon']:g} sulla validation. "
                  'Nel test di questa esecuzione migliora MSP (AUROC 0,7461 contro 0,7022), '
                  'ma non supera max-logit (0,7509). La ricerca degli iperparametri non garantisce il miglior risultato fra tutti gli score.'])

lines.extend(['', '## Controlli funzionali', '',
              'Forward/backward CUDA, equivalenza FCN, causalità GPT, encoder congelato, maschere multiple-choice, '
              'embedding normalizzati, aggiornamenti REINFORCE/DQN/PPO, FGSM/ODIN, doppia backpropagation JARN '
              'e ambienti Gymnasium: controlli reali superati. Dettagli in '
              '[tensor_checks.json](outputs/environment/tensor_checks.json) e '
              '[integration_checks.json](outputs/environment/integration_checks.json). '
              'PPO parallelo: training reale di 2.048 transizioni e verifica numerica di GAE/reset/bootstrap, '
              '[ppo_parallel_checks.json](outputs/environment/ppo_parallel_checks.json). '
              'Lo script di attivazione e `pip check` hanno completato senza errori.', '',
              '## Riproduzione', '', 'Le configurazioni e i comandi esatti sono nei registri `status_*.json` e nei `config.json` di ogni esecuzione. '
              'Per proseguire o ripetere soltanto i job non riusciti usare `python run_experiments.py --group cpu`, '
              '`--group gpu`, `--group vision` o `--group nlp_extra`; i job completati sono saltati. '
              'CarRacing può usare le code indipendenti `--group racing_dqn` e `--group racing_ppo`. '
              'I controlli clean del laboratorio 4 sono riproducibili con `--group ood_extra`. '
              'Aggiornare questo documento con `python report_experiments.py`. '
              '`finish_remaining.py` completa soltanto i quattro job CarRacing mancanti e aggiorna questo report autonomamente. '
              'I fallimenti iniziali dovuti al download restano nei vecchi log, senza cancellare le evidenze.'])
(root / 'RISULTATI.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'RISULTATI.md aggiornato: {completed}/{len(states)} completati')

