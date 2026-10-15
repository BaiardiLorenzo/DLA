import csv
import shutil
import numpy as np
import torch

from common.plots import curves
from common.runtime import load_checkpoint, write_csv, write_json
from lab3.envs import make_env
from lab3.models import tensor


def evaluate(agent, env, episodes, seed):
    rewards, lengths = [], []
    for number in range(episodes):
        state, _ = env.reset(seed=seed + number)
        total, length = 0.0, 0
        while True:
            state, reward, terminated, truncated, _ = env.step(agent.act(state, deterministic=True))
            total += reward
            length += 1
            if terminated or truncated:
                break
        rewards.append(total)
        lengths.append(length)
    return dict(eval_reward=float(np.mean(rewards)), eval_reward_std=float(np.std(rewards)),
                eval_length=float(np.mean(lengths)))


def save_agent(agent, config, out, name, progress=None):
    state = dict(kind="rl_agent", config=config, agent=agent.state_dict())
    if progress is not None:
        state["progress"] = progress
    temporary = out / (name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(out / name)


def continuation_checkpoint(folder):
    folder = folder.resolve()
    recovery = folder / "recovery.pt"
    return recovery if recovery.is_file() else folder / "best.pt"


def continuation(args, out):
    """Continua dai pesi salvati; optimizer, replay e simulatori ripartono nuovi."""
    progress = dict(episode=0, steps=0, update=0, best=-float("inf"))
    folder = getattr(args, "continue_run", None)
    if folder is None:
        return [], progress
    folder = folder.resolve()
    checkpoint = continuation_checkpoint(folder)
    saved = load_checkpoint(checkpoint, "cpu", "rl_agent")
    rows = []
    with (folder / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        for item in csv.DictReader(stream):
            rows.append({key: int(float(value)) if key in ("episode", "steps", "update", "env_index", "train_length")
                         else float(value) for key, value in item.items() if value not in (None, "")})
    if "progress" in saved:
        progress.update(saved["progress"])
        rows = [row for row in rows if row.get("steps", 0) <= progress["steps"]]
    else:
        # I checkpoint originali contengono solo i pesi migliori. Individuiamo
        # l'istante corrispondente, senza attribuire loro il training successivo.
        index = max((i for i, row in enumerate(rows) if "eval_reward" in row),
                    key=lambda i: rows[i]["eval_reward"])
        rows = rows[:index + 1]
        progress.update({key: rows[-1].get(key, 0) for key in ("episode", "steps", "update")})
    progress["best"] = max((row["eval_reward"] for row in rows if "eval_reward" in row), default=-float("inf"))
    shutil.copyfile(folder / "best.pt", out / "best.pt")
    write_json(out / "continuation.json", dict(source=str(folder), checkpoint=str(checkpoint),
               restored_progress=progress, optimizer_restored=False, replay_restored=False,
               environment_restored=False, history="prefisso fino al checkpoint caricato"))
    write_csv(out / "metrics.csv", rows)
    print(f"Continuazione dai pesi: {checkpoint}; contatori: {progress}", flush=True)
    return rows, progress


def report(rows, out):
    write_csv(out / "metrics.csv", rows)
    curves(rows, out / "rewards.png", ["train_reward", "eval_reward"], xlabel="episode")
    curves(rows, out / "lengths.png", ["train_length", "eval_length"], xlabel="episode")


def train_reinforce(agent, args, out, config):
    out.mkdir(parents=True, exist_ok=True)
    env, eval_env = make_env(config["env"]), make_env(config["env"])
    rows, best = [], -float("inf")
    try:
        for number in range(1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + number)
            states, actions, rewards = [], [], []
            while True:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                states.append(state.copy())
                actions.append(action)
                rewards.append(reward)
                state = next_state
                if terminated or truncated:
                    break
            row = dict(episode=number, train_reward=float(sum(rewards)), train_length=len(rewards),
                       **agent.update(states, actions, rewards, state, terminated))
            if number % args.eval_every == 0 or number == args.episodes:
                row.update(evaluate(agent, eval_env, args.eval_episodes, args.seed + 100000))
                print(row)
                if row["eval_reward"] > best:
                    best = row["eval_reward"]
                    save_agent(agent, config, out, "best.pt")
            rows.append(row)
            write_csv(out / "metrics.csv", rows)
        save_agent(agent, config, out, "last.pt")
        report(rows, out)
    finally:
        env.close()
        eval_env.close()
    return rows


def train_dqn(agent, args, out, config):
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(frames=args.frames, skip=args.frame_skip)
    env, eval_env = make_env(config["env"], **kwargs), make_env(config["env"], **kwargs)
    rows, progress = continuation(args, out)
    best = progress["best"]
    agent.steps = progress["steps"]
    try:
        for number in range(progress["episode"] + 1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + number)
            total, length, losses = 0.0, 0, []
            while True:
                epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * max(0, 1 - agent.steps / args.epsilon_steps)
                action = agent.act(state, epsilon=epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                loss = agent.observe(state, action, reward, next_state, terminated)
                if loss is not None:
                    losses.append(loss)
                total += reward
                length += 1
                state = next_state
                if terminated or truncated:
                    break
            row = dict(episode=number, steps=agent.steps, train_reward=total, train_length=length,
                       epsilon=epsilon, q_loss=float(np.mean(losses)) if losses else None)
            if number % args.eval_every == 0 or number == args.episodes:
                row.update(evaluate(agent, eval_env, args.eval_episodes, args.seed + 100000))
                print(row)
                if row["eval_reward"] > best:
                    best = row["eval_reward"]
                    save_agent(agent, config, out, "best.pt")
            rows.append(row)
            write_csv(out / "metrics.csv", rows)
            save_agent(agent, config, out, "recovery.pt", dict(episode=number, steps=agent.steps, update=0))
        save_agent(agent, config, out, "last.pt")
        report(rows, out)
    finally:
        env.close()
        eval_env.close()
    return rows


def train_ppo(agent, args, out, config):
    if args.num_envs > 1:
        return train_ppo_parallel(agent, args, out, config)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(frames=args.frames, skip=args.frame_skip)
    env, eval_env = make_env(config["env"], **kwargs), make_env(config["env"], **kwargs)
    rows, progress = continuation(args, out)
    buffer, best = [], progress["best"]
    episode, steps, total, length = progress["episode"] + 1, progress["steps"], 0.0, 0
    try:
        state, _ = env.reset(seed=args.seed + episode)
        while steps < args.total_steps:
            action, log_probability, value = agent.sample(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            with torch.no_grad():
                next_value = 0.0 if terminated else agent.value(tensor(next_state, agent.device)).item()
            done = terminated or truncated
            buffer.append((state.copy(), action, log_probability, value, reward, next_value, terminated, done))
            state = next_state
            steps += 1
            total += reward
            length += 1
            if done:
                rows.append(dict(episode=episode, steps=steps, train_reward=total, train_length=length))
                episode += 1
                total, length = 0.0, 0
                state, _ = env.reset(seed=args.seed + episode)
            if len(buffer) == args.rollout_steps or steps == args.total_steps:
                losses = agent.update(buffer)
                buffer.clear()
                update = (steps + args.rollout_steps - 1) // args.rollout_steps
                row = dict(episode=episode, steps=steps, update=update, **losses)
                # Ogni N rollout, non N episodi: frequenza esplicita per PPO.
                if update % args.eval_every == 0 or steps == args.total_steps:
                    row.update(evaluate(agent, eval_env, args.eval_episodes, args.seed + 100000))
                    print(row)
                    if row["eval_reward"] > best:
                        best = row["eval_reward"]
                        save_agent(agent, config, out, "best.pt")
                rows.append(row)
                write_csv(out / "metrics.csv", rows)
                save_agent(agent, config, out, "recovery.pt", dict(episode=episode, steps=steps, update=update))
        save_agent(agent, config, out, "last.pt")
        report(rows, out)
    finally:
        env.close()
        eval_env.close()
    return rows


def train_ppo_parallel(agent, args, out, config):
    """PPO su più simulatori; GAE resta separato per traiettoria e ambiente."""
    from functools import partial
    import gymnasium as gym

    out.mkdir(parents=True, exist_ok=True)
    factory = partial(make_env, config["env"], frames=args.frames, skip=args.frame_skip)
    env = gym.vector.AsyncVectorEnv([factory] * args.num_envs, context="spawn",
                                    autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    eval_env = factory()
    config = dict(config, num_envs=args.num_envs)
    rows, progress = continuation(args, out)
    best, update, steps, episode = progress["best"], progress["update"], progress["steps"], progress["episode"]
    totals = np.zeros(args.num_envs)
    lengths = np.zeros(args.num_envs, dtype=int)
    buffers = [[] for _ in range(args.num_envs)]
    try:
        states, _ = env.reset(seed=args.seed + episode * args.num_envs)
        while steps < args.total_steps:
            actions, log_probs, values = agent.sample_batch(states)
            next_states, rewards, terminated, truncated, _ = env.step(actions)
            with torch.no_grad():
                next_values = agent.value(torch.as_tensor(next_states, device=agent.device)).squeeze(-1).cpu().numpy()
            done = terminated | truncated
            next_values[terminated] = 0.0
            totals += rewards
            lengths += 1
            steps += args.num_envs
            for index in range(args.num_envs):
                buffers[index].append((states[index].copy(), int(actions[index]), float(log_probs[index]),
                                       float(values[index]), float(rewards[index]), float(next_values[index]),
                                       bool(terminated[index]), bool(done[index])))
                if done[index]:
                    episode += 1
                    rows.append(dict(episode=episode, env_index=index, steps=steps,
                                     train_reward=float(totals[index]), train_length=int(lengths[index])))
                    totals[index], lengths[index] = 0.0, 0
            # Il bootstrap sopra usa l'osservazione finale, prima del reset mascherato.
            states = env.reset(options={"reset_mask": done})[0] if done.any() else next_states
            if sum(map(len, buffers)) == args.rollout_steps or steps == args.total_steps:
                transitions = []
                for buffer in buffers:
                    # Evita che GAE attraversi il confine fra ambienti concatenati.
                    # Non è un terminale: il valore next_state continua a fare bootstrap.
                    buffer[-1] = (*buffer[-1][:-1], True)
                    transitions.extend(buffer)
                    buffer.clear()
                losses = agent.update(transitions)
                update += 1
                row = dict(episode=episode, steps=steps, update=update, **losses)
                if update % args.eval_every == 0 or steps == args.total_steps:
                    row.update(evaluate(agent, eval_env, args.eval_episodes, args.seed + 100000))
                    print(row)
                    if row["eval_reward"] > best:
                        best = row["eval_reward"]
                        save_agent(agent, config, out, "best.pt")
                rows.append(row)
                write_csv(out / "metrics.csv", rows)
                save_agent(agent, config, out, "recovery.pt", dict(episode=episode, steps=steps, update=update))
        save_agent(agent, config, out, "last.pt")
        report(rows, out)
    finally:
        env.close()
        eval_env.close()
    return rows


def checkpoint_evaluation(agent, args, out, config):
    video_folder = out / "video" if args.video else None
    env = make_env(config["env"], video_folder, frames=config.get("frames", 4), skip=config.get("frame_skip", 4))
    try:
        result = evaluate(agent, env, args.eval_episodes, args.seed + 200000)
        write_json(out / "evaluation.json", result)
        print(result)
    finally:
        env.close()
