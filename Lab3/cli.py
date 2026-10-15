from pathlib import Path

from common.plots import curves
from common.runtime import load_checkpoint, parser, positive_float, positive_int, seed_everything, setup, write_csv
from lab3.agents import DQN, PPO, Reinforce
from lab3.envs import make_env
from lab3.training import checkpoint_evaluation, continuation_checkpoint, train_dqn, train_ppo, train_reinforce


def rl_parser(description, episodes=1000):
    p = parser(description, training=False)
    p.add_argument("--lr", type=positive_float, default=1e-3)
    p.add_argument("--episodes", type=positive_int, default=episodes)
    p.add_argument("--eval-every", type=positive_int, default=50)
    p.add_argument("--eval-episodes", type=positive_int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--video", action="store_true", help="Registra il primo episodio quando si carica --checkpoint")
    return p


def reinforce_main(exercise, env_name="CartPole-v1", baselines=("none",)):
    p = rl_parser("REINFORCE: confronto delle baseline", 2000 if env_name == "LunarLander-v3" else 1000)
    p.add_argument("--baselines", nargs="+", choices=["none", "standardized", "value"], default=list(baselines))
    args = p.parse_args()
    if not 0 <= args.gamma <= 1:
        p.error("--gamma deve essere tra 0 e 1")
    device, out = setup(args, "lab3", exercise)
    saved = load_checkpoint(args.checkpoint, device, "rl_agent") if args.checkpoint else None
    env = make_env(saved["config"]["env"] if saved else env_name)
    shape, actions = env.observation_space.shape, env.action_space.n
    env.close()
    if saved:
        config = saved["config"]
        if config["algorithm"] != "reinforce":
            p.error("Checkpoint REINFORCE richiesto")
        agent = Reinforce(shape, actions, device, baseline=config["baseline"], gamma=config["gamma"])
        agent.load_state_dict(saved["agent"])
        checkpoint_evaluation(agent, args, out, config)
        return
    comparison, summary = [], []
    for baseline in args.baselines:
        seed_everything(args.seed)
        config = dict(algorithm="reinforce", env=env_name, baseline=baseline, gamma=args.gamma)
        rows = train_reinforce(Reinforce(shape, actions, device, args.lr, args.gamma, baseline), args, out / baseline, config)
        comparison.extend(dict(episode=row["episode"], **{baseline: row["eval_reward"]}) for row in rows if "eval_reward" in row)
        summary.append(dict(baseline=baseline, final_eval_reward=rows[-1]["eval_reward"]))
    curves(comparison, out / "comparison.png", args.baselines, xlabel="episode")
    write_csv(out / "comparison.csv", summary)


def add_dqn_arguments(p, racing=False):
    p.add_argument("--capacity", type=positive_int, default=10000 if racing else 50000)
    p.add_argument("--warmup", type=positive_int, default=1000)
    p.add_argument("--update-every", type=positive_int, default=4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-steps", type=positive_int, default=200000 if racing else 50000)


def control_main(exercise, racing=False):
    p = rl_parser("DQN/PPO con training e valutazione separati", 3000 if racing else 1000)
    p.set_defaults(lr=3e-4, batch_size=64)
    p.add_argument("--env", choices=["CarRacing-v3"] if racing else ["CartPole-v1", "LunarLander-v3"],
                   default="CarRacing-v3" if racing else "CartPole-v1")
    p.add_argument("--algorithm", choices=["dqn", "ppo"] if racing else ["dqn"], default="ppo" if racing else "dqn")
    p.add_argument("--frames", type=positive_int, default=4)
    p.add_argument("--frame-skip", type=positive_int, default=4)
    add_dqn_arguments(p, racing)
    p.add_argument("--total-steps", type=positive_int, default=1000000)
    p.add_argument("--num-envs", type=positive_int, default=8 if racing else 1,
                   help="Ambienti paralleli per PPO; 1 usa il loop sequenziale")
    p.add_argument("--rollout-steps", type=positive_int, default=1024)
    p.add_argument("--update-epochs", type=positive_int, default=4)
    p.add_argument("--minibatch", type=positive_int, default=64)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--continue-run", type=Path,
                   help="Continua dai pesi di recovery.pt/best.pt e dai contatori del run, con optimizer e buffer nuovi")
    args = p.parse_args()
    if args.checkpoint and args.continue_run:
        p.error("Usare solo uno fra checkpoint (evaluation) e continue-run (training)")
    if not (0 <= args.gamma <= 1 and 0 < args.tau <= 1 and 0 <= args.epsilon_end <= args.epsilon_start <= 1):
        p.error("gamma/tau/epsilon fuori intervallo")
    if args.capacity < max(args.warmup, args.batch_size) or not 0 <= args.gae_lambda <= 1 or args.clip <= 0:
        p.error("Replay insufficiente o parametri PPO non validi")
    if args.algorithm == "ppo" and (args.total_steps % args.num_envs or args.rollout_steps % args.num_envs):
        p.error("Per PPO total-steps e rollout-steps devono essere divisibili per num-envs")
    device, out = setup(args, "lab3", exercise)
    source = continuation_checkpoint(args.continue_run) if args.continue_run else args.checkpoint
    saved = load_checkpoint(source, device, "rl_agent") if source else None
    config = saved["config"] if saved else dict(algorithm=args.algorithm, env=args.env,
                                               frames=args.frames, frame_skip=args.frame_skip, gamma=args.gamma)
    if args.continue_run:
        for key in ("algorithm", "env", "frames", "frame_skip", "gamma"):
            setattr(args, key, config[key])
    env = make_env(config["env"], frames=config["frames"], skip=config["frame_skip"])
    shape, actions = env.observation_space.shape, env.action_space.n
    env.close()
    if config["algorithm"] == "dqn":
        agent = DQN(shape, actions, device, args.lr, config["gamma"], args.capacity, args.batch_size,
                    args.tau, args.warmup, args.update_every)
        train = train_dqn
    elif config["algorithm"] == "ppo":
        agent = PPO(shape, actions, device, args.lr, config["gamma"], args.gae_lambda, args.clip,
                    args.update_epochs, args.minibatch, args.entropy)
        train = train_ppo
    else:
        p.error("Checkpoint DQN/PPO richiesto")
    if saved:
        agent.load_state_dict(saved["agent"])
    if saved and not args.continue_run:
        checkpoint_evaluation(agent, args, out, config)
    else:
        train(agent, args, out, config)


