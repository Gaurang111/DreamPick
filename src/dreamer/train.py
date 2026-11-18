# Adapted from NM512/dreamerv3-torch's dreamer.py (MIT License - see
# LICENSE_DREAMERV3_TORCH.txt), a PyTorch port of DreamerV3. The Dreamer class
# and main() training loop are unchanged from upstream; make_env() was trimmed
# to only our SO101 task (other suites dmc/atari/crafter/minecraft/... removed
# since we did not vendor their env wrappers/dependencies), and the Linux-only
# MUJOCO_GL=osmesa line was dropped (Windows uses the default GLFW backend).
# https://github.com/NM512/dreamerv3-torch

import argparse
import functools
import json
import os
import pathlib
import sys

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
from parallel import Parallel, Damy
from thermal import ThermalGuard
from joint_pos_trace import TraceWriter

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, logdir):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        # Separate, much less frequent interval for the expensive image
        # reconstruction log (encodes+decodes both camera views) - now that
        # joint_pos_trace.bin tracks progression continuously and cheaply,
        # this doesn't need to run as often as the scalar metrics do.
        self._should_log_video = tools.Every(config.video_pred_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        # --- Added for SO101PickCube: frequent checkpointing (independent of
        # the slow eval_every cadence) and a GPU-temperature safety pause.
        # Dream/reconstruction videos are NOT saved during training (every log
        # event would be wasteful) - they still go to TensorBoard as before,
        # and generate_dream_video.py makes a standalone mp4 on demand from a
        # checkpoint, once there's something worth looking at.
        #
        # latest.pt (above) is overwritten every checkpoint_every steps, so it
        # only ever holds the most recent state - there's no way to look back
        # and see how training progressed from that alone. milestones/ keeps a
        # numbered, un-overwritten world-model-only snapshot (no optimizer
        # state/actor/critic - much smaller, all that's needed to render
        # reconstructions) every milestone_every steps, so
        # generate_progression_video.py can show the same episode reconstructed
        # by each milestone in sequence - i.e. literally watching it learn. ---
        self._logdir = pathlib.Path(logdir)
        self._should_checkpoint = tools.Every(config.checkpoint_every)
        self._milestones_dir = self._logdir / "milestones"
        self._milestones_dir.mkdir(parents=True, exist_ok=True)
        self._should_milestone = tools.Every(config.milestone_every)
        self._thermal_guard = ThermalGuard(
            pause_c=config.thermal_pause_c,
            resume_c=config.thermal_resume_c,
            poll_interval_s=config.thermal_check_interval_s,
        )

        # Milestones are sparse (every 5000 steps by default) because a full
        # world-model snapshot is ~105MB - too much to save every update. But
        # a predicted joint_pos is just 6 floats, so we CAN record that every
        # single training update: observe the same fixed reference episode,
        # imagine forward, decode ONLY joint_pos (skip the decoder's CNN image
        # branch entirely - see _record_joint_pos_probe - since we don't need
        # images here, this is much cheaper than a full video_pred() call).
        # generate_joint_pos_timelapse_video.py turns this into a video with
        # one frame per training update instead of per milestone.
        if getattr(config, "joint_pos_trace", True):
            self._joint_pos_probe_batch = next(self._dataset)
            self._joint_pos_tracer = TraceWriter(
                self._logdir / "joint_pos_trace.bin", config.batch_length
            )
        self._joint_pos_context_steps = 5

    def _record_joint_pos_probe(self):
        try:
            with torch.no_grad():
                data = self._wm.preprocess(self._joint_pos_probe_batch)
                embed = self._wm.encoder(data)
                ex = slice(0, 1)
                c = self._joint_pos_context_steps
                states, _ = self._wm.dynamics.observe(
                    embed[ex, :c], data["action"][ex, :c], data["is_first"][ex, :c]
                )
                decoder_mlp = self._wm.heads["decoder"]._mlp
                context_jp = decoder_mlp(self._wm.dynamics.get_feat(states))["joint_pos"].mode()
                init = {k: v[:, -1] for k, v in states.items()}
                prior = self._wm.dynamics.imagine_with_action(data["action"][ex, c:], init)
                imagined_jp = decoder_mlp(self._wm.dynamics.get_feat(prior))["joint_pos"].mode()
                trajectory = torch.cat([context_jp, imagined_jp], dim=1)
            # self._step (env step) only advances once per __call__, but many
            # training updates can happen within a single call (e.g. the
            # pretrain burst on the very first call) - self._update_count
            # increments once per actual update, so it's the correct unique
            # x-axis for this trace, not self._step.
            self._joint_pos_tracer.write(self._update_count, trajectory.detach().cpu().numpy()[0])
        except Exception as e:
            print(f"[joint_pos trace] FAILED at step {self._step}: {e}", flush=True)

    def _save_checkpoint(self):
        try:
            items_to_save = {
                "agent_state_dict": self.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(self),
            }
            torch.save(items_to_save, self._logdir / "latest.pt")
            print(f"[checkpoint] saved at step {self._step}", flush=True)
        except Exception as e:
            print(f"[checkpoint] FAILED to save at step {self._step}: {e}", flush=True)

    def _save_milestone(self):
        try:
            path = self._milestones_dir / f"step_{self._step}.pt"
            if getattr(self._config, "milestone_full_agent", False):
                torch.save({"agent_state_dict": self.state_dict()}, path)     # world model + actor + critic
            else:
                torch.save(self._wm.state_dict(), path)
            print(f"[milestone] saved {path}", flush=True)
        except Exception as e:
            print(f"[milestone] FAILED to save at step {self._step}: {e}", flush=True)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            self._thermal_guard.check()
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                if getattr(self._config, "joint_pos_trace", True):
                    self._record_joint_pos_probe()
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log and self._should_log_video(step):
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)
            if self._should_checkpoint(step):
                self._save_checkpoint()
            if self._should_milestone(step):
                self._save_milestone()

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def determine_checkpoint_source(logdir, init_weights):
    """Decides which checkpoint (if any) to seed the agent's weights from:
    'resume' (same-dir latest.pt - also restores optimizer state/pretrain
    flag), 'init_weights' (a previous curriculum stage's latest.pt - weights
    only, everything else starts fresh), or 'fresh' (random init). Pure path
    logic, no torch I/O, so it's testable without a real checkpoint file."""
    latest = pathlib.Path(logdir) / "latest.pt"
    if latest.exists():
        return "resume", latest
    if init_weights:
        return "init_weights", pathlib.Path(init_weights)
    return "fresh", None


def read_eval_history(metrics_path):
    """Every eval_return logged to this run's metrics.jsonl, oldest first.
    Read from disk rather than from the Logger, which clears its scalars as
    soon as it writes them (tools.simulate writes right after an eval) - and so
    the history survives a crash/resume. Unparseable lines (e.g. a write cut
    off by a crash) are skipped."""
    history = []
    try:
        lines = pathlib.Path(metrics_path).read_text().splitlines()
    except FileNotFoundError:
        return history
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval_return" in entry:
            history.append(float(entry["eval_return"]))
    return history


def read_best_eval(best_json_path):
    """(eval_return, step) of the best-scoring evaluation so far, or (None, None)."""
    try:
        d = json.loads(pathlib.Path(best_json_path).read_text())
        return float(d["eval_return"]), int(d["step"])
    except (FileNotFoundError, KeyError, ValueError):
        return None, None


def is_new_best(current, best):
    """True if `current` beats the best evaluation so far (always True for the first one)."""
    return best is None or current > best


def compute_early_stop(eval_return_history, patience, min_improvement, step=None, min_steps=0):
    """Pure decision logic for check_early_stop(): compares the best eval
    in the last `patience` evals against the eval from one window before
    that. Returns (should_stop, improvement_or_None) - improvement is None
    when there isn't yet enough history to decide, or `step` is still below
    `min_steps` (too early to call a plateau)."""
    if not patience or len(eval_return_history) <= patience:
        return False, None
    if step is not None and step < min_steps:
        return False, None
    baseline = eval_return_history[-patience - 1]
    recent_best = max(eval_return_history[-patience:])
    improvement = (recent_best - baseline) / max(abs(baseline), 1e-6)
    return improvement < min_improvement, improvement


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "so101":
        from envs.so101 import SO101Dreamer

        env = SO101Dreamer(reward_stage=config.reward_stage, seed=config.seed + id)
    elif suite == "so101state":
        from envs.so101_state import SO101StateDreamer

        env = SO101StateDreamer(reward_stage=config.reward_stage, difficulty=config.env_difficulty,
                                seed=config.seed + id, max_episode_steps=config.time_limit)
    else:
        raise NotImplementedError(suite)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        if getattr(config, "expert_episodes", 0) and (count_steps(config.traindir) == 0 or getattr(config, "expert_each_chunk", False)):
            from so101_env.expert import ScriptedExpert

            expert = ScriptedExpert(train_envs[0].sim)

            def expert_agent(o, d, s):
                if d[0]:
                    expert.reset()
                action = torch.tensor(expert.act(), dtype=torch.float32).unsqueeze(0)
                return {"action": action, "logprob": torch.zeros(1)}, None

            print(f"Collecting {config.expert_episodes} scripted-solver episodes.")
            tools.simulate(expert_agent, train_envs, train_eps, config.traindir, logger,
                           limit=config.dataset_size, episodes=config.expert_episodes)
            prefill = max(0, config.prefill - count_steps(config.traindir))
            logger.step += count_steps(config.traindir) * config.action_repeat
        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
        logdir,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    # Curriculum warm-start (init_weights): seed a fresh run's weights (world
    # model + actor + critic) from a previous stage's final checkpoint, e.g.
    # reach -> grasp -> lift (see train_curriculum.py). Everything else
    # (optimizer state, replay buffer, pretrain burst) starts fresh for this
    # stage, since the reward signal - and therefore what the actor/critic
    # should optimize for - has changed. A same-dir latest.pt (resuming this
    # exact stage after an interruption) always takes priority over that.
    checkpoint_source, checkpoint_path = determine_checkpoint_source(logdir, config.init_weights)
    if checkpoint_source == "resume":
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
    elif checkpoint_source == "init_weights":
        print(f"Seeding initial weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        agent.load_state_dict(checkpoint["agent_state_dict"])

    # Early stopping: track eval_return and stop once it plateaus, instead of
    # always running the full --steps. Set early_stop_patience to 0/null to
    # disable and always run the full step count. The history is this
    # logdir's metrics.jsonl, so it carries over when a run is resumed.
    def check_early_stop():
        if not config.early_stop_patience or config.eval_episode_num <= 0:
            return False
        history = read_eval_history(logdir / "metrics.jsonl")
        if not history:
            return False
        current = history[-1]
        should_stop, improvement = compute_early_stop(
            history, config.early_stop_patience, config.early_stop_min_improvement,
            step=agent._step, min_steps=config.early_stop_min_steps,
        )
        if improvement is not None:
            print(
                f"[early stop check] eval_return {current:.2f}, best of last "
                f"{config.early_stop_patience} evals improved {improvement * 100:.1f}% "
                f"over the one before that (threshold {config.early_stop_min_improvement * 100:.1f}%)"
            )
        return should_stop

    # One chunk only (run3_train.py): evaluate, train eval_every steps, save, exit - from wherever the step counter is now.
    if getattr(config, "one_chunk", False):
        config.steps = agent._step
    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
            # Keep the best-scoring weights (policy + world model + critic): the last ones aren't
            # necessarily the best - run 1's lift stage ended on weights worse than ones it had passed.
            history = read_eval_history(logdir / "metrics.jsonl")
            if history:
                best_return, _ = read_best_eval(logdir / "best.json")
                if is_new_best(history[-1], best_return):
                    torch.save({"agent_state_dict": agent.state_dict()}, logdir / "best.pt")
                    (logdir / "best.json").write_text(json.dumps({"eval_return": history[-1], "step": agent._step}))
                    print(f"[best] new best eval_return {history[-1]:.1f} at step {agent._step} -> best.pt")
            if check_early_stop():
                print(f"[early stop] eval_return plateaued - stopping at step {agent._step}")
                items_to_save = {
                    "agent_state_dict": agent.state_dict(),
                    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                }
                torch.save(items_to_save, logdir / "latest.pt")
                break
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
