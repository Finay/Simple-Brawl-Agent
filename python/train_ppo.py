"""PPO for the platform fighter.

The simulation runs on the CPU in one batched C call per step; the policy,
rollout storage, advantage computation and updates all stay on the GPU, so the
only per-step transfers are observation batches up and action batches down.

    python3 python/train_ppo.py --opponent bot
    python3 python/train_ppo.py --opponent self      # latest vs latest
    python3 python/train_ppo.py --opponent mixed     # scripted bot + frozen past checkpoints

Build the simulation first with `make sim`.
"""
import argparse
import collections
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfenv import VecPlatformFighter  # noqa: E402

BUTTONS = ("jump", "down", "light", "heavy", "shield")
ACTION_SIZE = 1 + len(BUTTONS)
# Start with buttons mostly released: an untrained policy that holds shield or
# mashes jump half the time never gets to see what attacking does.
BUTTON_BIAS = (-2.0, -2.0, -1.0, -1.5, -2.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--opponent", choices=("bot", "self", "mixed"), default="bot")
    p.add_argument("--num-envs", type=int, default=1024, help="matches, for --opponent bot or self")
    p.add_argument("--bot-envs", type=int, default=256, help="mixed: matches against the scripted bot")
    p.add_argument("--pool-envs", type=int, default=768, help="mixed: matches against frozen checkpoints")
    p.add_argument("--pool-groups", type=int, default=8, help="mixed: frozen opponents in play at once")
    p.add_argument("--pool-max", type=int, default=40, help="mixed: snapshots kept")
    p.add_argument("--pool-init", nargs="*", default=[], help="mixed: checkpoints to seed the pool with")
    p.add_argument("--snapshot-every", type=int, default=50, help="mixed: updates between new snapshots")
    p.add_argument("--resample-every", type=int, default=10, help="mixed: updates between opponent reshuffles")
    p.add_argument("--latest-prob", type=float, default=0.3, help="mixed: chance a group gets the newest snapshot")
    p.add_argument("--distance-penalty", type=float, default=1e-4,
                   help="per-frame cost for being far from the opponent; 0 = off")
    p.add_argument("--penalty-near", type=float, default=2.0, help="no cost within this distance")
    p.add_argument("--penalty-far", type=float, default=8.0, help="full cost beyond this distance")
    p.add_argument("--rollout", type=int, default=64, help="steps per env between updates")
    p.add_argument("--frame-skip", type=int, default=4, help="simulation frames per agent step")
    p.add_argument("--total-steps", type=int, default=300_000_000, help="agent steps to train for")
    p.add_argument("--time-limit-min", type=float, default=0, help="stop cleanly after this long; 0 = none")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--init", default=None, help="checkpoint to start the learner from")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-every", type=int, default=25, help="updates between bot evaluations (self-play only)")
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=100, help="updates between checkpoints")
    p.add_argument("--wandb-entity", default="kianbenner")
    p.add_argument("--wandb-project", default="pf-agent")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def mlp(sizes, out_gain):
    layers = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(linear.weight, out_gain if last else 2 ** 0.5)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not last:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def sample_actions(logits):
    move = torch.multinomial(F.softmax(logits[:, :3], dim=-1), 1)
    buttons = (torch.rand_like(logits[:, 3:]) < torch.sigmoid(logits[:, 3:])).long()
    return torch.cat([move, buttons], dim=1)


class Agent(nn.Module):
    """Separate policy and value networks. The action is one 3-way choice
    (move left / none / right) and five independent buttons."""

    def __init__(self, obs_size, hidden):
        super().__init__()
        self.policy = mlp([obs_size, hidden, hidden, 3 + len(BUTTONS)], 0.01)
        self.value = mlp([obs_size, hidden, hidden, 1], 1.0)
        with torch.no_grad():
            self.policy[-1].bias[3:] = torch.tensor(BUTTON_BIAS)

    def _log_prob_entropy(self, logits, actions):
        move_logp = F.log_softmax(logits[:, :3], dim=-1)
        button_logits = logits[:, 3:]
        pressed = actions[:, 1:].float()
        logp = move_logp.gather(1, actions[:, :1]).squeeze(1)
        logp = logp - F.binary_cross_entropy_with_logits(button_logits, pressed, reduction="none").sum(1)
        p = torch.sigmoid(button_logits)
        entropy = -(move_logp.exp() * move_logp).sum(1)
        entropy = entropy + F.binary_cross_entropy_with_logits(button_logits, p, reduction="none").sum(1)
        return logp, entropy

    @torch.no_grad()
    def act(self, obs):
        logits = self.policy(obs)
        actions = sample_actions(logits)
        logp, _ = self._log_prob_entropy(logits, actions)
        return actions, logp, self.value(obs).squeeze(1)

    def evaluate(self, obs, actions):
        logp, entropy = self._log_prob_entropy(self.policy(obs), actions)
        return logp, entropy, self.value(obs).squeeze(1)


def make_env(args, num_envs, opponent):
    return VecPlatformFighter(num_envs, opponent=opponent, frame_skip=args.frame_skip,
                              distance_penalty=args.distance_penalty, penalty_near=args.penalty_near,
                              penalty_far=args.penalty_far)


def to_device(array, device):
    # The env's arrays are views of buffers the next step overwrites, so always copy.
    return torch.from_numpy(array).to(device, copy=True)


def step_env(env, actions, device):
    obs, rewards, terminated, truncated, winners = env.step(actions.to(torch.int32).cpu().numpy())
    done = np.logical_or(terminated, truncated)
    return (to_device(obs, device).view(-1, env.obs_size), to_device(rewards, device).view(-1),
            to_device(done, device), winners.copy())


@torch.no_grad()
def evaluate_vs_bot(agent, env, device, episodes):
    """Win rate of player 1 (the agent) against the scripted bot."""
    obs = to_device(env.reset(), device).view(-1, env.obs_size)
    wins = finished = 0
    for _ in range(4000):
        actions, _, _ = agent.act(obs)
        obs, _, done, winners = step_env(env, actions, device)
        done = done.cpu().numpy()
        finished += int(done.sum())
        wins += int((winners[done] == 0).sum())
        if finished >= episodes:
            break
    return wins / max(finished, 1), finished


class SimpleArena:
    """All matches against the bot, or all latest-vs-latest self-play.

    An arena presents one row per character the learner controls. step() returns
    the next observations, rewards, which rows' matches ended, and (for those
    rows) whether the learner won."""

    def __init__(self, args, device):
        self.device = device
        self.env = make_env(args, args.num_envs, args.opponent)
        self.agents = self.env.num_agents
        self.obs_size = self.env.obs_size
        self.batch = self.env.num_envs * self.agents
        self.kinds = [args.opponent] * self.batch
        self.player = torch.arange(self.batch, device=device) % self.agents

    def reset(self):
        return to_device(self.env.reset(), self.device).view(self.batch, self.obs_size)

    def step(self, actions):
        obs, reward, done, winners = step_env(self.env, actions, self.device)
        won = torch.from_numpy(winners).to(self.device).repeat_interleave(self.agents) == self.player
        return obs, reward, done.repeat_interleave(self.agents), won

    def after_update(self, update, agent):
        return {}


class FrozenOpponents:
    """Several frozen policy networks evaluated in one batched pass."""

    def __init__(self, groups, obs_size, hidden, device):
        sizes = [obs_size, hidden, hidden, 3 + len(BUTTONS)]
        self.weights = [torch.zeros(groups, sizes[i + 1], sizes[i], device=device) for i in range(3)]
        self.biases = [torch.zeros(groups, 1, sizes[i + 1], device=device) for i in range(3)]

    def assign(self, group, snapshot):
        for i in range(3):
            self.weights[i][group] = snapshot[f"{2 * i}.weight"]
            self.biases[i][group, 0] = snapshot[f"{2 * i}.bias"]

    @torch.no_grad()
    def act(self, obs):
        """obs: (groups, rows per group, obs_size) -> actions (groups * rows, 6)."""
        x = obs
        for i in range(3):
            x = torch.baddbmm(self.biases[i], x, self.weights[i].transpose(1, 2))
            if i < 2:
                x = torch.tanh(x)
        return sample_actions(x.reshape(-1, x.shape[-1]))


class MixedArena:
    """Some matches against the scripted bot and the rest against a pool of
    frozen snapshots of earlier learners. Against the pool the learner plays
    player 1 in half the matches and player 2 in the other half, and only its
    own side's experience is used."""

    def __init__(self, args, device):
        if args.pool_envs % (2 * args.pool_groups) != 0:
            raise ValueError("--pool-envs must be a multiple of 2 * --pool-groups")
        self.args, self.device = args, device
        self.bot_env = make_env(args, args.bot_envs, "bot")
        self.pool_env = make_env(args, args.pool_envs, "self")
        self.obs_size = self.bot_env.obs_size
        self.batch = args.bot_envs + args.pool_envs
        self.kinds = ["bot"] * args.bot_envs + ["pool"] * args.pool_envs
        self.rows = torch.arange(args.pool_envs, device=device)
        self.side = self.rows % 2  # which player the learner controls in each pool match
        self.opponents = FrozenOpponents(args.pool_groups, self.obs_size, args.hidden, device)
        self.pool = []
        self.opp_obs = None
        self.rng = random.Random(args.seed)

    def add_snapshot(self, policy_state):
        self.pool.append({k: v.detach().clone() for k, v in policy_state.items()})
        if len(self.pool) > self.args.pool_max:
            # Keep the oldest entry as a fixed reference; drop a random middle one.
            self.pool.pop(self.rng.randrange(1, len(self.pool) - 1))

    def resample(self):
        for group in range(self.args.pool_groups):
            newest = self.rng.random() < self.args.latest_prob
            self.opponents.assign(group, self.pool[-1] if newest else self.rng.choice(self.pool))

    def _split(self, pool_obs):
        pool_obs = pool_obs.view(self.args.pool_envs, 2, self.obs_size)
        self.opp_obs = pool_obs[self.rows, 1 - self.side]
        return pool_obs[self.rows, self.side]

    def reset(self):
        bot_obs = to_device(self.bot_env.reset(), self.device)
        mine = self._split(to_device(self.pool_env.reset(), self.device))
        return torch.cat([bot_obs, mine])

    def step(self, actions):
        n_bot = self.args.bot_envs
        bot_obs, bot_reward, bot_done, bot_winners = step_env(self.bot_env, actions[:n_bot], self.device)

        opp_actions = self.opponents.act(self.opp_obs.view(self.args.pool_groups, -1, self.obs_size))
        both = torch.zeros(self.args.pool_envs, 2, ACTION_SIZE, dtype=torch.long, device=self.device)
        both[self.rows, self.side] = actions[n_bot:]
        both[self.rows, 1 - self.side] = opp_actions
        pool_obs, pool_reward, pool_done, pool_winners = step_env(self.pool_env, both, self.device)
        mine = self._split(pool_obs)
        my_reward = pool_reward.view(-1, 2)[self.rows, self.side]

        won = torch.cat([torch.from_numpy(bot_winners).to(self.device) == 0,
                         torch.from_numpy(pool_winners).to(self.device) == self.side])
        return (torch.cat([bot_obs, mine]), torch.cat([bot_reward, my_reward]),
                torch.cat([bot_done, pool_done]), won)

    def after_update(self, update, agent):
        if update % self.args.snapshot_every == 0:
            self.add_snapshot(agent.policy.state_dict())
        if update % self.args.resample_every == 0:
            self.resample()
        return {"charts/pool_size": len(self.pool)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_float32_matmul_precision("high")

    arena = MixedArena(args, device) if args.opponent == "mixed" else SimpleArena(args, device)
    obs_size, batch = arena.obs_size, arena.batch
    eval_env = None
    if args.opponent == "self":
        eval_env = VecPlatformFighter(args.eval_envs, opponent="bot", frame_skip=args.frame_skip)
    steps_per_update = batch * args.rollout
    num_updates = max(1, args.total_steps // steps_per_update)
    minibatch = steps_per_update // args.minibatches

    agent = Agent(obs_size, args.hidden).to(device)
    if args.init:
        agent.load_state_dict(torch.load(args.init, map_location=device)["model"])
    if args.opponent == "mixed":
        for path in args.pool_init:
            model = torch.load(path, map_location=device)["model"]
            arena.add_snapshot({k[len("policy."):]: v for k, v in model.items() if k.startswith("policy.")})
        if not arena.pool:
            arena.add_snapshot(agent.policy.state_dict())
        arena.resample()
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.run_name,
                         config=vars(args))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tag = run.id if run else time.strftime("%Y%m%d-%H%M%S")
    latest_path = os.path.join(args.checkpoint_dir, f"ppo-{tag}-latest.pt")

    def save(path, steps):
        torch.save({"model": agent.state_dict(), "args": vars(args), "obs_size": obs_size,
                    "hidden": args.hidden, "steps": steps}, path)

    print(f"device {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}), "
          f"opponent {args.opponent}, {batch} learner rows, {steps_per_update} steps/update, "
          f"{num_updates} updates, frame skip {args.frame_skip}", flush=True)

    obs_buf = torch.zeros(args.rollout, batch, obs_size, device=device)
    act_buf = torch.zeros(args.rollout, batch, ACTION_SIZE, dtype=torch.long, device=device)
    logp_buf = torch.zeros(args.rollout, batch, device=device)
    val_buf = torch.zeros(args.rollout, batch, device=device)
    rew_buf = torch.zeros(args.rollout, batch, device=device)
    done_buf = torch.zeros(args.rollout, batch, device=device)

    ep_return = torch.zeros(batch, device=device)
    ep_length = torch.zeros(batch, device=device)
    kinds = sorted(set(arena.kinds))
    kind_of_row = np.array([kinds.index(k) for k in arena.kinds])
    recent = {k: collections.deque(maxlen=2000) for k in kinds}  # (return, length, won) per finished match
    episodes_total = 0

    obs = arena.reset()
    global_step = 0
    start = time.time()

    for update in range(1, num_updates + 1):
        lr = args.lr * (1.0 - (update - 1) / num_updates)
        for group in optimizer.param_groups:
            group["lr"] = lr

        rollout_start = time.time()
        for t in range(args.rollout):
            actions, logp, value = agent.act(obs)
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, actions, logp, value
            obs, reward, done, won = arena.step(actions)
            rew_buf[t], done_buf[t] = reward, done
            ep_return += reward
            ep_length += 1
            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                rows = idx.cpu().numpy()
                results = zip(ep_return[idx].cpu().numpy(), ep_length[idx].cpu().numpy(), won[idx].cpu().numpy())
                for row, result in zip(rows, results):
                    recent[kinds[kind_of_row[row]]].append(result)
                episodes_total += len(rows)
                ep_return[idx] = 0
                ep_length[idx] = 0
        global_step += steps_per_update
        rollout_time = time.time() - rollout_start

        # Generalised advantage estimation, on the GPU.
        with torch.no_grad():
            next_value = agent.value(obs).squeeze(1)
            adv_buf = torch.zeros_like(rew_buf)
            last = torch.zeros(batch, device=device)
            for t in reversed(range(args.rollout)):
                alive = 1.0 - done_buf[t]
                delta = rew_buf[t] + args.gamma * next_value * alive - val_buf[t]
                last = delta + args.gamma * args.gae_lambda * alive * last
                adv_buf[t] = last
                next_value = val_buf[t]
            ret_buf = adv_buf + val_buf

        b_obs = obs_buf.view(-1, obs_size)
        b_act = act_buf.view(-1, ACTION_SIZE)
        b_logp, b_adv, b_ret, b_val = logp_buf.view(-1), adv_buf.view(-1), ret_buf.view(-1), val_buf.view(-1)

        update_start = time.time()
        clip_fracs = []
        for _ in range(args.epochs):
            order = torch.randperm(steps_per_update, device=device)
            for i in range(0, steps_per_update, minibatch):
                mb = order[i:i + minibatch]
                new_logp, entropy, new_value = agent.evaluate(b_obs[mb], b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                adv = b_adv[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
                value_loss = 0.5 * (new_value - b_ret[mb]).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_mean
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    clip_fracs.append(((ratio - 1).abs() > args.clip).float().mean())
                    approx_kl = ((ratio - 1) - (new_logp - b_logp[mb])).mean()
        update_time = time.time() - update_start

        elapsed = time.time() - start
        log = {
            "charts/learning_rate": lr,
            "charts/episodes": episodes_total,
            "losses/policy": policy_loss.item(),
            "losses/value": value_loss.item(),
            "losses/entropy": entropy_mean.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clip_fraction": torch.stack(clip_fracs).mean().item(),
            "losses/explained_variance": (1 - (b_ret - b_val).var() / (b_ret.var() + 1e-8)).item(),
            "perf/steps_per_second": steps_per_update / (rollout_time + update_time),
            "perf/frames_per_second": steps_per_update * args.frame_skip / (rollout_time + update_time),
            "perf/rollout_seconds": rollout_time,
            "perf/update_seconds": update_time,
            "perf/hours": elapsed / 3600,
        }
        rates = b_act[:, 1:].float().mean(0)
        for name, rate in zip(BUTTONS, rates.tolist()):
            log[f"actions/{name}_rate"] = rate
        log["actions/idle_move_rate"] = (b_act[:, 0] == 1).float().mean().item()
        # Win rates are over each opponent kind's last 2000 matches. Against
        # "self" it is ~0.5 by construction and says nothing about strength.
        for kind in kinds:
            if recent[kind]:
                results = np.array(recent[kind], dtype=np.float64)
                log[f"charts/episode_return_vs_{kind}"] = results[:, 0].mean()
                log[f"charts/episode_length_vs_{kind}"] = results[:, 1].mean()
                log[f"charts/win_rate_vs_{kind}"] = results[:, 2].mean()
        if eval_env is not None and (update % args.eval_every == 0 or update == num_updates):
            win_rate, played = evaluate_vs_bot(agent, eval_env, device, args.eval_envs)
            log["charts/win_rate_vs_bot"] = win_rate
            log["charts/eval_episodes"] = played
        log.update(arena.after_update(update, agent))
        if run:
            run.log(log, step=global_step)
        if update % 10 == 0 or update == 1 or update == num_updates:
            wins = " ".join(f"{k.split('/')[1]} {v:.3f}" for k, v in sorted(log.items()) if "win_rate" in k)
            print(f"update {update}/{num_updates} steps {global_step:,} {wins} "
                  f"entropy {log['losses/entropy']:.3f} sps {log['perf/steps_per_second']:,.0f} "
                  f"elapsed {elapsed / 60:.1f}m", flush=True)

        out_of_time = args.time_limit_min > 0 and elapsed > args.time_limit_min * 60
        if update % args.checkpoint_every == 0 or update == num_updates or out_of_time:
            save(latest_path, global_step)
        if out_of_time:
            print(f"time limit reached after {update} updates", flush=True)
            break

    save(latest_path, global_step)
    print(f"saved {latest_path}", flush=True)
    if run:
        import wandb
        artifact = wandb.Artifact(f"ppo-{tag}", type="model", metadata={"steps": global_step})
        artifact.add_file(latest_path)
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
