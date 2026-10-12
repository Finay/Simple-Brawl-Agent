"""Python bindings for the platform fighter simulation (ctypes, headless).

Build the shared library first with `make sim` from the project root. No
display or OpenGL is needed.

Two interfaces:

- `VecPlatformFighter`: many matches stepped per call, for training. Uses
  numpy arrays when numpy is installed, plain ctypes arrays otherwise.
- `PlatformFighterEnv`: one match, frame by frame, plain Python lists.
"""
import ctypes
import os
import sys

try:
    import numpy as np
except ImportError:  # numpy is optional
    np = None

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_NAME = "libpfenv.dylib" if sys.platform == "darwin" else "libpfenv.so"

_int_p = ctypes.POINTER(ctypes.c_int)
_float_p = ctypes.POINTER(ctypes.c_float)
_byte_p = ctypes.POINTER(ctypes.c_ubyte)


class _Status(ctypes.Structure):
    _fields_ = [
        ("frame", ctypes.c_int),
        ("done", ctypes.c_int),
        ("winner", ctypes.c_int),
        ("damage", ctypes.c_float * 2),
        ("stocks", ctypes.c_int * 2),
        ("damage_taken", ctypes.c_float * 2),
        ("damage_blocked", ctypes.c_float * 2),
        ("ko", ctypes.c_int * 2),
    ]


def _load(path=None):
    lib = ctypes.CDLL(path or os.path.join(_ROOT, "build", _LIB_NAME))
    lib.pf_create.restype = ctypes.c_void_p
    lib.pf_create.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.pf_destroy.argtypes = [ctypes.c_void_p]
    lib.pf_reset.argtypes = [ctypes.c_void_p]
    lib.pf_step.argtypes = [ctypes.c_void_p, _int_p]
    lib.pf_step_vs_bot.argtypes = [ctypes.c_void_p, ctypes.c_int, _int_p]
    lib.pf_obs_size.restype = ctypes.c_int
    lib.pf_observe.argtypes = [ctypes.c_void_p, ctypes.c_int, _float_p]
    lib.pf_get_status.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Status)]
    lib.pf_vec_create.restype = ctypes.c_void_p
    lib.pf_vec_create.argtypes = [ctypes.c_int] * 5
    lib.pf_vec_destroy.argtypes = [ctypes.c_void_p]
    lib.pf_vec_num_envs.argtypes = [ctypes.c_void_p]
    lib.pf_vec_agents.argtypes = [ctypes.c_void_p]
    lib.pf_vec_reset.argtypes = [ctypes.c_void_p, _float_p]
    lib.pf_vec_step.argtypes = [ctypes.c_void_p, _int_p, _float_p, _float_p, _byte_p, _byte_p, _int_p]
    return lib


class VecPlatformFighter:
    """A batch of matches stepped together, with automatic resets.

    opponent="bot": one agent per match (player 1) against the scripted bot.
    opponent="self": two agents per match, for self-play.

    An action is six ints per agent, matching `ACTION_NVEC`:
    move (0 left, 1 none, 2 right), jump, down, light, heavy, shield.
    Each step's buttons count as new presses. With frame_skip > 1 the action
    is held for that many frames (one frame is 1/60 s).

    Shapes, with N = num_envs: observations (N, obs_size), rewards (N,) and
    actions (N, 6) against the bot; (N, 2, obs_size), (N, 2) and (N, 2, 6) in
    self-play. terminated, truncated and winners are (N,).

    The reward is (damage dealt - damage taken) / 100, plus 1 for each stock
    taken and minus 1 for each stock lost. Observation features are listed in
    `Game::observe` in sim/src/sim.cpp.

    Returned arrays are views of buffers that the next step overwrites; copy
    them if they need to outlive it. When a match ends its observation is
    already the first of the next match.
    """

    ACTION_NVEC = (3, 2, 2, 2, 2, 2)

    def __init__(self, num_envs, opponent="bot", frame_skip=1, stocks=3, max_frames=60 * 60 * 3,
                 lib_path=None):
        if opponent not in ("bot", "self"):
            raise ValueError("opponent must be 'bot' or 'self'")
        self._lib = _load(lib_path)
        self._vec = self._lib.pf_vec_create(num_envs, stocks, max_frames, frame_skip,
                                            1 if opponent == "bot" else 0)
        self.num_envs = self._lib.pf_vec_num_envs(self._vec)
        self.num_agents = self._lib.pf_vec_agents(self._vec)
        self.obs_size = self._lib.pf_obs_size()
        self.frame_skip = frame_skip

        n, a = self.num_envs, self.num_agents
        self._obs = (ctypes.c_float * (n * a * self.obs_size))()
        self._rewards = (ctypes.c_float * (n * a))()
        self._terminated = (ctypes.c_ubyte * n)()
        self._truncated = (ctypes.c_ubyte * n)()
        self._winners = (ctypes.c_int * n)()
        self._actions = (ctypes.c_int * (n * a * 6))()

        self.obs, self.rewards = self._obs, self._rewards
        self.terminated, self.truncated, self.winners = self._terminated, self._truncated, self._winners
        if np is not None:
            agent_shape = (n,) if a == 1 else (n, a)
            self.obs = np.ctypeslib.as_array(self._obs).reshape(agent_shape + (self.obs_size,))
            self.rewards = np.ctypeslib.as_array(self._rewards).reshape(agent_shape)
            self.terminated = np.ctypeslib.as_array(self._terminated).view(np.bool_)
            self.truncated = np.ctypeslib.as_array(self._truncated).view(np.bool_)
            self.winners = np.ctypeslib.as_array(self._winners)
            self._actions_np = np.ctypeslib.as_array(self._actions)

    def __del__(self):
        if getattr(self, "_vec", None):
            self._lib.pf_vec_destroy(self._vec)
            self._vec = None

    def reset(self):
        self._lib.pf_vec_reset(self._vec, self._obs)
        return self.obs

    def step(self, actions):
        """actions: an int array of the shape above, or any flat sequence of that many ints."""
        if np is not None:
            self._actions_np[:] = np.asarray(actions).reshape(-1)
        else:
            self._actions[:] = list(actions)
        self._lib.pf_vec_step(self._vec, self._actions, self._obs, self._rewards, self._terminated,
                              self._truncated, self._winners)
        return self.obs, self.rewards, self.terminated, self.truncated, self.winners


class PlatformFighterEnv:
    """One match, one frame per step. step() returns per-player observations and rewards.

    An action is six ints: (move_x, jump, down, light, heavy, shield), where
    move_x is -1, 0 or 1 (unlike the vector interface) and buttons act on the
    frame they go from 0 to 1. The reward is the same as the vector interface.
    """

    ACTION_SIZE = 6

    def __init__(self, stocks=3, max_frames=60 * 60 * 3, lib_path=None):
        self._lib = _load(lib_path)
        self._env = self._lib.pf_create(stocks, max_frames)
        self.obs_size = self._lib.pf_obs_size()
        self._obs_buf = (ctypes.c_float * self.obs_size)()
        self._status = _Status()

    def __del__(self):
        if getattr(self, "_env", None):
            self._lib.pf_destroy(self._env)
            self._env = None

    def reset(self):
        self._lib.pf_reset(self._env)
        return self.observe(0), self.observe(1)

    def observe(self, player):
        self._lib.pf_observe(self._env, player, self._obs_buf)
        return list(self._obs_buf)

    def step(self, action0, action1):
        """Advance one frame (1/60 s) with both players' actions."""
        acts = (ctypes.c_int * 12)(*action0, *action1)
        self._lib.pf_step(self._env, acts)
        return self._result()

    def step_vs_bot(self, action, player=0):
        """Advance one frame with the scripted bot playing the other side."""
        acts = (ctypes.c_int * 6)(*action)
        self._lib.pf_step_vs_bot(self._env, player, acts)
        return self._result()

    def _result(self):
        self._lib.pf_get_status(self._env, self._status)
        st = self._status
        rewards = []
        for p in (0, 1):
            o = 1 - p
            rewards.append((st.damage_taken[o] - st.damage_taken[p]) / 100.0 + st.ko[o] - st.ko[p])
        info = {
            "frame": st.frame,
            "winner": st.winner,
            "damage": (st.damage[0], st.damage[1]),
            "stocks": (st.stocks[0], st.stocks[1]),
        }
        return (self.observe(0), self.observe(1)), tuple(rewards), bool(st.done), info


def _smoke_test():
    """Random agents in both modes: checks shapes and prints throughput."""
    import random
    import time

    rng = random.Random(0)
    for opponent, num_envs, steps in (("bot", 64, 2000), ("self", 64, 2000)):
        env = VecPlatformFighter(num_envs, opponent=opponent, frame_skip=4)
        obs = env.reset()
        count = env.num_envs * env.num_agents
        episodes, wins, total = 0, 0, 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            if np is not None:
                actions = np.random.randint(0, env.ACTION_NVEC, size=(count, 6))
            else:
                actions = [rng.randrange(k) for _ in range(count) for k in env.ACTION_NVEC]
            obs, rewards, terminated, truncated, winners = env.step(actions)
            for e in range(env.num_envs):
                if terminated[e] or truncated[e]:
                    episodes += 1
                    wins += winners[e] == 0
            total += float(sum(rewards)) if np is None else float(rewards.sum())
        dt = time.perf_counter() - t0
        frames = steps * env.num_envs * env.frame_skip
        shape = getattr(obs, "shape", (len(obs),))
        print(f"vs {opponent}: obs {shape}, {episodes} matches, player 1 won {wins}, "
              f"reward sum {total:+.1f}, {frames / dt / 1e6:.2f}M frames/s "
              f"({steps * env.num_envs / dt:,.0f} env steps/s)")
    print("numpy:", np.__version__ if np is not None else "not installed (ctypes arrays returned)")


if __name__ == "__main__":
    _smoke_test()
