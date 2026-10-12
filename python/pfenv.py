"""Python binding for the platform fighter simulation (ctypes, no dependencies).

Build the shared library first with `make sim` from the project root.

An action is a sequence of five ints: (move_x, jump, down, light, heavy),
where move_x is -1, 0 or 1 and the rest are 0 or 1.
"""
import ctypes
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_NAME = "libpfenv.dylib" if sys.platform == "darwin" else "libpfenv.so"


class _Status(ctypes.Structure):
    _fields_ = [
        ("frame", ctypes.c_int),
        ("done", ctypes.c_int),
        ("winner", ctypes.c_int),
        ("damage", ctypes.c_float * 2),
        ("stocks", ctypes.c_int * 2),
        ("damage_taken", ctypes.c_float * 2),
        ("ko", ctypes.c_int * 2),
    ]


def _load(path=None):
    lib = ctypes.CDLL(path or os.path.join(_ROOT, "build", _LIB_NAME))
    lib.pf_create.restype = ctypes.c_void_p
    lib.pf_create.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.pf_destroy.argtypes = [ctypes.c_void_p]
    lib.pf_reset.argtypes = [ctypes.c_void_p]
    lib.pf_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.pf_step_vs_bot.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.pf_obs_size.restype = ctypes.c_int
    lib.pf_observe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.pf_get_status.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Status)]
    return lib


class PlatformFighterEnv:
    """Two-player environment. step() returns per-player observations and rewards.

    The default reward is zero-sum: damage dealt minus damage taken (as a
    fraction of 100%), plus 1 for taking a stock and minus 1 for losing one.
    """

    ACTION_SIZE = 5

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
        acts = (ctypes.c_int * 10)(*action0, *action1)
        self._lib.pf_step(self._env, acts)
        return self._result()

    def step_vs_bot(self, action, player=0):
        """Advance one frame with the scripted bot playing the other side."""
        acts = (ctypes.c_int * 5)(*action)
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


if __name__ == "__main__":
    import random
    import time

    env = PlatformFighterEnv()
    env.reset()
    rng = random.Random(0)
    steps, total, t0 = 0, 0.0, time.perf_counter()
    done, info = False, {}
    while not done:
        action = (rng.choice((-1, 0, 1)), rng.random() < 0.1, 0, rng.random() < 0.15, rng.random() < 0.05)
        (_obs0, _obs1), rewards, done, info = env.step_vs_bot([int(a) for a in action])
        total += rewards[0]
        steps += 1
    dt = time.perf_counter() - t0
    print(f"random agent vs bot: {steps} frames, return {total:+.2f}, info {info}")
    print(f"observation size {env.obs_size}, {steps / dt:,.0f} steps/s through ctypes")
