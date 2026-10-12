# Platform-Fighter-Agent

A small Smash-style platform fighter built as a training environment for reinforcement-learning agents, drawn in the vector style of 1979 Asteroids.

Everything that can collide is a convex polygon, and the renderer draws those same polygons as outlines. A character's outline is its hurtbox. An attack's outline is its hitbox: it grows in yellow during the wind-up, when it cannot hit, and turns red at full size on the frames it can.

## Status

The environment is playable and tested. There is no training code yet.

## Build and run

Requires a C++17 compiler and `make`. The viewer also needs GLFW (`brew install glfw`).

```
make            # library, tests and viewer
make test       # run the simulation tests
make play       # play against the scripted bot
make sim        # headless only: library and tests, no OpenGL needed
```

Viewer options: `build/pf_play --p1 human|bot|idle|agent:FILE --p2 human|bot|idle|agent:FILE`. `agent:FILE` plays a trained policy exported with `python/export_policy.py`, for example `build/pf_play --p1 agent:models/bot-champion.pfp --p2 bot`.

| | Move | Jump | Down | Light | Heavy | Shield |
|---|---|---|---|---|---|---|
| Player 1 (white) | A / D | W | S | F | G | H |
| Player 2 (cyan) | Left / Right | Up | Down | , | . | / |

R restarts, Esc quits. `--screenshot out.bmp --frames N [--wait hitbox|windup|shield]` renders one frame off-screen.

## Rules

- **Stage:** one solid main platform, an inverted trapezoid with bevelled undersides, and three soft platforms above it. Soft platforms can be jumped through from below and dropped through by holding down.
- **Damage:** hits add to a damage number. Knockback grows with the target's damage, so high-damage characters fly further.
- **Stocks:** leaving the dashed blast zone costs a stock. Three stocks each; last one standing wins. A three-minute time limit breaks ties by stocks, then by lower damage.
- **Movement:** run, jump, one air jump, fast-fall (down while falling).
- **Shield:** hold the shield button on the ground for a bubble that blocks any attack touching it. It drains while held and when hit, and shrinks as it drains, leaving the parts of the body outside it open to hits. If it runs out it breaks and stuns its owner. It refills when down. You can jump out of it but not move or attack.
- **Air shield:** pressing shield in the air raises the same bubble for a fixed 0.3 seconds and cancels vertical speed at that moment, then gravity resumes. Once per airtime; landing restores it.
- **Character:** an equilateral triangle on top of a semicircle. It leans toward the direction it faces. Grounded characters that overlap push each other apart.

### Attacks

There are two attack buttons. The movement state when the button is pressed picks the move:

| State | Light | Heavy |
|---|---|---|
| Standing | jab: fast, short | smash: longest wind-up, the main knockout move |
| Running | dash poke: long reach | sweep: low and wide, launches upward |
| Rising | up spike: above the head | uppercut: lifts the user, once per airtime (recovery move) |
| Falling | down spike: below, knocks downward | dive: plunges until landing |

Moves with larger hitboxes have longer wind-ups: the four light moves take 3 to 5 frames, the dive 8, the sweep 10, the uppercut 12 and the smash 16.

The two running attacks stop the character dead for the wind-up, then thrust it forward as the hitbox appears.

The two running attacks share a one-second cooldown; pressing attack while running during it gives the standing move instead.

Getting hit locks out all input for a short knockback period: about a sixth of a second for the weakest hit, longer for harder ones.

Move timings, damage and hitbox shapes are in the table at the top of `sim/src/sim.cpp`.

## Layout

- `sim/` is the simulation: no dependencies, deterministic, fixed 60 Hz step. `pf/sim.h` is the C++ interface, `pf/c_api.h` the flat C interface, `pf/bot.h` a scripted opponent.
- `app/main.cpp` is the OpenGL 3.3 viewer.
- `python/pfenv.py` wraps the C interface with ctypes: `reset()`, `step(a0, a1)`, `step_vs_bot(a)`.
- `tests/sim_tests.cpp` covers the rules and prints a speed benchmark.

## Training interface

Training runs headless: `make sim` builds `build/libpfenv.so` (`.dylib` on macOS) with no OpenGL or display, and `python/pfenv.py` loads it.

```python
from pfenv import VecPlatformFighter

env = VecPlatformFighter(num_envs=256, opponent="bot", frame_skip=4)  # or opponent="self"
obs = env.reset()                       # (256, 49) float32
obs, rewards, terminated, truncated, winners = env.step(actions)   # actions: (256, 6) ints
```

- **Action:** six integers per agent, a MultiDiscrete `[3, 2, 2, 2, 2, 2]`: move (0 left, 1 none, 2 right), jump, down, light, heavy, shield. Each step's buttons count as new presses.
- **Observation:** 49 floats from the acting player's point of view (own state, opponent state, relative position, time left); see `Game::observe`. Not mirrored for player 2.
- **Reward:** (damage dealt − damage taken) / 100, +1 per stock taken, −1 per stock lost.
- **Distance penalty (optional):** `distance_penalty=c` costs each agent `c` per frame while the fighters are more than 8 units apart, fading to nothing within 2, to discourage passive play. Off by default here; `train_ppo.py` uses `1e-4`. With it on, rewards are not zero-sum.
- **Episode end:** `terminated` when a player runs out of stocks, `truncated` on the time limit. Matches reset automatically.
- **Self-play:** `opponent="self"` gives two agents per match; shapes gain an agent axis, e.g. observations `(N, 2, 49)`.
- **Arrays:** numpy views when numpy is installed, ctypes arrays otherwise. They are overwritten by the next step.

```
make sim && python3 python/pfenv.py
```

runs random agents in both modes as a smoke test and prints throughput (about 1.1 million frames per second on one core of the lab login host).

`PlatformFighterEnv` in the same file steps a single match frame by frame, with held-button semantics identical to the viewer.

## Compute

Training can run on the UofT Teaching Labs GPU cluster. Usage notes: `../Claude-Instructions/teach-cs-gpu-cluster.md`. Use `make sim` there, since the nodes have no display.

## Related

Earlier work on the UTMIST AI² Warehouse Brawl environment lives on the lab server at `~/Projects/UTMIST-AI2` (remote `github.com/Finay/UTMIST-AI2`).
