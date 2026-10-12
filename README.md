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

Viewer options: `build/pf_play --p1 human|bot|idle --p2 human|bot|idle`.

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
| Running | dash poke: long reach, keeps momentum | sweep: low and wide, launches upward |
| Rising | up spike: above the head | uppercut: lifts the user, once per airtime (recovery move) |
| Falling | down spike: below, knocks downward | dive: plunges until landing |

Moves with larger hitboxes have longer wind-ups: the four light moves take 3 to 5 frames, the dive 8, the sweep 10, the uppercut 12 and the smash 16.

Move timings, damage and hitbox shapes are in the table at the top of `sim/src/sim.cpp`.

## Layout

- `sim/` is the simulation: no dependencies, deterministic, fixed 60 Hz step. `pf/sim.h` is the C++ interface, `pf/c_api.h` the flat C interface, `pf/bot.h` a scripted opponent.
- `app/main.cpp` is the OpenGL 3.3 viewer.
- `python/pfenv.py` wraps the C interface with ctypes: `reset()`, `step(a0, a1)`, `step_vs_bot(a)`.
- `tests/sim_tests.cpp` covers the rules and prints a speed benchmark.

## Training interface

An action is six integers per player: `move_x` (-1, 0, 1), `jump`, `down`, `light`, `heavy`, `shield`. One step is one frame. The observation is 47 floats from the acting player's point of view (own state, opponent state, relative position, time left); see `Game::observe`. Observations are not mirrored for player 2.

```
make sim && python3 python/pfenv.py
```

runs a random agent against the bot as a smoke test.

## Compute

Training can run on the UofT Teaching Labs GPU cluster. Usage notes: `../Claude-Instructions/teach-cs-gpu-cluster.md`. Use `make sim` there, since the nodes have no display.

## Related

Earlier work on the UTMIST AI² Warehouse Brawl environment lives on the lab server at `~/Projects/UTMIST-AI2` (remote `github.com/Finay/UTMIST-AI2`).
