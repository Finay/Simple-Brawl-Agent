// A simple scripted opponent: a sparring partner for people and a baseline
// for trained agents. Stateless, so it is deterministic given the game state.
#pragma once

#include "pf/sim.h"

namespace pf {

Input botInput(const Game& game, int player);

}  // namespace pf
