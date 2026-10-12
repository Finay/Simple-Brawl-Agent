#include "pf/c_api.h"

#include "pf/bot.h"
#include "pf/sim.h"

struct pf_env {
    pf::Game game;
};

namespace {

pf::Input toInput(const int* a) {
    pf::Input in;
    in.move_x = a[0];
    in.jump = a[1] != 0;
    in.down = a[2] != 0;
    in.light = a[3] != 0;
    in.heavy = a[4] != 0;
    in.shield = a[5] != 0;
    return in;
}

}  // namespace

pf_env* pf_create(int stocks, int max_frames) {
    pf::Config cfg;
    if (stocks > 0) cfg.stocks = stocks;
    if (max_frames >= 0) cfg.max_frames = max_frames;
    return new pf_env{pf::Game(cfg)};
}

void pf_destroy(pf_env* env) { delete env; }

void pf_reset(pf_env* env) { env->game.reset(); }

void pf_step(pf_env* env, const int* actions) {
    pf::Input in[pf::kNumPlayers] = {toInput(actions), toInput(actions + 6)};
    env->game.step(in);
}

void pf_step_vs_bot(pf_env* env, int player, const int* action) {
    pf::Input in[pf::kNumPlayers];
    in[player] = toInput(action);
    in[1 - player] = pf::botInput(env->game, 1 - player);
    env->game.step(in);
}

int pf_obs_size(void) { return pf::Game::kObsSize; }

void pf_observe(const pf_env* env, int player, float* out) { env->game.observe(player, out); }

void pf_get_status(const pf_env* env, pf_status* out) {
    const pf::State& s = env->game.state();
    out->frame = s.frame;
    out->done = s.done ? 1 : 0;
    out->winner = s.winner;
    for (int p = 0; p < pf::kNumPlayers; ++p) {
        out->damage[p] = s.ch[p].damage;
        out->stocks[p] = s.ch[p].stocks;
        out->damage_taken[p] = s.events.damage_taken[p];
        out->damage_blocked[p] = s.events.damage_blocked[p];
        out->ko[p] = s.events.ko[p] ? 1 : 0;
    }
}
