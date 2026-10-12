#include "pf/c_api.h"

#include <algorithm>
#include <cmath>
#include <vector>

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

struct pf_vec {
    std::vector<pf::Game> games;
    int frame_skip = 1;
    int agents = 1;
    float proximity = 0.0f;  // per-frame bonus at full closeness
    float near_dist = 2.0f;
    float far_dist = 8.0f;
};

namespace {

pf::Input toVecInput(const int* a) {
    pf::Input in = toInput(a);
    in.move_x = a[0] - 1;
    return in;
}

void writeObs(const pf_vec* vec, float* obs) {
    const int size = pf::Game::kObsSize;
    for (size_t e = 0; e < vec->games.size(); ++e) {
        for (int p = 0; p < vec->agents; ++p) {
            vec->games[e].observe(p, obs + (e * vec->agents + p) * size);
        }
    }
}

}  // namespace

pf_vec* pf_vec_create(int num_envs, int stocks, int max_frames, int frame_skip, int opponent) {
    pf::Config cfg;
    if (stocks > 0) cfg.stocks = stocks;
    if (max_frames >= 0) cfg.max_frames = max_frames;
    pf_vec* vec = new pf_vec;
    vec->games.assign(static_cast<size_t>(num_envs > 0 ? num_envs : 1), pf::Game(cfg));
    vec->frame_skip = frame_skip > 0 ? frame_skip : 1;
    vec->agents = opponent == 0 ? 2 : 1;
    return vec;
}

void pf_vec_destroy(pf_vec* vec) { delete vec; }

int pf_vec_num_envs(const pf_vec* vec) { return static_cast<int>(vec->games.size()); }

int pf_vec_agents(const pf_vec* vec) { return vec->agents; }

void pf_vec_reset(pf_vec* vec, float* obs) {
    for (pf::Game& g : vec->games) g.reset();
    writeObs(vec, obs);
}

void pf_vec_set_proximity_reward(pf_vec* vec, float per_frame, float near_dist, float far_dist) {
    vec->proximity = per_frame;
    vec->near_dist = near_dist;
    vec->far_dist = std::max(far_dist, near_dist + 1e-3f);
}

void pf_vec_step(pf_vec* vec, const int* actions, float* obs, float* rewards,
                 unsigned char* terminated, unsigned char* truncated, int* winners) {
    const int agents = vec->agents;
    for (size_t e = 0; e < vec->games.size(); ++e) {
        pf::Game& g = vec->games[e];
        const int* act = actions + e * agents * 6;
        float* reward = rewards + e * agents;
        for (int p = 0; p < agents; ++p) {
            reward[p] = 0.0f;
            // Treat this step's buttons as fresh presses.
            g.mutableState().ch[p].prev = pf::Input{};
        }
        for (int k = 0; k < vec->frame_skip && !g.state().done; ++k) {
            pf::Input in[pf::kNumPlayers];
            in[0] = toVecInput(act);
            in[1] = agents == 2 ? toVecInput(act + 6) : pf::botInput(g, 1);
            g.step(in);
            const pf::StepEvents& ev = g.state().events;
            for (int p = 0; p < agents; ++p) {
                int o = 1 - p;
                reward[p] += (ev.damage_taken[o] - ev.damage_taken[p]) / 100.0f +
                             (ev.ko[o] ? 1.0f : 0.0f) - (ev.ko[p] ? 1.0f : 0.0f);
            }
            const pf::Character& a = g.state().ch[0];
            const pf::Character& b = g.state().ch[1];
            if (vec->proximity != 0.0f && a.action != pf::Action::Dead && b.action != pf::Action::Dead) {
                float dist = std::hypot(a.pos.x - b.pos.x, a.pos.y - b.pos.y);
                float w = std::clamp((vec->far_dist - dist) / (vec->far_dist - vec->near_dist), 0.0f, 1.0f);
                for (int p = 0; p < agents; ++p) reward[p] += vec->proximity * w;
            }
        }
        const pf::State& s = g.state();
        bool out_of_stocks = s.ch[0].stocks <= 0 || s.ch[1].stocks <= 0;
        terminated[e] = s.done && out_of_stocks ? 1 : 0;
        truncated[e] = s.done && !out_of_stocks ? 1 : 0;
        winners[e] = s.winner;
        if (s.done) g.reset();
    }
    writeObs(vec, obs);
}
