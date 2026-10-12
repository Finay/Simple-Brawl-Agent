/* Flat C interface to the simulation, for ctypes / cffi bindings. */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pf_env pf_env;

typedef struct pf_status {
    int frame;
    int done;
    int winner;            /* player index, or -1 for none / draw */
    float damage[2];
    int stocks[2];
    float damage_taken[2]; /* during the most recent step */
    float damage_blocked[2]; /* absorbed by this player's shield during the most recent step */
    int ko[2];             /* lost a stock during the most recent step */
} pf_status;

/* stocks <= 0 or max_frames < 0 keep the defaults; max_frames == 0 means no time limit. */
pf_env* pf_create(int stocks, int max_frames);
void pf_destroy(pf_env* env);
void pf_reset(pf_env* env);

/* actions: 12 ints, 6 per player: move_x (-1/0/1), jump, down, light, heavy, shield. */
void pf_step(pf_env* env, const int* actions);

/* Like pf_step for one player, with the scripted bot controlling the other. */
void pf_step_vs_bot(pf_env* env, int player, const int* action);

int pf_obs_size(void);
void pf_observe(const pf_env* env, int player, float* out);
void pf_get_status(const pf_env* env, pf_status* out);

/* ---- Batched stepping for training: many matches per call, no rendering. ----
 *
 * opponent: 0 = both players are agents (2 agents per match),
 *           1 = player 2 is the scripted bot (1 agent per match).
 * frame_skip: frames simulated per step (>= 1); the action is held for all of them.
 *
 * Buffer layouts (A = pf_vec_agents, N = pf_vec_num_envs):
 *   actions    N * A * 6 ints: move (0 left, 1 none, 2 right), jump, down, light, heavy, shield.
 *              Note move is 0/1/2 here, unlike pf_step's -1/0/1.
 *              Every step's buttons count as new presses, so no release step is needed.
 *   obs        N * A * pf_obs_size() floats
 *   rewards    N * A floats: (damage dealt - damage taken) / 100, +1 per stock taken, -1 per stock lost,
 *              minus the optional distance penalty below
 *   terminated N bytes: the match ended by a player running out of stocks
 *   truncated  N bytes: the match ended on the time limit
 *   winners    N ints: winning player or -1, meaningful only where a match ended
 *
 * A finished match resets automatically; its obs is then the first of the new match.
 */
typedef struct pf_vec pf_vec;

pf_vec* pf_vec_create(int num_envs, int stocks, int max_frames, int frame_skip, int opponent);
void pf_vec_destroy(pf_vec* vec);
int pf_vec_num_envs(const pf_vec* vec);
int pf_vec_agents(const pf_vec* vec);
void pf_vec_reset(pf_vec* vec, float* obs);

/* Optional shaping against passive play, off by default. Every frame in which
 * both fighters are alive, each agent loses per_frame * w, where w is 0 when
 * they are within near_dist of each other, 1 beyond far_dist, and linear in
 * between. Both sides pay it, so rewards are no longer zero-sum. */
void pf_vec_set_distance_penalty(pf_vec* vec, float per_frame, float near_dist, float far_dist);
void pf_vec_step(pf_vec* vec, const int* actions, float* obs, float* rewards,
                 unsigned char* terminated, unsigned char* truncated, int* winners);

#ifdef __cplusplus
}
#endif
