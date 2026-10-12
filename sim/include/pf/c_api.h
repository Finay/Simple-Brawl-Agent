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
    int ko[2];             /* lost a stock during the most recent step */
} pf_status;

/* stocks <= 0 or max_frames < 0 keep the defaults; max_frames == 0 means no time limit. */
pf_env* pf_create(int stocks, int max_frames);
void pf_destroy(pf_env* env);
void pf_reset(pf_env* env);

/* actions: 10 ints, 5 per player: move_x (-1/0/1), jump, down, light, heavy. */
void pf_step(pf_env* env, const int* actions);

/* Like pf_step for one player, with the scripted bot controlling the other. */
void pf_step_vs_bot(pf_env* env, int player, const int* action);

int pf_obs_size(void);
void pf_observe(const pf_env* env, int player, float* out);
void pf_get_status(const pf_env* env, pf_status* out);

#ifdef __cplusplus
}
#endif
