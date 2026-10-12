// Two-player platform fighter simulation. Deterministic, fixed 60 Hz step,
// no rendering or I/O dependencies.
#pragma once

#include <array>
#include <cstdint>

#include "pf/geometry.h"

namespace pf {

constexpr int kNumPlayers = 2;
constexpr float kDt = 1.0f / 60.0f;

// One frame of controller state. jump, light and heavy act on the press
// (the frame they go from false to true); move_x and down act while held.
struct Input {
    int move_x = 0;  // -1 left, 0 none, +1 right
    bool jump = false;
    bool down = false;  // fast-fall, and fall through soft platforms
    bool light = false;
    bool heavy = false;
    bool shield = false;  // held, grounded only
};

enum class Action : uint8_t { Free, Attack, Hitstun, Lag, Dead, Shield };

// The movement state at the moment an attack button is pressed picks the move.
enum class MoveState : uint8_t { Idle, Run, Rise, Fall };

enum MoveId : int8_t {
    kNoMove = -1,
    kLightIdle,  // jab
    kLightRun,   // dash poke
    kLightRise,  // up spike
    kLightFall,  // down spike
    kHeavyIdle,  // smash
    kHeavyRun,   // sweep
    kHeavyRise,  // uppercut, once per airtime, lifts the user
    kHeavyFall,  // dive, active until landing
    kNumMoves,
};

struct Move {
    const char* name;
    int startup;   // wind-up frames before the hitbox appears; longer for larger hitboxes
    int active;    // frames the hitbox is out
    int recovery;  // frames after, unable to act
    float damage;
    float base_kb;    // launch speed at 0%
    float kb_scale;   // extra launch speed per 1% on the target
    float angle_deg;  // launch angle, 0 = forward, 90 = straight up
    // Body-local, facing right. The first edge is the base on the body's side,
    // which the wind-up outline grows out from.
    Poly hitbox;
    float friction_scale;  // ground friction multiplier while attacking
    float vx_scale;        // applied to the user's vx when the hitbox appears
    bool sets_vy;
    float vy_set;      // user's vy when the hitbox appears, if sets_vy
    bool until_land;   // hitbox stays out until landing (at most `active` frames)
};

const Move& moveData(int id);

struct SoftPlatform {
    float x0, x1, y;
};

struct Stage {
    Poly solid;                         // main platform, an inverted trapezoid, solid from every side
    float left, right;                  // ends of its top surface
    std::array<SoftPlatform, 3> soft;   // land on from above, pass through otherwise
    Rect blast;                         // leaving this box loses a stock
};

const Stage& stage();

struct Config {
    int stocks = 3;
    int max_frames = 60 * 60 * 3;  // 0 = no time limit
};

struct Character {
    Vec2 pos;  // centre of the semicircle's flat edge
    Vec2 vel;
    int facing = 1;
    bool grounded = false;
    bool fastfall = false;
    bool up_used = false;  // uppercut spent until next landing
    int air_jumps = 0;
    float damage = 0.0f;
    float shield_hp = 0.0f;
    int stocks = 0;
    Action action = Action::Free;
    int timer = 0;  // attack: frames elapsed; hitstun/lag/dead: frames left
    int8_t move = kNoMove;
    bool attack_hit = false;
    int invuln = 0;
    Input prev;
};

// What happened during the most recent step, for reward shaping.
struct StepEvents {
    std::array<float, kNumPlayers> damage_taken{};
    std::array<float, kNumPlayers> damage_blocked{};  // absorbed by this player's shield
    std::array<bool, kNumPlayers> ko{};
};

struct State {
    std::array<Character, kNumPlayers> ch;
    int frame = 0;
    bool done = false;
    int winner = -1;  // player index, or -1 for none / draw
    StepEvents events;
};

class Game {
public:
    static constexpr int kObsSize = 45;

    explicit Game(Config cfg = {});

    void reset();
    void step(const Input in[kNumPlayers]);

    const State& state() const { return s_; }
    State& mutableState() { return s_; }
    const Config& config() const { return cfg_; }

    // The character's collision polygon, which is also exactly what is drawn.
    Poly body(int player) const;
    static Poly bodyAt(Vec2 pos, int facing);

    // The attack polygon, only while it can actually hit.
    bool activeHitbox(int player, Poly& out) const;

    // During an attack's wind-up: the hitbox outline growing out from its base
    // to full size. It cannot hit; it shows what is coming.
    bool windupHitbox(int player, Poly& out) const;

    // The shield bubble while it is up. Attacks that touch it are blocked.
    // It shrinks as it weakens, exposing the parts of the body outside it.
    bool shield(int player, Poly& out) const;

    MoveState moveState(int player, const Input& in) const;

    // Fixed-size feature vector from one player's point of view.
    void observe(int player, float* out) const;

private:
    void control(Character& c, const Input& in);
    void integrate(Character& c, const Input& in);
    void startAttack(Character& c, const Input& in, bool heavy);
    void advanceAttack(Character& c);
    void land(Character& c);
    void breakShield(Character& c);
    void separate();
    void resolveHits();
    void checkBlast(int player);
    void respawn(int player);
    void finishIfOver();

    Config cfg_;
    State s_;
};

}  // namespace pf
