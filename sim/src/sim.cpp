#include "pf/sim.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>

namespace pf {

namespace {

constexpr float kPi = 3.14159265358979f;

// Character shape: an equilateral triangle sitting on the flat edge of a
// semicircle whose curve faces down.
constexpr float kRadius = 0.5f;
constexpr float kTriHeight = kRadius * 1.7320508f;
constexpr int kArcSegments = 12;
// The body leans toward the way it faces, which is how facing is shown.
constexpr float kLean = 14.0f * kPi / 180.0f;

constexpr float kGravity = 40.0f;
constexpr float kMaxFall = 16.0f;
constexpr float kFastFall = 24.0f;
constexpr float kDiveSpeed = 24.0f;
constexpr float kRunSpeed = 7.0f;
constexpr float kGroundAccel = 80.0f;
constexpr float kGroundFriction = 60.0f;
constexpr float kAirSpeed = 6.0f;
constexpr float kAirAccel = 30.0f;
constexpr float kAirFriction = 6.0f;
constexpr float kJumpSpeed = 15.0f;
constexpr float kAirJumpSpeed = 14.0f;
constexpr int kAirJumps = 1;
constexpr float kRunThreshold = 1.0f;

constexpr float kHitstunDrag = 0.8f;
constexpr float kHitstunPerKb = 1.2f;
constexpr int kMinHitstun = 10;
constexpr int kLandingLag = 5;
constexpr int kSideCooldown = 60;  // frames before another running attack
constexpr int kRespawnDelay = 45;
constexpr int kRespawnInvuln = 120;
constexpr float kPlatformSnap = 0.01f;
constexpr float kPushSpeed = 3.0f;

constexpr float kShieldMax = 50.0f;
constexpr float kShieldDrain = 0.2f;        // per frame while held
constexpr float kShieldRegen = 0.1f;        // per frame while down
constexpr float kShieldDamageScale = 1.2f;  // shield health lost per point blocked
constexpr float kShieldRadiusMax = 0.9f;
constexpr float kShieldRadiusMin = 0.45f;
constexpr float kShieldCenterY = (kTriHeight - kRadius) / 2.0f;
constexpr float kShieldPush = 3.0f;
constexpr int kShieldDropLag = 5;
constexpr int kAirShieldFrames = 18;
constexpr int kShieldBreakStun = 100;

Poly poly(std::initializer_list<Vec2> pts) {
    Poly p;
    for (Vec2 v : pts) p.push(v);
    return p;
}

Poly makeLocalBody() {
    Poly p;
    p.push({kRadius, 0.0f});
    p.push({0.0f, kTriHeight});
    p.push({-kRadius, 0.0f});
    for (int k = 1; k < kArcSegments; ++k) {
        float a = kPi + kPi * static_cast<float>(k) / kArcSegments;
        p.push({kRadius * std::cos(a), kRadius * std::sin(a)});
    }
    return p;
}

const Poly kLocalBody = makeLocalBody();

//                         name         su  act rec  dmg  base  scale angle  hitbox
const std::array<Move, kNumMoves> kMoves = {{
    {"jab",        3,  3,  8,  3.0f, 5.0f, 0.03f, 30.0f,
     poly({{0.5f, 0.5f}, {0.5f, 0.0f}, {1.3f, 0.25f}}),
     1.0f, 1.0f, false, 0.0f, false, false, 0.0f},
    {"dash poke",  4,  5, 12,  6.0f, 7.0f, 0.06f, 40.0f,
     poly({{0.5f, 0.45f}, {0.5f, 0.05f}, {1.9f, 0.25f}}),
     0.5f, 1.0f, false, 0.0f, false, true, 12.0f},
    {"up spike",   3,  5, 10,  5.0f, 7.0f, 0.07f, 85.0f,
     poly({{-0.3f, 0.9f}, {0.3f, 0.9f}, {0.0f, 1.9f}}),
     1.0f, 1.0f, false, 0.0f, false, false, 0.0f},
    {"down spike", 5,  6, 12,  7.0f, 6.0f, 0.08f, -80.0f,
     poly({{-0.3f, -0.5f}, {0.3f, -0.5f}, {0.0f, -1.4f}}),
     1.0f, 1.0f, false, 0.0f, false, false, 0.0f},
    {"smash",     16,  4, 22, 14.0f, 9.0f, 0.16f, 40.0f,
     poly({{0.5f, 0.9f}, {0.5f, -0.3f}, {2.1f, 0.3f}}),
     1.0f, 1.0f, false, 0.0f, false, false, 0.0f},
    {"sweep",     10,  6, 18,  9.0f, 8.0f, 0.10f, 70.0f,
     poly({{0.4f, 0.1f}, {0.4f, -0.5f}, {1.8f, -0.5f}, {1.8f, -0.15f}}),
     0.4f, 1.0f, false, 0.0f, false, true, 9.0f},
    {"uppercut",  12, 10, 16,  8.0f, 8.0f, 0.11f, 80.0f,
     poly({{-0.6f, 0.6f}, {0.6f, 0.6f}, {0.9f, 1.5f}, {0.0f, 2.0f}, {-0.9f, 1.5f}}),
     1.0f, 1.0f, true, 13.0f, false, false, 0.0f},
    {"dive",       8, 45, 14, 10.0f, 8.0f, 0.12f, 55.0f,
     poly({{-0.45f, -0.5f}, {0.45f, -0.5f}, {0.0f, -1.5f}}),
     1.0f, 0.3f, true, -kDiveSpeed, true, false, 0.0f},
}};

const Stage kStage = {
    poly({{-6.0f, 0.0f}, {-4.8f, -1.2f}, {4.8f, -1.2f}, {6.0f, 0.0f}}),
    -6.0f,
    6.0f,
    {{{-4.5f, -1.5f, 2.2f}, {1.5f, 4.5f, 2.2f}, {-1.5f, 1.5f, 4.4f}}},
    {-14.0f, -8.0f, 14.0f, 12.5f},
};

float approach(float value, float target, float step) {
    if (value < target) return std::min(value + step, target);
    return std::max(value - step, target);
}

bool isAerialMove(int move) { return move % 4 >= 2; }

bool hitboxOut(const Character& c) {
    if (c.action != Action::Attack) return false;
    const Move& m = kMoves[c.move];
    return c.timer >= m.startup && c.timer < m.startup + m.active;
}

// The move's hitbox in world space, scaled about the midpoint of its base edge.
Poly worldHitbox(const Character& c, float scale) {
    const Poly& local = kMoves[c.move].hitbox;
    Vec2 base = (local.v[0] + local.v[1]) * 0.5f;
    Poly out;
    for (int i = 0; i < local.n; ++i) {
        Vec2 v = base + (local.v[i] - base) * scale;
        out.push({c.pos.x + v.x * static_cast<float>(c.facing), c.pos.y + v.y});
    }
    return out;
}

}  // namespace

const Move& moveData(int id) { return kMoves[id]; }

const Stage& stage() { return kStage; }

Game::Game(Config cfg) : cfg_(cfg) { reset(); }

void Game::reset() {
    s_ = State{};
    for (int p = 0; p < kNumPlayers; ++p) {
        Character& c = s_.ch[p];
        c.pos = {p == 0 ? -3.0f : 3.0f, kRadius};
        c.facing = p == 0 ? 1 : -1;
        c.grounded = true;
        c.air_jumps = kAirJumps;
        c.stocks = cfg_.stocks;
        c.shield_hp = kShieldMax;
    }
}

Poly Game::bodyAt(Vec2 pos, int facing) {
    float a = -static_cast<float>(facing) * kLean;
    float ca = std::cos(a), sa = std::sin(a);
    Poly out;
    for (int i = 0; i < kLocalBody.n; ++i) {
        Vec2 v = kLocalBody.v[i];
        out.push({pos.x + v.x * ca - v.y * sa, pos.y + v.x * sa + v.y * ca});
    }
    return out;
}

Poly Game::body(int player) const {
    const Character& c = s_.ch[player];
    return bodyAt(c.pos, c.facing);
}

bool Game::activeHitbox(int player, Poly& out) const {
    const Character& c = s_.ch[player];
    if (!hitboxOut(c)) return false;
    out = worldHitbox(c, 1.0f);
    return true;
}

bool Game::windupHitbox(int player, Poly& out) const {
    const Character& c = s_.ch[player];
    if (c.action != Action::Attack) return false;
    const Move& m = kMoves[c.move];
    if (c.timer >= m.startup) return false;
    out = worldHitbox(c, static_cast<float>(c.timer + 1) / static_cast<float>(m.startup));
    return true;
}

bool Game::shield(int player, Poly& out) const {
    const Character& c = s_.ch[player];
    if (c.action != Action::Shield) return false;
    float strength = std::clamp(c.shield_hp / kShieldMax, 0.0f, 1.0f);
    float r = kShieldRadiusMin + (kShieldRadiusMax - kShieldRadiusMin) * strength;
    out = Poly{};
    for (int k = 0; k < Poly::kMax; ++k) {
        float a = 2.0f * kPi * static_cast<float>(k) / Poly::kMax;
        out.push({c.pos.x + r * std::cos(a), c.pos.y + kShieldCenterY + r * std::sin(a)});
    }
    return true;
}

void Game::breakShield(Character& c) {
    c.action = Action::Lag;
    c.timer = kShieldBreakStun;
    c.shield_hp = kShieldMax * 0.3f;
}

MoveState Game::moveState(int player, const Input& in) const {
    const Character& c = s_.ch[player];
    if (c.grounded) {
        bool moving = in.move_x != 0 || std::fabs(c.vel.x) > kRunThreshold;
        return moving ? MoveState::Run : MoveState::Idle;
    }
    return c.vel.y > 0.0f ? MoveState::Rise : MoveState::Fall;
}

void Game::startAttack(Character& c, const Input& in, bool heavy) {
    int player = static_cast<int>(&c - s_.ch.data());
    MoveState ms = moveState(player, in);
    // Running attacks share a cooldown; until it ends the standing move comes out.
    if (ms == MoveState::Run) {
        if (c.side_cooldown > 0) {
            ms = MoveState::Idle;
        } else {
            c.side_cooldown = kSideCooldown;
        }
    }
    int id = (heavy ? 4 : 0) + static_cast<int>(ms);
    if (id == kHeavyRise) {
        if (c.up_used) return;
        c.up_used = true;
    }
    c.action = Action::Attack;
    c.move = static_cast<int8_t>(id);
    c.timer = 0;
    c.attack_hit = false;
    if (kMoves[id].pause) c.vel.x = 0.0f;
}

void Game::advanceAttack(Character& c) {
    const Move& m = kMoves[c.move];
    ++c.timer;
    if (c.timer == m.startup) {
        c.vel.x *= m.vx_scale;
        if (m.lunge != 0.0f) c.vel.x = static_cast<float>(c.facing) * m.lunge;
        if (m.sets_vy) {
            c.vel.y = m.vy_set;
            c.grounded = false;
        }
    }
    if (c.timer >= m.startup + m.active + m.recovery) {
        c.action = Action::Free;
        c.move = kNoMove;
    }
}

void Game::control(Character& c, const Input& in) {
    if (c.invuln > 0) --c.invuln;
    if (c.side_cooldown > 0) --c.side_cooldown;
    if (c.action != Action::Shield) c.shield_hp = std::min(kShieldMax, c.shield_hp + kShieldRegen);
    bool jump = in.jump && !c.prev.jump;
    bool down = in.down && !c.prev.down;
    bool light = in.light && !c.prev.light;
    bool heavy = in.heavy && !c.prev.heavy;
    bool shield_press = in.shield && !c.prev.shield;

    switch (c.action) {
        case Action::Hitstun:
        case Action::Lag:
            if (--c.timer <= 0) c.action = Action::Free;
            break;
        case Action::Attack:
            advanceAttack(c);
            break;
        case Action::Free:
            if (in.move_x != 0) c.facing = in.move_x;
            if (in.shield && c.grounded) {
                c.action = Action::Shield;
                c.timer = 0;
            } else if (shield_press && !c.grounded && !c.air_shield_used) {
                c.action = Action::Shield;
                c.timer = kAirShieldFrames;
                c.air_shield_used = true;
                c.fastfall = false;
                c.vel.y = 0.0f;
            } else if (heavy || light) {
                startAttack(c, in, heavy);
            } else if (jump) {
                if (c.grounded) {
                    c.vel.y = kJumpSpeed;
                    c.grounded = false;
                } else if (c.air_jumps > 0) {
                    c.vel.y = kAirJumpSpeed;
                    c.fastfall = false;
                    --c.air_jumps;
                }
            } else if (down && !c.grounded && c.vel.y <= 0.0f) {
                c.fastfall = true;
                c.vel.y = -kFastFall;
            }
            break;
        case Action::Shield:
            // An air shield that lands carries on as a ground shield.
            if (c.timer > 0 && c.grounded) c.timer = 0;
            if (c.timer > 0) {
                c.shield_hp -= kShieldDrain;
                if (c.shield_hp <= 0.0f) {
                    breakShield(c);
                } else if (--c.timer == 0) {
                    c.action = Action::Lag;
                    c.timer = kShieldDropLag;
                }
            } else if (jump) {
                c.action = Action::Free;
                c.vel.y = kJumpSpeed;
                c.grounded = false;
            } else if (!in.shield || !c.grounded) {
                c.action = Action::Lag;
                c.timer = kShieldDropLag;
            } else {
                c.shield_hp -= kShieldDrain;
                if (c.shield_hp <= 0.0f) breakShield(c);
            }
            break;
        case Action::Dead:
            break;
    }

    bool attacking = c.action == Action::Attack;
    if (c.grounded) {
        if (c.action == Action::Free) {
            float rate = in.move_x != 0 ? kGroundAccel : kGroundFriction;
            c.vel.x = approach(c.vel.x, static_cast<float>(in.move_x) * kRunSpeed, rate * kDt);
        } else {
            float scale = attacking ? kMoves[c.move].friction_scale : 1.0f;
            c.vel.x = approach(c.vel.x, 0.0f, kGroundFriction * scale * kDt);
        }
    } else if (c.action == Action::Free || attacking) {
        if (in.move_x != 0) {
            float target = static_cast<float>(in.move_x) * kAirSpeed;
            // Steer toward the target but never brake momentum already beyond it.
            if ((target - c.vel.x) * static_cast<float>(in.move_x) > 0.0f) {
                float accel = kAirAccel * (attacking ? 0.5f : 1.0f);
                c.vel.x = approach(c.vel.x, target, accel * kDt);
            }
        } else {
            c.vel.x = approach(c.vel.x, 0.0f, kAirFriction * kDt);
        }
    }
}

void Game::land(Character& c) {
    c.air_jumps = kAirJumps;
    c.fastfall = false;
    c.up_used = false;
    c.air_shield_used = false;
    if (c.action == Action::Attack && isAerialMove(c.move)) {
        const Move& m = kMoves[c.move];
        if (m.until_land) {
            c.timer = std::max(c.timer, m.startup + m.active);
        } else {
            c.action = Action::Lag;
            c.timer = kLandingLag;
            c.move = kNoMove;
        }
    }
}

void Game::integrate(Character& c, const Input& in) {
    bool was_grounded = c.grounded;
    Poly before = bodyAt(c.pos, c.facing);

    c.vel.y -= kGravity * kDt;
    if (c.action == Action::Hitstun) {
        c.vel = c.vel * (1.0f - kHitstunDrag * kDt);
    } else {
        float limit = c.fastfall ? kFastFall : kMaxFall;
        if (c.action == Action::Attack && c.move == kHeavyFall) limit = kDiveSpeed;
        c.vel.y = std::max(c.vel.y, -limit);
    }
    c.pos = c.pos + c.vel * kDt;
    c.grounded = false;

    Vec2 normal;
    float depth = 0.0f;
    if (penetration(bodyAt(c.pos, c.facing), kStage.solid, normal, depth)) {
        c.pos = c.pos + normal * depth;
        float into = dot(c.vel, normal);
        if (into < 0.0f) c.vel = c.vel - normal * into;
        if (normal.y > 0.6f) c.grounded = true;
    }

    bool pass_through = in.down && c.action == Action::Free;
    if (!c.grounded && !pass_through && c.vel.y <= 0.0f) {
        Poly after = bodyAt(c.pos, c.facing);
        for (const SoftPlatform& sp : kStage.soft) {
            float y_before = 0.0f, y_after = 0.0f;
            if (!minYInStrip(before, sp.x0, sp.x1, y_before)) continue;
            if (!minYInStrip(after, sp.x0, sp.x1, y_after)) continue;
            // Land only when the outline crosses the platform line from above.
            if (y_before >= sp.y - kPlatformSnap && y_after < sp.y) {
                c.pos.y += sp.y - y_after;
                c.vel.y = 0.0f;
                c.grounded = true;
                break;
            }
        }
    }

    if (c.grounded && !was_grounded) land(c);
}

// Grounded characters standing inside each other slide apart, so neither can
// hide inside the other where forward attacks cannot reach.
void Game::separate() {
    Character& a = s_.ch[0];
    Character& b = s_.ch[1];
    if (a.action == Action::Dead || b.action == Action::Dead) return;
    if (!a.grounded || !b.grounded) return;
    if (!overlap(body(0), body(1))) return;
    float dir = a.pos.x > b.pos.x ? 1.0f : -1.0f;
    a.pos.x += dir * kPushSpeed * kDt;
    b.pos.x -= dir * kPushSpeed * kDt;
}

void Game::resolveHits() {
    enum class Outcome { Miss, Hit, Blocked };
    std::array<Outcome, kNumPlayers> outcome{};
    for (int a = 0; a < kNumPlayers; ++a) {
        const Character& atk = s_.ch[a];
        const Character& tgt = s_.ch[1 - a];
        Poly hitbox, bubble;
        if (atk.attack_hit || !activeHitbox(a, hitbox)) continue;
        if (tgt.action == Action::Dead || tgt.invuln > 0) continue;
        // A hitbox touching the shield is blocked even if it also reaches the body.
        if (shield(1 - a, bubble) && overlap(hitbox, bubble)) {
            outcome[a] = Outcome::Blocked;
        } else if (overlap(hitbox, body(1 - a))) {
            outcome[a] = Outcome::Hit;
        }
    }
    // Decide both hits before applying either, so simultaneous hits trade.
    std::array<Character, kNumPlayers> snapshot = s_.ch;
    for (int a = 0; a < kNumPlayers; ++a) {
        if (outcome[a] == Outcome::Miss) continue;
        const Character& atk = snapshot[a];
        const Move& m = kMoves[atk.move];
        Character& tgt = s_.ch[1 - a];
        s_.ch[a].attack_hit = true;

        if (outcome[a] == Outcome::Blocked) {
            tgt.shield_hp -= m.damage * kShieldDamageScale;
            tgt.vel.x += static_cast<float>(atk.facing) * kShieldPush;
            s_.events.damage_blocked[1 - a] += m.damage;
            if (tgt.shield_hp <= 0.0f) breakShield(tgt);
            continue;
        }

        tgt.damage += m.damage;
        float kb = m.base_kb + m.kb_scale * tgt.damage;
        float ang = m.angle_deg * kPi / 180.0f;
        Vec2 dir{std::cos(ang) * static_cast<float>(atk.facing), std::sin(ang)};
        // A downward hit on a grounded target bounces it off the floor.
        if (snapshot[1 - a].grounded && dir.y < 0.0f) dir.y = -dir.y * 0.8f;

        tgt.vel = dir * kb;
        tgt.action = Action::Hitstun;
        tgt.timer = std::max(kMinHitstun, static_cast<int>(kb * kHitstunPerKb));
        tgt.move = kNoMove;
        tgt.grounded = false;
        tgt.fastfall = false;
        s_.events.damage_taken[1 - a] += m.damage;
    }
}

void Game::checkBlast(int player) {
    Character& c = s_.ch[player];
    if (c.action == Action::Dead) return;
    const Rect& b = kStage.blast;
    if (c.pos.x >= b.x0 && c.pos.x <= b.x1 && c.pos.y >= b.y0 && c.pos.y <= b.y1) return;
    --c.stocks;
    c.action = Action::Dead;
    c.timer = kRespawnDelay;
    c.move = kNoMove;
    c.vel = {};
    s_.events.ko[player] = true;
}

void Game::respawn(int player) {
    Character& c = s_.ch[player];
    int stocks = c.stocks;
    c = Character{};
    c.stocks = stocks;
    c.pos = {player == 0 ? -1.0f : 1.0f, 6.5f};
    c.facing = player == 0 ? 1 : -1;
    c.air_jumps = kAirJumps;
    c.shield_hp = kShieldMax;
    c.invuln = kRespawnInvuln;
}

void Game::finishIfOver() {
    const Character& a = s_.ch[0];
    const Character& b = s_.ch[1];
    if (a.stocks <= 0 || b.stocks <= 0) {
        s_.done = true;
        s_.winner = a.stocks > 0 ? 0 : (b.stocks > 0 ? 1 : -1);
        return;
    }
    if (cfg_.max_frames > 0 && s_.frame >= cfg_.max_frames) {
        s_.done = true;
        if (a.stocks != b.stocks) {
            s_.winner = a.stocks > b.stocks ? 0 : 1;
        } else if (a.damage != b.damage) {
            s_.winner = a.damage < b.damage ? 0 : 1;
        } else {
            s_.winner = -1;
        }
    }
}

void Game::step(const Input in[kNumPlayers]) {
    if (s_.done) return;
    s_.events = StepEvents{};

    std::array<Input, kNumPlayers> clean;
    for (int p = 0; p < kNumPlayers; ++p) {
        clean[p] = in[p];
        clean[p].move_x = std::clamp(in[p].move_x, -1, 1);
    }

    for (int p = 0; p < kNumPlayers; ++p) {
        Character& c = s_.ch[p];
        if (c.action == Action::Dead) {
            if (c.stocks > 0 && --c.timer <= 0) respawn(p);
        } else {
            control(c, clean[p]);
        }
    }
    for (int p = 0; p < kNumPlayers; ++p) {
        if (s_.ch[p].action != Action::Dead) integrate(s_.ch[p], clean[p]);
    }
    separate();
    resolveHits();
    for (int p = 0; p < kNumPlayers; ++p) checkBlast(p);
    for (int p = 0; p < kNumPlayers; ++p) s_.ch[p].prev = clean[p];

    ++s_.frame;
    finishIfOver();
}

void Game::observe(int player, float* out) const {
    int i = 0;
    auto put = [&](const Character& c) {
        out[i++] = c.pos.x / 10.0f;
        out[i++] = c.pos.y / 10.0f;
        out[i++] = c.vel.x / 20.0f;
        out[i++] = c.vel.y / 20.0f;
        out[i++] = static_cast<float>(c.facing);
        out[i++] = c.grounded ? 1.0f : 0.0f;
        out[i++] = static_cast<float>(c.air_jumps);
        out[i++] = c.damage / 100.0f;
        out[i++] = c.shield_hp / kShieldMax;
        out[i++] = static_cast<float>(c.stocks) / static_cast<float>(std::max(1, cfg_.stocks));
        for (int a = 0; a < 6; ++a) out[i++] = static_cast<int>(c.action) == a ? 1.0f : 0.0f;
        out[i++] = static_cast<float>(c.move + 1) / static_cast<float>(kNumMoves);
        float phase = 0.0f;
        if (c.action == Action::Attack) {
            const Move& m = kMoves[c.move];
            phase = static_cast<float>(c.timer) / static_cast<float>(m.startup + m.active + m.recovery);
        } else if (c.action != Action::Free) {
            phase = std::min(1.0f, static_cast<float>(c.timer) / 60.0f);
        }
        out[i++] = phase;
        out[i++] = hitboxOut(c) ? 1.0f : 0.0f;
        out[i++] = c.invuln > 0 ? 1.0f : 0.0f;
        out[i++] = c.up_used ? 1.0f : 0.0f;
        out[i++] = c.air_shield_used ? 1.0f : 0.0f;
        out[i++] = static_cast<float>(c.side_cooldown) / static_cast<float>(kSideCooldown);
    };
    const Character& me = s_.ch[player];
    const Character& opp = s_.ch[1 - player];
    put(me);
    put(opp);
    out[i++] = (opp.pos.x - me.pos.x) / 10.0f;
    out[i++] = (opp.pos.y - me.pos.y) / 10.0f;
    out[i++] = cfg_.max_frames > 0
                   ? 1.0f - static_cast<float>(s_.frame) / static_cast<float>(cfg_.max_frames)
                   : 1.0f;
}

}  // namespace pf
