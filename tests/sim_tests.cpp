#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "pf/bot.h"
#include "pf/c_api.h"
#include "pf/sim.h"

using namespace pf;

static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static void run(Game& g, Input a, Input b, int frames) {
    Input in[2] = {a, b};
    for (int i = 0; i < frames; ++i) g.step(in);
}

static void place(Game& g, int p, float x, float y, int facing, bool grounded) {
    Character& c = g.mutableState().ch[p];
    c.pos = {x, y};
    c.vel = {};
    c.facing = facing;
    c.grounded = grounded;
}

static bool near(float a, float b, float tol = 0.02f) { return std::fabs(a - b) < tol; }

static void testRestsOnStage() {
    Game g;
    run(g, {}, {}, 120);
    for (int p = 0; p < 2; ++p) {
        const Character& c = g.state().ch[p];
        CHECK(c.grounded);
        Rect b = bounds(g.body(p));
        CHECK(near(b.y0, 0.0f, 0.005f));  // outline rests exactly on the stage top
    }
}

static void testBodyShape() {
    Poly b = Game::bodyAt({0.0f, 0.0f}, 1);
    CHECK(b.n == 14);
    Rect r = bounds(b);
    CHECK(near(r.y0, -0.5f, 0.005f));
    // Leaning toward +x puts the apex right of centre.
    CHECK(b.v[1].x > 0.1f);
    CHECK(Game::bodyAt({0.0f, 0.0f}, -1).v[1].x < -0.1f);
}

static void testJabHitsAndOnlyWhenActive() {
    Game g;
    place(g, 0, 0.0f, 0.5f, 1, true);
    place(g, 1, 1.2f, 0.5f, -1, true);
    Input press;
    press.light = true;
    Poly hb;
    run(g, press, {}, 1);
    CHECK(g.state().ch[0].move == kLightIdle);
    CHECK(!g.activeHitbox(0, hb));  // still in startup
    CHECK(g.state().ch[1].damage == 0.0f);
    run(g, press, {}, 3);
    CHECK(near(g.state().ch[1].damage, 3.0f));
    CHECK(g.state().ch[1].action == Action::Hitstun);
    CHECK(g.state().ch[1].vel.x > 0.0f);
    // One hit per swing even though the hitbox stays out.
    run(g, press, {}, 2);
    CHECK(near(g.state().ch[1].damage, 3.0f));
}

static void testJabMissesOutOfReach() {
    Game g;
    place(g, 0, 0.0f, 0.5f, 1, true);
    place(g, 1, 2.2f, 0.5f, -1, true);
    Input press;
    press.light = true;
    run(g, press, {}, 14);
    CHECK(g.state().ch[1].damage == 0.0f);
}

static void testMoveSelection() {
    struct Case {
        bool heavy;
        int expect;
        int run_frames;   // frames of holding right first
        int jump_frames;  // frames after a jump before pressing (0 = stay grounded)
    };
    const Case cases[] = {
        {false, kLightIdle, 0, 0}, {true, kHeavyIdle, 0, 0},
        {false, kLightRun, 10, 0}, {true, kHeavyRun, 10, 0},
        {false, kLightRise, 0, 5}, {true, kHeavyRise, 0, 5},
        {false, kLightFall, 0, 30}, {true, kHeavyFall, 0, 30},
    };
    for (const Case& k : cases) {
        Game g;
        place(g, 0, -2.0f, 0.5f, 1, true);
        place(g, 1, 5.0f, 0.5f, -1, true);
        Input hold;
        if (k.run_frames > 0) {
            hold.move_x = 1;
            run(g, hold, {}, k.run_frames);
        }
        if (k.jump_frames > 0) {
            Input j;
            j.jump = true;
            run(g, j, {}, 1);
            run(g, {}, {}, k.jump_frames - 1);
        }
        Input press = hold;
        (k.heavy ? press.heavy : press.light) = true;
        run(g, press, {}, 1);
        CHECK(g.state().ch[0].move == k.expect);
    }
}

static void testKnockbackGrowsWithDamage() {
    float speed[2];
    for (int i = 0; i < 2; ++i) {
        Game g;
        place(g, 0, 0.0f, 0.5f, 1, true);
        place(g, 1, 1.2f, 0.5f, -1, true);
        g.mutableState().ch[1].damage = i == 0 ? 0.0f : 120.0f;
        Input press;
        press.heavy = true;
        Input in[2] = {press, {}};
        speed[i] = 0.0f;
        for (int f = 0; f < 30 && speed[i] == 0.0f; ++f) {
            g.step(in);
            const Character& t = g.state().ch[1];
            if (t.action == Action::Hitstun) speed[i] = std::hypot(t.vel.x, t.vel.y);
        }
    }
    CHECK(speed[0] > 0.0f);
    CHECK(speed[1] > speed[0] * 2.0f);
}

static void testSmashKosAtHighDamageOnly() {
    for (int i = 0; i < 2; ++i) {
        Game g;
        place(g, 0, 0.0f, 0.5f, 1, true);
        place(g, 1, 1.2f, 0.5f, -1, true);
        g.mutableState().ch[1].damage = i == 0 ? 20.0f : 150.0f;
        Input press;
        press.heavy = true;
        run(g, press, {}, 1);
        run(g, {}, {}, 240);
        CHECK(g.state().ch[1].stocks == (i == 0 ? 3 : 2));
    }
}

static void testSoftPlatform() {
    Game g;
    place(g, 0, -3.0f, 4.0f, 1, false);
    run(g, {}, {}, 90);
    CHECK(g.state().ch[0].grounded);
    CHECK(near(bounds(g.body(0)).y0, 2.2f, 0.005f));

    // Jumping up from below passes through it and lands on top.
    place(g, 1, 3.0f, 0.5f, -1, true);
    Input j;
    j.jump = true;
    run(g, {}, j, 1);
    run(g, {}, {}, 90);
    CHECK(near(bounds(g.body(1)).y0, 2.2f, 0.005f));

    // Holding down drops through to the main stage.
    Input d;
    d.down = true;
    run(g, d, {}, 60);
    CHECK(near(bounds(g.body(0)).y0, 0.0f, 0.005f));
}

static void testWalkOffLosesStockAndRespawns() {
    Game g;
    Input left;
    left.move_x = -1;
    run(g, left, {}, 200);
    CHECK(g.state().ch[0].stocks == 2);
    run(g, {}, {}, 200);
    const Character& c = g.state().ch[0];
    CHECK(c.action == Action::Free);
    CHECK(c.grounded);
    CHECK(c.damage == 0.0f);
}

static void testUppercutOncePerAirtime() {
    Game g;
    place(g, 1, 5.0f, 0.5f, -1, true);
    Input j;
    j.jump = true;
    run(g, j, {}, 1);
    run(g, {}, {}, 3);
    Input h;
    h.heavy = true;
    run(g, h, {}, 1);
    CHECK(g.state().ch[0].move == kHeavyRise);
    run(g, {}, {}, moveData(kHeavyRise).startup);
    CHECK(g.state().ch[0].vel.y > 8.0f);  // lifted by the move
    // Wait out the move, jump again to be rising, and try a second one.
    while (g.state().ch[0].action != Action::Free) run(g, {}, {}, 1);
    CHECK(!g.state().ch[0].grounded);
    run(g, j, {}, 1);
    run(g, {}, {}, 2);
    run(g, h, {}, 1);
    CHECK(g.state().ch[0].move != kHeavyRise);
}

static void testOverlappingBodiesSeparate() {
    Game g;
    place(g, 0, 0.0f, 0.5f, 1, true);
    place(g, 1, 0.0f, 0.5f, -1, true);
    run(g, {}, {}, 60);
    CHECK(!overlap(g.body(0), g.body(1)));
    CHECK(std::fabs(g.state().ch[0].pos.x - g.state().ch[1].pos.x) < 1.6f);
    // Airborne characters pass through each other.
    Game h;
    place(h, 0, 0.0f, 3.0f, 1, false);
    place(h, 1, 0.0f, 0.5f, -1, true);
    run(h, {}, {}, 4);
    CHECK(h.state().ch[1].pos.x == 0.0f);
}

static void testSideAttacksPauseThenLunge() {
    for (bool heavy : {false, true}) {
        Game g;
        place(g, 0, -3.0f, 0.5f, 1, true);
        place(g, 1, 5.5f, 0.5f, -1, true);
        Input hold;
        hold.move_x = 1;
        run(g, hold, {}, 10);
        CHECK(g.state().ch[0].vel.x > 5.0f);
        Input press = hold;
        (heavy ? press.heavy : press.light) = true;
        run(g, press, {}, 1);
        const Move& m = moveData(g.state().ch[0].move);
        CHECK(g.state().ch[0].move == (heavy ? kHeavyRun : kLightRun));
        // Stopped dead for the whole wind-up, even with the direction held.
        float x0 = g.state().ch[0].pos.x;
        CHECK(g.state().ch[0].vel.x == 0.0f);
        run(g, hold, {}, m.startup - 1);
        CHECK(near(g.state().ch[0].pos.x, x0, 1e-4f));
        // Thrust forward the moment the hitbox appears.
        run(g, hold, {}, 1);
        Poly hb;
        CHECK(g.activeHitbox(0, hb));
        CHECK(g.state().ch[0].vel.x > 7.0f);
        run(g, hold, {}, m.active);
        CHECK(g.state().ch[0].pos.x > x0 + 0.6f);
    }
}

static void testHitLocksOutInputs() {
    Game g;
    place(g, 0, 0.0f, 0.5f, 1, true);
    place(g, 1, 1.2f, 0.5f, -1, true);
    Input press;
    press.light = true;
    run(g, press, {}, 4);
    CHECK(g.state().ch[1].action == Action::Hitstun);
    int locked = g.state().ch[1].timer;
    CHECK(locked >= 15);  // a quarter of a second even for the weakest hit
    // The victim mashes everything; nothing happens until the lockout ends.
    Input mash;
    mash.move_x = -1;
    for (int f = 0; f < locked - 1; ++f) {
        mash.jump = mash.light = mash.heavy = mash.shield = f % 2 == 0;
        run(g, {}, mash, 1);
        const Character& c = g.state().ch[1];
        CHECK(c.action == Action::Hitstun);
        CHECK(c.move == kNoMove);
        CHECK(c.facing == -1);
        CHECK(c.vel.x >= 0.0f);  // still travelling away, not steering back
        CHECK(c.air_jumps == 1);
    }
    run(g, {}, {}, 2);
    CHECK(g.state().ch[1].action == Action::Free);
}

static float boundsArea(const Poly& p) {
    Rect b = bounds(p);
    return (b.x1 - b.x0) * (b.y1 - b.y0);
}

static void testWindupGrowsThenTurnsActive() {
    Game g;
    place(g, 1, 5.0f, 0.5f, -1, true);
    int startup = moveData(kHeavyIdle).startup;
    Input press;
    press.heavy = true;
    run(g, press, {}, 1);
    Poly windup, active;
    CHECK(g.windupHitbox(0, windup));
    CHECK(!g.activeHitbox(0, active));
    float first = boundsArea(windup);
    run(g, {}, {}, startup / 2);
    CHECK(g.windupHitbox(0, windup));
    CHECK(boundsArea(windup) > first * 4.0f);
    // The last wind-up frame is the full hitbox; the next frame it is live.
    run(g, {}, {}, startup - 1 - startup / 2);
    CHECK(g.windupHitbox(0, windup));
    CHECK(!g.activeHitbox(0, active));
    run(g, {}, {}, 1);
    CHECK(g.activeHitbox(0, active));
    Poly none;
    CHECK(!g.windupHitbox(0, none));
    CHECK(near(boundsArea(windup), boundsArea(active), 1e-4f));
    CHECK(near(bounds(windup).x1, bounds(active).x1, 1e-4f));
}

static void testLargerHitboxesWindUpLonger() {
    for (int id = 0; id < kNumMoves; ++id) {
        const Move& m = moveData(id);
        float twice_area = 0.0f;
        for (int i = 0; i < m.hitbox.n; ++i) {
            Vec2 a = m.hitbox.v[i], b = m.hitbox.v[(i + 1) % m.hitbox.n];
            twice_area += a.x * b.y - b.x * a.y;
        }
        float area = std::fabs(twice_area) / 2.0f;
        if (area >= 0.4f) {
            CHECK(m.startup >= 8);
        } else {
            CHECK(m.startup <= 5);
        }
    }
}

static void testShieldBlocks() {
    Game g;
    place(g, 0, 0.0f, 0.5f, 1, true);
    place(g, 1, 1.2f, 0.5f, -1, true);
    Input guard, press;
    guard.shield = true;
    press.light = true;
    run(g, press, guard, 1);
    Poly bubble;
    CHECK(g.shield(1, bubble));
    CHECK(bubble.n == 16);
    float hp = g.state().ch[1].shield_hp;
    float blocked = 0.0f;
    for (int f = 0; f < 8; ++f) {
        run(g, press, guard, 1);
        blocked += g.state().events.damage_blocked[1];
    }
    CHECK(near(blocked, 3.0f));
    CHECK(g.state().ch[1].damage == 0.0f);
    CHECK(g.state().ch[1].action == Action::Shield);
    CHECK(g.state().ch[1].shield_hp < hp - 3.0f);
    // Releasing drops the shield after a short lag.
    run(g, {}, {}, 1);
    CHECK(!g.shield(1, bubble));
    run(g, {}, {}, 10);
    CHECK(g.state().ch[1].action == Action::Free);
}

static void testAirShieldIsShortAndStopsVerticalMotion() {
    Game g;
    place(g, 1, 5.0f, 0.5f, -1, true);
    // High above the left end of the stage, clear of the soft platforms, rising.
    place(g, 0, -5.4f, 7.0f, 1, false);
    g.mutableState().ch[0].vel.y = 12.0f;
    Input jump, guard;
    jump.jump = true;
    guard.shield = true;
    run(g, guard, {}, 1);
    Poly bubble;
    CHECK(g.shield(0, bubble));
    // Rising speed is gone; only this frame's gravity remains.
    CHECK(g.state().ch[0].vel.y <= 0.0f && g.state().ch[0].vel.y > -1.0f);
    // It ends by itself even while the button stays held.
    int frames = 1;
    while (g.shield(0, bubble) && frames < 100) {
        run(g, guard, {}, 1);
        ++frames;
    }
    CHECK(frames >= 15 && frames <= 25);
    CHECK(!g.state().ch[0].grounded);
    // Only one per airtime.
    while (g.state().ch[0].action != Action::Free) run(g, {}, {}, 1);
    CHECK(!g.state().ch[0].grounded);
    run(g, guard, {}, 1);
    CHECK(!g.shield(0, bubble));
    // Landing gives it back.
    run(g, {}, {}, 90);
    CHECK(g.state().ch[0].grounded);
    run(g, jump, {}, 1);
    run(g, {}, {}, 4);
    run(g, guard, {}, 1);
    CHECK(g.shield(0, bubble));
}

static void testAirShieldBlocksAndStopsAFall() {
    Game g;
    place(g, 0, 0.0f, 0.5f, 1, true);
    place(g, 1, 0.0f, 2.0f, -1, false);
    g.mutableState().ch[1].vel.y = -10.0f;
    Input guard, press;
    guard.shield = true;
    press.light = true;  // standing jab does not reach up; use a jump-in up spike instead
    Input jump;
    jump.jump = true;
    run(g, jump, guard, 1);
    CHECK(g.state().ch[1].vel.y > -1.0f);  // the fall was cancelled
    run(g, press, guard, 1);
    CHECK(g.state().ch[0].move == kLightRise);
    float blocked = 0.0f;
    for (int f = 0; f < 12; ++f) {
        run(g, {}, guard, 1);
        blocked += g.state().events.damage_blocked[1];
    }
    CHECK(blocked > 0.0f);
    CHECK(g.state().ch[1].damage == 0.0f);
}

static void testShieldShrinksAndBreaks() {
    Game g;
    Input guard;
    guard.shield = true;
    run(g, guard, {}, 1);
    Poly bubble;
    CHECK(g.shield(0, bubble));
    float full = boundsArea(bubble);
    run(g, guard, {}, 150);
    CHECK(g.shield(0, bubble));
    CHECK(boundsArea(bubble) < full * 0.6f);
    run(g, guard, {}, 120);
    CHECK(g.state().ch[0].action == Action::Lag);  // broken: stunned
    CHECK(!g.shield(0, bubble));
}

static void testShrunkShieldLeavesBodyExposed() {
    Game g;
    place(g, 0, 0.0f, 0.5f, 1, true);
    // Directly above the defender's apex, which leans to x = 0.21.
    place(g, 1, 0.21f, 3.1f, -1, false);
    g.mutableState().ch[0].shield_hp = 4.0f;
    Input guard, press;
    guard.shield = true;
    press.light = true;  // falling: down spike, onto the apex above the small shield
    run(g, guard, {}, 1);
    run(g, guard, press, 1);
    CHECK(g.state().ch[1].move == kLightFall);
    float taken = 0.0f, blocked = 0.0f;
    for (int f = 0; f < 30; ++f) {
        run(g, guard, {}, 1);
        taken += g.state().events.damage_taken[0];
        blocked += g.state().events.damage_blocked[0];
    }
    CHECK(taken > 0.0f);
    CHECK(blocked == 0.0f);
}

static void testStageIsInvertedTrapezoid() {
    const Stage& st = stage();
    CHECK(st.solid.n == 4);
    Rect b = bounds(st.solid);
    CHECK(b.x0 == st.left && b.x1 == st.right && b.y1 == 0.0f);
    float bottom_lo = 1e9f, bottom_hi = -1e9f;
    for (int i = 0; i < st.solid.n; ++i) {
        if (st.solid.v[i].y != b.y0) continue;
        bottom_lo = std::fmin(bottom_lo, st.solid.v[i].x);
        bottom_hi = std::fmax(bottom_hi, st.solid.v[i].x);
    }
    CHECK(bottom_lo > st.left && bottom_hi < st.right);
    // The top surface still reaches the corners.
    Game g;
    place(g, 0, 5.8f, 0.5f, 1, true);
    run(g, {}, {}, 30);
    CHECK(g.state().ch[0].grounded);
    // Under the overhang is open space that a rectangle would have filled.
    CHECK(!overlap(Game::bodyAt({6.2f, -1.0f}, 1), st.solid));
    CHECK(overlap(Game::bodyAt({6.2f, -1.0f}, 1), rectPoly({st.left, b.y0, st.right, 0.0f})));
}

static void testGameEndsAndReportsWinner() {
    Config cfg;
    cfg.stocks = 1;
    Game g(cfg);
    Input left;
    left.move_x = -1;
    run(g, left, {}, 300);
    CHECK(g.state().done);
    CHECK(g.state().winner == 1);

    Config timed;
    timed.max_frames = 100;
    Game t(timed);
    t.mutableState().ch[0].damage = 30.0f;
    run(t, {}, {}, 200);
    CHECK(t.state().done);
    CHECK(t.state().frame == 100);
    CHECK(t.state().winner == 1);
}

static Input randomInput(unsigned& seed) {
    auto next = [&seed] {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 16;
    };
    Input in;
    in.move_x = static_cast<int>(next() % 3) - 1;
    in.jump = next() % 8 == 0;
    in.down = next() % 10 == 0;
    in.light = next() % 6 == 0;
    in.heavy = next() % 9 == 0;
    in.shield = next() % 12 == 0;
    return in;
}

static void testDeterministicAndFinite() {
    Game a, b;
    unsigned sa = 7, sb = 7;
    for (int f = 0; f < 20000; ++f) {
        Input ia[2] = {randomInput(sa), randomInput(sa)};
        Input ib[2] = {randomInput(sb), randomInput(sb)};
        a.step(ia);
        b.step(ib);
        if (a.state().done) {
            a.reset();
            b.reset();
        }
    }
    bool same = true, finite = true;
    for (int p = 0; p < 2; ++p) {
        const Character& ca = a.state().ch[p];
        const Character& cb = b.state().ch[p];
        same = same && ca.pos.x == cb.pos.x && ca.pos.y == cb.pos.y && ca.damage == cb.damage &&
               ca.stocks == cb.stocks;
        finite = finite && std::isfinite(ca.pos.x) && std::isfinite(ca.pos.y);
    }
    CHECK(same);
    CHECK(finite);
    float obs[Game::kObsSize];
    a.observe(0, obs);
    for (float v : obs) CHECK(std::isfinite(v));
}

static void testBotMatchFinishes() {
    Game g;
    int hits = 0;
    while (!g.state().done) {
        Input in[2] = {botInput(g, 0), botInput(g, 1)};
        g.step(in);
        for (int p = 0; p < 2; ++p) hits += g.state().events.damage_taken[p] > 0.0f ? 1 : 0;
    }
    CHECK(g.state().frame <= g.config().max_frames);
    CHECK(hits > 0);
    std::printf("  bot vs bot: %d frames, %d hits, winner %d, stocks %d-%d\n", g.state().frame, hits,
                g.state().winner, g.state().ch[0].stocks, g.state().ch[1].stocks);
}

static void testVecApiRunsAndResets() {
    for (int opponent : {1, 0}) {
        const int n = 8;
        pf_vec* v = pf_vec_create(n, 1, 600, 4, opponent);
        const int agents = pf_vec_agents(v);
        const int size = pf_obs_size();
        CHECK(agents == (opponent == 0 ? 2 : 1));
        CHECK(pf_vec_num_envs(v) == n);
        std::vector<float> obs(n * agents * size, NAN), rew(n * agents);
        std::vector<unsigned char> term(n), trunc(n);
        std::vector<int> win(n), act(n * agents * 6);
        pf_vec_reset(v, obs.data());
        unsigned seed = 3;
        auto next = [&seed] {
            seed = seed * 1664525u + 1013904223u;
            return seed >> 16;
        };
        int terminated = 0, truncated = 0;
        bool finite = true, zero_sum = true, fresh = true, exclusive = true;
        for (int step = 0; step < 3000; ++step) {
            for (size_t i = 0; i < act.size(); ++i) act[i] = static_cast<int>(next() % (i % 6 == 0 ? 3 : 2));
            pf_vec_step(v, act.data(), obs.data(), rew.data(), term.data(), trunc.data(), win.data());
            for (int e = 0; e < n; ++e) {
                terminated += term[e];
                truncated += trunc[e];
                exclusive = exclusive && !(term[e] && trunc[e]);
                // After a match ends, the observation is frame 0 of a new one: full time left.
                if (term[e] || trunc[e]) fresh = fresh && obs[e * agents * size + size - 1] == 1.0f;
                if (agents == 2) zero_sum = zero_sum && near(rew[e * 2] + rew[e * 2 + 1], 0.0f, 1e-5f);
            }
            for (float x : obs) finite = finite && std::isfinite(x);
        }
        CHECK(finite);
        CHECK(zero_sum);
        CHECK(fresh);
        CHECK(exclusive);
        CHECK(terminated > 0);
        // Random agents kill each other slowly enough that some matches time out.
        if (opponent == 0) CHECK(truncated > 0);
        pf_vec_destroy(v);
    }
}

static void testVecApiActions() {
    pf_vec* v = pf_vec_create(1, 3, 0, 1, 0);
    const int size = pf_obs_size();
    std::vector<float> obs(2 * size), rew(2);
    unsigned char term = 0, trunc = 0;
    int win = 0;
    pf_vec_reset(v, obs.data());
    float x0 = obs[0], x1 = obs[size];
    // Move index 2 is right, 0 is left, 1 is none.
    int right_and_left[12] = {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 20; ++i) pf_vec_step(v, right_and_left, obs.data(), rew.data(), &term, &trunc, &win);
    CHECK(obs[0] > x0 + 0.1f);
    CHECK(obs[size] < x1 - 0.1f);
    int still[12] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    for (int i = 0; i < 30; ++i) pf_vec_step(v, still, obs.data(), rew.data(), &term, &trunc, &win);
    float rest = obs[0];
    pf_vec_step(v, still, obs.data(), rew.data(), &term, &trunc, &win);
    CHECK(near(obs[0], rest, 1e-5f));
    // Jump held on consecutive steps is two presses: ground jump, then the air jump.
    int jump[12] = {1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    CHECK(obs[6] == 1.0f);  // air jumps available
    pf_vec_step(v, jump, obs.data(), rew.data(), &term, &trunc, &win);
    CHECK(obs[5] == 0.0f);  // airborne
    CHECK(obs[6] == 1.0f);
    pf_vec_step(v, jump, obs.data(), rew.data(), &term, &trunc, &win);
    CHECK(obs[6] == 0.0f);
    pf_vec_destroy(v);
}

static void benchmark() {
    Game g;
    unsigned seed = 1;
    const int steps = 2000000;
    auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < steps; ++f) {
        Input in[2] = {randomInput(seed), randomInput(seed)};
        g.step(in);
        if (g.state().done) g.reset();
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  benchmark: %.2f million steps/s on one core (random inputs)\n", steps / secs / 1e6);
}

int main() {
    struct {
        const char* name;
        void (*fn)();
    } tests[] = {
        {"rests on stage", testRestsOnStage},
        {"body shape", testBodyShape},
        {"jab hits, only when active", testJabHitsAndOnlyWhenActive},
        {"jab misses out of reach", testJabMissesOutOfReach},
        {"move selection by movement state", testMoveSelection},
        {"knockback grows with damage", testKnockbackGrowsWithDamage},
        {"smash KOs at high damage only", testSmashKosAtHighDamageOnly},
        {"soft platforms", testSoftPlatform},
        {"walk-off loses a stock, respawns", testWalkOffLosesStockAndRespawns},
        {"uppercut once per airtime", testUppercutOncePerAirtime},
        {"overlapping bodies separate", testOverlappingBodiesSeparate},
        {"side attacks pause, then lunge", testSideAttacksPauseThenLunge},
        {"a hit locks out inputs", testHitLocksOutInputs},
        {"wind-up grows, then turns active", testWindupGrowsThenTurnsActive},
        {"larger hitboxes wind up longer", testLargerHitboxesWindUpLonger},
        {"shield blocks", testShieldBlocks},
        {"air shield is short and stops vertical motion", testAirShieldIsShortAndStopsVerticalMotion},
        {"air shield blocks and stops a fall", testAirShieldBlocksAndStopsAFall},
        {"shield shrinks and breaks", testShieldShrinksAndBreaks},
        {"shrunk shield leaves body exposed", testShrunkShieldLeavesBodyExposed},
        {"stage is an inverted trapezoid", testStageIsInvertedTrapezoid},
        {"game end and winner", testGameEndsAndReportsWinner},
        {"deterministic and finite", testDeterministicAndFinite},
        {"bot match finishes", testBotMatchFinishes},
        {"batched API runs and resets", testVecApiRunsAndResets},
        {"batched API actions", testVecApiActions},
    };
    for (auto& t : tests) {
        int before = g_failures;
        t.fn();
        std::printf("%s %s\n", g_failures == before ? "ok  " : "FAIL", t.name);
    }
    benchmark();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
