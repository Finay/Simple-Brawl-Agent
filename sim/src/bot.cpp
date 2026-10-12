#include "pf/bot.h"

#include <cmath>

namespace pf {

Input botInput(const Game& game, int player) {
    const State& s = game.state();
    const Character& me = s.ch[player];
    const Character& opp = s.ch[1 - player];
    const Stage& st = stage();
    Input in;
    if (me.action == Action::Dead) return in;

    // Buttons act on the press, so a stateless bot presses on alternate frames.
    bool tap = s.frame % 2 == 0;
    // Attack timing varies from frame to frame, differently for each player,
    // so two bots do not mirror each other forever.
    unsigned h = static_cast<unsigned>(s.frame) * 2654435761u + static_cast<unsigned>(player) * 40503u;
    bool strike = (h >> 13) % 3 == 0;
    auto sign = [](float v) { return v > 0.0f ? 1 : -1; };

    bool below = me.pos.y < -0.2f;
    bool off_stage = me.pos.x < st.left || me.pos.x > st.right || below;
    if (off_stage) {
        // Underneath the stage, head outward first to get around its edge.
        bool under = below && std::fabs(me.pos.x) < st.right + 0.4f;
        in.move_x = under ? sign(me.pos.x) : -sign(me.pos.x);
        if (me.vel.y <= 0.0f && me.air_jumps > 0) {
            in.jump = tap;
        } else if (me.vel.y > 0.0f && me.vel.y < 2.5f && !me.up_used && me.pos.y < 1.5f) {
            in.heavy = tap;  // uppercut as a last lift back to the stage
        }
        return in;
    }

    if (opp.action == Action::Dead) {
        if (std::fabs(me.pos.x) > 1.0f) in.move_x = -sign(me.pos.x);
        return in;
    }

    float dx = opp.pos.x - me.pos.x;
    float dy = opp.pos.y - me.pos.y;
    float adx = std::fabs(dx);
    bool opp_off_stage = opp.pos.x < st.left || opp.pos.x > st.right;
    bool at_edge = std::fabs(me.pos.x) > st.right - 0.8f;

    // Close in further while airborne, where the attacks reach up and down.
    float keep_away = me.grounded ? 1.0f : 0.3f;
    if (adx > keep_away && !(opp_off_stage && at_edge)) in.move_x = sign(dx);

    // Sometimes shield a slow attack from its wind-up until its hitbox is gone.
    // Whether it does is fixed for half a second at a time, so it mostly
    // commits to the block or ignores the swing.
    unsigned mood = static_cast<unsigned>(s.frame / 30) * 2246822519u + static_cast<unsigned>(player) * 97u;
    bool wary = (mood >> 11) % 2 == 0;
    bool threatened = false;
    if (opp.action == Action::Attack && opp.move >= kHeavyIdle) {
        const Move& m = moveData(opp.move);
        threatened = opp.timer < m.startup + m.active && adx < 2.6f && std::fabs(dy) < 1.5f;
    }
    bool can_shield = me.grounded || !me.air_shield_used || me.action == Action::Shield;
    if (can_shield && wary && threatened && me.shield_hp > 15.0f) {
        Input guard;
        guard.shield = true;
        return guard;
    }

    if (me.grounded) {
        if (std::fabs(dy) < 1.0f && adx < 1.7f) {
            if (me.facing != sign(dx)) {
                in.move_x = sign(dx);
            } else if (opp.damage > 80.0f && adx < 1.5f) {
                in.move_x = 0;
                in.heavy = strike;
            } else {
                in.light = strike;
            }
        } else if (dy > 1.2f && adx < 2.0f) {
            in.jump = tap;
        }
    } else {
        if (me.vel.y > 0.0f && dy > 0.3f && dy < 2.2f && adx < 0.8f) in.light = strike;
        if (me.vel.y <= 0.0f && dy < -0.5f && dy > -2.0f && adx < 0.6f) in.light = strike;
    }
    return in;
}

}  // namespace pf
