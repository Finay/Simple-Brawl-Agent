#include "pf/geometry.h"

#include <algorithm>
#include <limits>

namespace pf {

namespace {

struct Range {
    float lo, hi;
};

Range project(const Poly& p, Vec2 axis) {
    Range r{std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
    for (int i = 0; i < p.n; ++i) {
        float d = dot(p.v[i], axis);
        r.lo = std::min(r.lo, d);
        r.hi = std::max(r.hi, d);
    }
    return r;
}

bool boundsDisjoint(const Poly& a, const Poly& b) {
    Rect ra = bounds(a), rb = bounds(b);
    return ra.x1 <= rb.x0 || rb.x1 <= ra.x0 || ra.y1 <= rb.y0 || rb.y1 <= ra.y0;
}

}  // namespace

Poly rectPoly(const Rect& r) {
    Poly p;
    p.push({r.x0, r.y0});
    p.push({r.x1, r.y0});
    p.push({r.x1, r.y1});
    p.push({r.x0, r.y1});
    return p;
}

Rect bounds(const Poly& p) {
    Rect r{p.v[0].x, p.v[0].y, p.v[0].x, p.v[0].y};
    for (int i = 1; i < p.n; ++i) {
        r.x0 = std::min(r.x0, p.v[i].x);
        r.y0 = std::min(r.y0, p.v[i].y);
        r.x1 = std::max(r.x1, p.v[i].x);
        r.y1 = std::max(r.y1, p.v[i].y);
    }
    return r;
}

bool overlap(const Poly& a, const Poly& b) {
    if (boundsDisjoint(a, b)) return false;
    for (const Poly* p : {&a, &b}) {
        for (int i = 0; i < p->n; ++i) {
            Vec2 e = p->v[(i + 1) % p->n] - p->v[i];
            Vec2 axis{-e.y, e.x};
            Range ra = project(a, axis), rb = project(b, axis);
            if (ra.hi <= rb.lo || rb.hi <= ra.lo) return false;
        }
    }
    return true;
}

bool penetration(const Poly& a, const Poly& b, Vec2& normal, float& depth) {
    if (boundsDisjoint(a, b)) return false;
    depth = std::numeric_limits<float>::max();
    for (const Poly* p : {&a, &b}) {
        for (int i = 0; i < p->n; ++i) {
            Vec2 e = p->v[(i + 1) % p->n] - p->v[i];
            float len = std::sqrt(dot(e, e));
            if (len < 1e-6f) continue;
            Vec2 axis{-e.y / len, e.x / len};
            Range ra = project(a, axis), rb = project(b, axis);
            float push_pos = rb.hi - ra.lo;  // move a along +axis
            float push_neg = ra.hi - rb.lo;  // move a along -axis
            if (push_pos <= 0.0f || push_neg <= 0.0f) return false;
            if (push_pos < depth) {
                depth = push_pos;
                normal = axis;
            }
            if (push_neg < depth) {
                depth = push_neg;
                normal = axis * -1.0f;
            }
        }
    }
    return true;
}

bool minYInStrip(const Poly& p, float x0, float x1, float& min_y) {
    bool hit = false;
    min_y = std::numeric_limits<float>::max();
    for (int i = 0; i < p.n; ++i) {
        Vec2 a = p.v[i], b = p.v[(i + 1) % p.n];
        if (a.x > b.x) std::swap(a, b);
        if (b.x < x0 || a.x > x1) continue;
        float dx = b.x - a.x;
        float ya = a.y, yb = b.y;
        if (dx > 1e-9f) {
            float ta = std::max(0.0f, (x0 - a.x) / dx);
            float tb = std::min(1.0f, (x1 - a.x) / dx);
            ya = a.y + (b.y - a.y) * ta;
            yb = a.y + (b.y - a.y) * tb;
        }
        min_y = std::min(min_y, std::min(ya, yb));
        hit = true;
    }
    return hit;
}

}  // namespace pf
