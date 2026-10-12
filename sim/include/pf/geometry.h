// Convex-polygon geometry. Every collidable thing in the game is one of these,
// and the renderer draws the same vertices, so what you see is what collides.
#pragma once

#include <array>
#include <cmath>

namespace pf {

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(Vec2 a, float s) { return {a.x * s, a.y * s}; }
inline float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }

// Fixed capacity so the simulation never allocates.
struct Poly {
    static constexpr int kMax = 16;
    std::array<Vec2, kMax> v{};
    int n = 0;

    void push(Vec2 p) { v[n++] = p; }
};

struct Rect {
    float x0, y0, x1, y1;
};

Poly rectPoly(const Rect& r);
Rect bounds(const Poly& p);

// True if two convex polygons overlap. Winding order does not matter.
bool overlap(const Poly& a, const Poly& b);

// If a and b overlap, returns true and the shortest translation
// (unit normal * depth) that moves a out of b.
bool penetration(const Poly& a, const Poly& b, Vec2& normal, float& depth);

// Lowest y of the part of the polygon lying within x0 <= x <= x1.
// Returns false if the polygon does not reach into that strip.
bool minYInStrip(const Poly& p, float x0, float x1, float& min_y);

}  // namespace pf
