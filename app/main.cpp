// OpenGL viewer. Everything on screen is a line; characters, attacks and
// platforms are drawn from the same polygons the simulation collides with.
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "pf/bot.h"
#include "pf/sim.h"

using namespace pf;

namespace {

struct Color {
    float r, g, b;
};

const Color kPlayerColor[kNumPlayers] = {{1.0f, 1.0f, 1.0f}, {0.45f, 0.9f, 1.0f}};
const Color kStageColor = {0.8f, 0.8f, 0.8f};
const Color kBlastColor = {0.22f, 0.22f, 0.22f};
const Color kWindupColor = {1.0f, 0.85f, 0.1f};
const Color kActiveColor = {1.0f, 0.2f, 0.15f};

// World region that is always visible; it contains the whole blast zone.
const float kViewW = 30.0f, kViewH = 22.5f, kViewCx = 0.0f, kViewCy = 2.25f;
const float kLinePixels = 2.0f;

const char* kVertexSrc = R"(#version 330 core
layout(location = 0) in vec2 pos;
layout(location = 1) in vec3 col;
uniform vec4 view;  // origin.xy, scale.xy
out vec3 v_col;
void main() {
    v_col = col;
    gl_Position = vec4((pos - view.xy) * view.zw - 1.0, 0.0, 1.0);
}
)";

const char* kFragmentSrc = R"(#version 330 core
in vec3 v_col;
out vec4 frag;
void main() { frag = vec4(v_col, 1.0); }
)";

GLuint compile(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(sh, sizeof log, nullptr, log);
        std::fprintf(stderr, "shader error: %s\n", log);
    }
    return sh;
}

// Collects line segments and draws them as thin quads, since core-profile
// OpenGL on macOS only supports 1-pixel GL_LINES.
class LineBatch {
public:
    void init() {
        GLuint vs = compile(GL_VERTEX_SHADER, kVertexSrc);
        GLuint fs = compile(GL_FRAGMENT_SHADER, kFragmentSrc);
        program_ = glCreateProgram();
        glAttachShader(program_, vs);
        glAttachShader(program_, fs);
        glLinkProgram(program_);
        glDeleteShader(vs);
        glDeleteShader(fs);
        view_loc_ = glGetUniformLocation(program_, "view");
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(2 * sizeof(float)));
    }

    void begin(float half_width) {
        verts_.clear();
        half_width_ = half_width;
    }

    void line(Vec2 a, Vec2 b, Color c) {
        Vec2 d = b - a;
        float len = std::sqrt(dot(d, d));
        if (len < 1e-6f) return;
        Vec2 n{-d.y / len * half_width_, d.x / len * half_width_};
        Vec2 q[4] = {a + n, a - n, b - n, b + n};
        for (int i : {0, 1, 2, 0, 2, 3}) {
            verts_.insert(verts_.end(), {q[i].x, q[i].y, c.r, c.g, c.b});
        }
    }

    void loop(const Poly& p, Color c) {
        for (int i = 0; i < p.n; ++i) line(p.v[i], p.v[(i + 1) % p.n], c);
    }

    void dashed(Vec2 a, Vec2 b, Color c, float dash) {
        Vec2 d = b - a;
        float len = std::sqrt(dot(d, d));
        int n = std::max(1, static_cast<int>(len / dash));
        for (int i = 0; i < n; i += 2) {
            line(a + d * (static_cast<float>(i) / n), a + d * (static_cast<float>(i + 1) / n), c);
        }
    }

    void flush(float x0, float y0, float w, float h) {
        glUseProgram(program_);
        glUniform4f(view_loc_, x0, y0, 2.0f / w, 2.0f / h);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts_.size() * sizeof(float)),
                     verts_.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verts_.size() / 5));
    }

private:
    GLuint program_ = 0, vao_ = 0, vbo_ = 0;
    GLint view_loc_ = -1;
    float half_width_ = 0.01f;
    std::vector<float> verts_;
};

// Stroke font on a 4-wide, 6-tall grid.
struct Stroke {
    float x0, y0, x1, y1;
};

void segments(const char* which, std::vector<Stroke>& out) {
    for (const char* s = which; *s; ++s) {
        switch (*s) {
            case 'a': out.push_back({0, 6, 4, 6}); break;
            case 'b': out.push_back({4, 6, 4, 3}); break;
            case 'c': out.push_back({4, 3, 4, 0}); break;
            case 'd': out.push_back({0, 0, 4, 0}); break;
            case 'e': out.push_back({0, 0, 0, 3}); break;
            case 'f': out.push_back({0, 3, 0, 6}); break;
            case 'g': out.push_back({0, 3, 4, 3}); break;
        }
    }
}

std::vector<Stroke> glyph(char ch) {
    std::vector<Stroke> s;
    switch (ch) {
        case '0': segments("abcdef", s); break;
        case '1': segments("bc", s); break;
        case '2': segments("abged", s); break;
        case '3': segments("abgcd", s); break;
        case '4': segments("fgbc", s); break;
        case '5': case 'S': segments("afgcd", s); break;
        case '6': segments("afgecd", s); break;
        case '7': segments("abc", s); break;
        case '8': segments("abcdefg", s); break;
        case '9': segments("abcdfg", s); break;
        case 'P': segments("efabg", s); break;
        case 'A': segments("efabcg", s); break;
        case 'R': segments("efabg", s); s.push_back({1, 3, 4, 0}); break;
        case 'I': s = {{2, 0, 2, 6}}; break;
        case 'N': s = {{0, 0, 0, 6}, {0, 6, 4, 0}, {4, 0, 4, 6}}; break;
        case 'W': s = {{0, 6, 1, 0}, {1, 0, 2, 3}, {2, 3, 3, 0}, {3, 0, 4, 6}}; break;
        case 'D': s = {{0, 0, 0, 6}, {0, 6, 3, 6}, {3, 6, 4, 4}, {4, 4, 4, 2}, {4, 2, 3, 0}, {3, 0, 0, 0}}; break;
        default: break;
    }
    return s;
}

float textWidth(const std::string& text, float height) {
    return (static_cast<float>(text.size()) * 6.0f - 2.0f) * height / 6.0f;
}

void drawText(LineBatch& b, const std::string& text, Vec2 at, float height, Color c) {
    float u = height / 6.0f;
    for (char ch : text) {
        for (const Stroke& s : glyph(ch)) {
            b.line({at.x + s.x0 * u, at.y + s.y0 * u}, {at.x + s.x1 * u, at.y + s.y1 * u}, c);
        }
        at.x += 6.0f * u;
    }
}

Color dim(Color c, float k) { return {c.r * k, c.g * k, c.b * k}; }

void drawScene(LineBatch& b, const Game& game) {
    const State& s = game.state();
    const Stage& st = stage();

    const Rect& bz = st.blast;
    Vec2 corners[4] = {{bz.x0, bz.y0}, {bz.x1, bz.y0}, {bz.x1, bz.y1}, {bz.x0, bz.y1}};
    for (int i = 0; i < 4; ++i) b.dashed(corners[i], corners[(i + 1) % 4], kBlastColor, 0.4f);

    b.loop(st.solid, kStageColor);
    for (const SoftPlatform& sp : st.soft) b.line({sp.x0, sp.y}, {sp.x1, sp.y}, kStageColor);

    for (int p = 0; p < kNumPlayers; ++p) {
        const Character& c = s.ch[p];
        if (c.action == Action::Dead) continue;
        bool blink = c.invuln > 0 && (s.frame / 4) % 2 == 0;
        Color col = blink ? dim(kPlayerColor[p], 0.4f) : kPlayerColor[p];
        b.loop(game.body(p), col);
        Poly shape;
        if (game.shield(p, shape)) b.loop(shape, col);
        if (game.windupHitbox(p, shape)) b.loop(shape, kWindupColor);
        if (game.activeHitbox(p, shape)) b.loop(shape, kActiveColor);
    }

    // HUD: damage and remaining stocks, below the stage.
    for (int p = 0; p < kNumPlayers; ++p) {
        const Character& c = s.ch[p];
        std::string dmg = std::to_string(static_cast<int>(std::lround(c.damage)));
        float x = p == 0 ? -8.0f : 8.0f - textWidth("000", 1.1f);
        drawText(b, dmg, {x, -7.4f}, 1.1f, kPlayerColor[p]);
        Poly icon = Game::bodyAt({0.0f, 0.0f}, p == 0 ? 1 : -1);
        for (int k = 0; k < c.stocks; ++k) {
            Poly small;
            for (int i = 0; i < icon.n; ++i) {
                small.push({x + 0.3f + 0.7f * static_cast<float>(k) + icon.v[i].x * 0.45f,
                            -5.7f + icon.v[i].y * 0.45f});
            }
            b.loop(small, kPlayerColor[p]);
        }
    }

    if (s.done) {
        std::string msg = s.winner < 0 ? "DRAW" : (s.winner == 0 ? "P1 WINS" : "P2 WINS");
        Color col = s.winner < 0 ? kStageColor : kPlayerColor[s.winner];
        drawText(b, msg, {-textWidth(msg, 1.4f) / 2.0f, 8.5f}, 1.4f, col);
    }
}

void render(GLFWwindow* window, LineBatch& batch, const Game& game) {
    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    if (fbw == 0 || fbh == 0) return;
    float aspect = static_cast<float>(fbw) / static_cast<float>(fbh);
    float w = kViewW, h = kViewH;
    if (aspect > kViewW / kViewH) {
        w = kViewH * aspect;
    } else {
        h = kViewW / aspect;
    }
    glViewport(0, 0, fbw, fbh);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    batch.begin(0.5f * kLinePixels * w / static_cast<float>(fbw));
    drawScene(batch, game);
    batch.flush(kViewCx - w / 2.0f, kViewCy - h / 2.0f, w, h);
}

enum class Controller { Human, Bot, Idle };

Controller parseController(const char* s) {
    if (std::strcmp(s, "bot") == 0) return Controller::Bot;
    if (std::strcmp(s, "idle") == 0) return Controller::Idle;
    return Controller::Human;
}

Input keyboardInput(GLFWwindow* w, int player) {
    auto key = [w](int k) { return glfwGetKey(w, k) == GLFW_PRESS; };
    Input in;
    if (player == 0) {
        in.move_x = (key(GLFW_KEY_D) ? 1 : 0) - (key(GLFW_KEY_A) ? 1 : 0);
        in.jump = key(GLFW_KEY_W);
        in.down = key(GLFW_KEY_S);
        in.light = key(GLFW_KEY_F);
        in.heavy = key(GLFW_KEY_G);
        in.shield = key(GLFW_KEY_H);
    } else {
        in.move_x = (key(GLFW_KEY_RIGHT) ? 1 : 0) - (key(GLFW_KEY_LEFT) ? 1 : 0);
        in.jump = key(GLFW_KEY_UP);
        in.down = key(GLFW_KEY_DOWN);
        in.light = key(GLFW_KEY_COMMA);
        in.heavy = key(GLFW_KEY_PERIOD);
        in.shield = key(GLFW_KEY_SLASH);
    }
    return in;
}

bool writeBmp(const std::string& path, int w, int h, const std::vector<unsigned char>& bgr) {
    int row = (w * 3 + 3) & ~3;
    unsigned size = 54 + static_cast<unsigned>(row * h);
    unsigned char hdr[54] = {'B', 'M'};
    auto put32 = [&hdr](int at, unsigned v) {
        for (int i = 0; i < 4; ++i) hdr[at + i] = static_cast<unsigned char>(v >> (8 * i));
    };
    put32(2, size);
    put32(10, 54);
    put32(14, 40);
    put32(18, static_cast<unsigned>(w));
    put32(22, static_cast<unsigned>(h));
    hdr[26] = 1;
    hdr[28] = 24;
    put32(34, static_cast<unsigned>(row * h));
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(hdr), 54);
    f.write(reinterpret_cast<const char*>(bgr.data()), static_cast<std::streamsize>(bgr.size()));
    return static_cast<bool>(f);
}

void usage() {
    std::puts(
        "pf_play [--p1 human|bot|idle] [--p2 human|bot|idle]\n"
        "        [--screenshot out.bmp --frames N [--wait hitbox|windup|shield]]\n"
        "\n"
        "P1: A/D move, W jump, S down, F light, G heavy, H shield\n"
        "P2: arrows move/jump/down, comma light, period heavy, slash shield\n"
        "R restart, Esc quit");
}

}  // namespace

int main(int argc, char** argv) {
    Controller ctl[kNumPlayers] = {Controller::Human, Controller::Bot};
    std::string shot;
    int shot_frames = 0;
    std::string wait_for;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        bool has_value = i + 1 < argc;
        if (a == "--p1" && has_value) {
            ctl[0] = parseController(argv[++i]);
        } else if (a == "--p2" && has_value) {
            ctl[1] = parseController(argv[++i]);
        } else if (a == "--screenshot" && has_value) {
            shot = argv[++i];
        } else if (a == "--frames" && has_value) {
            shot_frames = std::atoi(argv[++i]);
        } else if (a == "--wait" && has_value) {
            wait_for = argv[++i];
        } else {
            usage();
            return a == "--help" ? 0 : 1;
        }
    }

    if (!glfwInit()) {
        std::fprintf(stderr, "could not initialise GLFW\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    if (!shot.empty()) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(960, 720, "Platform Fighter", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "could not create an OpenGL 3.3 window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glEnable(GL_MULTISAMPLE);

    LineBatch batch;
    batch.init();
    Game game;

    auto inputFor = [&](int p) {
        if (ctl[p] == Controller::Bot) return botInput(game, p);
        if (ctl[p] == Controller::Human && shot.empty()) return keyboardInput(window, p);
        return Input{};
    };
    auto stepOnce = [&] {
        Input in[kNumPlayers] = {inputFor(0), inputFor(1)};
        game.step(in);
    };

    if (!shot.empty()) {
        for (int f = 0; f < shot_frames; ++f) stepOnce();
        // Optionally keep going until the scene shows the thing being checked.
        auto showing = [&] {
            for (int p = 0; p < kNumPlayers; ++p) {
                const Character& c = game.state().ch[p];
                Poly shape;
                if (wait_for == "hitbox" && game.activeHitbox(p, shape)) return true;
                if (wait_for == "shield" && game.shield(p, shape)) return true;
                if (wait_for == "windup" && game.windupHitbox(p, shape) &&
                    c.timer * 2 >= moveData(c.move).startup && c.timer >= 4) {
                    return true;
                }
            }
            return false;
        };
        for (int f = 0; !wait_for.empty() && f < 20000 && !showing(); ++f) stepOnce();
        render(window, batch, game);
        glFinish();
        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        int row = (w * 3 + 3) & ~3;
        std::vector<unsigned char> pixels(static_cast<size_t>(row) * static_cast<size_t>(h));
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadBuffer(GL_BACK);
        glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
        bool ok = writeBmp(shot, w, h, pixels);
        std::printf("%s %s (%dx%d, frame %d)\n", ok ? "wrote" : "failed to write", shot.c_str(), w, h,
                    game.state().frame);
        glfwTerminate();
        return ok ? 0 : 1;
    }

    double last = glfwGetTime();
    double acc = 0.0;
    bool r_was_down = false;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        bool r_down = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
        if (r_down && !r_was_down) game.reset();
        r_was_down = r_down;

        double now = glfwGetTime();
        acc += std::min(now - last, 0.25);
        last = now;
        while (acc >= kDt) {
            stepOnce();
            acc -= kDt;
        }
        render(window, batch, game);
        glfwSwapBuffers(window);
    }
    glfwTerminate();
    return 0;
}
