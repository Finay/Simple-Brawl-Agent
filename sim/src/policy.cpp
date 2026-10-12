#include "pf/policy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace pf {

namespace {

constexpr int kActionLogits = 8;  // 3 move choices + 5 buttons

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

template <typename T>
bool readInto(std::ifstream& f, T* out, size_t count) {
    f.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(sizeof(T) * count));
    return static_cast<bool>(f);
}

}  // namespace

bool Policy::load(const std::string& path, std::string* error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail(error, "cannot open " + path);
    char magic[4];
    int32_t skip = 0, count = 0;
    if (!readInto(f, magic, 4) || std::memcmp(magic, "PFP1", 4) != 0) return fail(error, "not a policy file");
    if (!readInto(f, &skip, 1) || !readInto(f, &count, 1) || skip < 1 || count < 1 || count > 16) {
        return fail(error, "bad header");
    }
    std::vector<Layer> layers(static_cast<size_t>(count));
    for (Layer& l : layers) {
        int32_t dims[2];
        if (!readInto(f, dims, 2) || dims[0] < 1 || dims[1] < 1 || dims[0] > 8192 || dims[1] > 8192) {
            return fail(error, "bad layer size");
        }
        l.in = dims[0];
        l.out = dims[1];
        l.weight.resize(static_cast<size_t>(l.in) * static_cast<size_t>(l.out));
        l.bias.resize(static_cast<size_t>(l.out));
        if (!readInto(f, l.weight.data(), l.weight.size()) || !readInto(f, l.bias.data(), l.bias.size())) {
            return fail(error, "truncated file");
        }
    }
    for (size_t i = 1; i < layers.size(); ++i) {
        if (layers[i].in != layers[i - 1].out) return fail(error, "layer sizes do not chain");
    }
    if (layers.front().in != Game::kObsSize) {
        return fail(error, "policy expects " + std::to_string(layers.front().in) + " observations, this build has " +
                               std::to_string(Game::kObsSize));
    }
    if (layers.back().out != kActionLogits) return fail(error, "unexpected action size");

    layers_ = std::move(layers);
    frame_skip_ = skip;
    countdown_ = 0;

    // The exporter stores the logits PyTorch produced for an all-0.1 observation.
    float expected[kActionLogits];
    if (!readInto(f, expected, kActionLogits)) return fail(error, "missing forward-pass check");
    std::vector<float> got = forward(std::vector<float>(Game::kObsSize, 0.1f));
    for (int i = 0; i < kActionLogits; ++i) {
        if (std::fabs(got[static_cast<size_t>(i)] - expected[i]) > 1e-3f) {
            return fail(error, "forward pass does not match the exporter's");
        }
    }
    return true;
}

std::vector<float> Policy::forward(const std::vector<float>& obs) const {
    std::vector<float> x = obs;
    for (size_t i = 0; i < layers_.size(); ++i) {
        const Layer& l = layers_[i];
        std::vector<float> y(static_cast<size_t>(l.out));
        for (int o = 0; o < l.out; ++o) {
            const float* w = &l.weight[static_cast<size_t>(o) * static_cast<size_t>(l.in)];
            float sum = l.bias[static_cast<size_t>(o)];
            for (int k = 0; k < l.in; ++k) sum += w[k] * x[static_cast<size_t>(k)];
            // Hidden layers use tanh, as in python/train_ppo.py.
            y[static_cast<size_t>(o)] = i + 1 < layers_.size() ? std::tanh(sum) : sum;
        }
        x = std::move(y);
    }
    return x;
}

Input Policy::next(Game& game, int player) {
    if (countdown_ == 0) {
        std::vector<float> obs(Game::kObsSize);
        game.observe(player, obs.data());
        std::vector<float> logits = forward(obs);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

        float top = std::max(logits[0], std::max(logits[1], logits[2]));
        float p[3], total = 0.0f;
        for (int i = 0; i < 3; ++i) total += p[i] = std::exp(logits[static_cast<size_t>(i)] - top);
        float r = uniform(rng_) * total;
        int move = r < p[0] ? 0 : (r < p[0] + p[1] ? 1 : 2);
        bool random_action = uniform(rng_) < noise_;
        if (random_action) move = static_cast<int>(uniform(rng_) * 3.0f) % 3;
        auto pressed = [&](int i) {
            if (random_action) return uniform(rng_) < 0.5f;
            return uniform(rng_) < 1.0f / (1.0f + std::exp(-logits[static_cast<size_t>(3 + i)]));
        };

        held_ = Input{};
        held_.move_x = move - 1;
        held_.jump = pressed(0);
        held_.down = pressed(1);
        held_.light = pressed(2);
        held_.heavy = pressed(3);
        held_.shield = pressed(4);
        game.mutableState().ch[player].prev = Input{};
        countdown_ = frame_skip_;
    }
    --countdown_;
    return held_;
}

}  // namespace pf
