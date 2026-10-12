// Runs a trained policy network without PyTorch, so the viewer can play
// checkpoints. Files come from python/export_policy.py.
#pragma once

#include <random>
#include <string>
#include <vector>

#include "pf/sim.h"

namespace pf {

class Policy {
public:
    // Returns false and fills `error` if the file is missing, malformed, built
    // for a different observation size, or fails its built-in forward-pass check.
    bool load(const std::string& path, std::string* error = nullptr);

    // The input for this frame. As in training, a new action is sampled every
    // frame-skip frames, its buttons count as fresh presses, and it is then held.
    Input next(Game& game, int player);

    int frameSkip() const { return frame_skip_; }

    // Sampling is seeded from the system by default, so matches differ.
    void seed(unsigned value) { rng_.seed(value); }

    // Chance that a decision is replaced by a uniformly random action.
    void setNoise(float probability) { noise_ = probability; }

private:
    struct Layer {
        int in = 0, out = 0;
        std::vector<float> weight, bias;  // weight is row-major, out x in
    };

    std::vector<float> forward(const std::vector<float>& obs) const;

    std::vector<Layer> layers_;
    int frame_skip_ = 1;
    int countdown_ = 0;
    Input held_;
    float noise_ = 0.0f;
    std::mt19937 rng_{std::random_device{}()};
};

}  // namespace pf
