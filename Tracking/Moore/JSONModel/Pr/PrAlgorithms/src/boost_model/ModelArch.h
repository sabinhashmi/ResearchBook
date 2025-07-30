#ifndef PR_PRALGORITHMS_BOOST_MODEL_MODELARCH_H
#define PR_PRALGORITHMS_BOOST_MODEL_MODELARCH_H

#include <vector>
#include <string>
#include <cmath>
#include <optional>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// --------- Internal Structures ---------
struct Split {
    int featureIndex;
    float border;
};

struct Tree {
    std::vector<Split> splits;
    std::vector<float> leafValues;
};

struct Model {
    std::vector<Tree> trees;
    float scale = 1.0f;
    float bias = 0.0f;
    std::vector<float> feature_mins;
    std::vector<float> feature_maxs;
};

// --------- Global Model (static for independent instances per translation unit) ---------
static Model CatboostModel;
static bool isModelLoaded = false;

// --------- Utility Functions ---------
inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

inline void min_max_scale(std::vector<float>& features, const Model& model) {
    if (model.feature_mins.empty() || model.feature_maxs.empty()) return;
    for (size_t i = 0; i < features.size(); ++i) {
        if (i >= model.feature_mins.size() || i >= model.feature_maxs.size()) break;
        float denom = model.feature_maxs[i] - model.feature_mins[i];
        if (denom != 0.f) {
            features[i] = (features[i] - model.feature_mins[i]) / denom;
        } else {
            features[i] = 0.f; // Handle division by zero
        }
    }
}

inline Model load_model_from_json(const std::string& buffer) {
    Model model;

    if (buffer.empty())
        throw std::runtime_error("Cannot load model from empty buffer");

    json j = json::parse(buffer.begin(), buffer.end());

    for (const auto& treeJson : j["oblivious_trees"]) {
        Tree tree;
        for (const auto& splitJson : treeJson["splits"]) {
            tree.splits.push_back({
                splitJson["float_feature_index"].get<int>(),
                splitJson["border"].get<float>()
            });
        }
        for (const auto& leafVal : treeJson["leaf_values"])
            tree.leafValues.push_back(leafVal.get<float>());

        model.trees.push_back(tree);
    }

    if (j.contains("scale_and_bias")) {
        auto& sb = j["scale_and_bias"];
        if (sb.size() >= 2) {
            model.scale = sb[0].get<float>();
            model.bias = sb[1].is_array() ? sb[1][0].get<float>() : sb[1].get<float>();
        }
    }

    if (j.contains("feature_mins")) {
        for (const auto& val : j["feature_mins"])
            model.feature_mins.push_back(val.get<float>());
    }

    if (j.contains("feature_maxs")) {
        for (const auto& val : j["feature_maxs"])
            model.feature_maxs.push_back(val.get<float>());
    }

    return model;
}

// --------- Inference Core ---------
inline float predict(const Model& model, const std::vector<float>& features) {
    float result = 0.f;
    for (const auto& tree : model.trees) {
        int leafIndex = 0;
        for (size_t d = 0; d < tree.splits.size(); ++d) {
            const auto& split = tree.splits[d];
            if (features[split.featureIndex] > split.border)
                leafIndex |= (1 << d);
        }
        result += tree.leafValues[leafIndex];
    }
    return result * model.scale + model.bias;
}

// --------- Public Interface ---------
inline void EnsureCatboostModelLoaded(const std::optional<std::string>& model_buffer_opt) {
    if (!isModelLoaded) {
        if (!model_buffer_opt) {
            throw std::runtime_error("Catboost model buffer is empty.");
        }
        CatboostModel = load_model_from_json(*model_buffer_opt);
        isModelLoaded = true;
    }
}

inline double ApplyCatboostModel(const std::vector<float>& floatFeatures) {
    if (!isModelLoaded) {
        throw std::runtime_error("Catboost model not loaded. Call EnsureCatboostModelLoaded during initialization.");
    }
    std::vector<float> features = floatFeatures;
    min_max_scale(features, CatboostModel);
    return predict(CatboostModel, features);
}

#endif // PR_PRALGORITHMS_BOOST_MODEL_MODELARCH_H
