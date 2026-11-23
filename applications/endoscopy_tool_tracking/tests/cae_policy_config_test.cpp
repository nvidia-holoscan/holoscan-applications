// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host-only checks of the CAE policy JSON configuration. argv[1] is the shipped default file.

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../cpp/cae_policy.hpp"

namespace endo = endoscopy_tool_tracking;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_rejected(std::string_view json, std::string_view expected_text) {
  try {
    (void)endo::parse_cae_policy_config(json);
  } catch (const std::invalid_argument& error) {
    require(std::string_view(error.what()).find(expected_text) != std::string_view::npos,
            "unexpected error for " + std::string(json) + ": " + error.what());
    return;
  }
  throw std::runtime_error("accepted invalid config: " + std::string(json));
}

bool same(const endo::CaePolicyConfig& a, const endo::CaePolicyConfig& b) {
  return a.class_threshold == b.class_threshold && a.mask_threshold == b.mask_threshold &&
         a.background_policy == b.background_policy && a.class_policies == b.class_policies;
}

}  // namespace

int main(int argc, char** argv) try {
  require(argc == 2, "usage: cae_policy_config_test cae_policy_default.json");

  // The shipped default file and the built-in defaults must agree.
  const auto shipped = endo::load_cae_policy_config(argv[1]);
  require(same(shipped, endo::CaePolicyConfig{}), "cae_policy_default.json differs from defaults");
  require(shipped.background_policy == 3U, "default background must be deprioritize");

  // Names (case-insensitive), numbers and numeric strings select classes and policies.
  const auto mapped = endo::parse_cae_policy_config(R"({
    "schema_version": 1, "class_threshold": 0.25, "mask_threshold": 0.75,
    "background_policy": 4,
    "classes": {"Grasper": "protect", "1": "PROTECT_STRONG", "hook": 3, "6": "1"}})");
  require(mapped.class_threshold == 0.25F && mapped.mask_threshold == 0.75F, "thresholds");
  require(mapped.background_policy == 4U, "numeric background policy");
  require(mapped.class_policies[0] == 1U && mapped.class_policies[1] == 2U &&
              mapped.class_policies[2] == 3U && mapped.class_policies[6] == 1U,
          "class mapping by name and number");
  require(mapped.class_policies[3] == 0U && mapped.class_policies[5] == 0U,
          "unlisted classes must be neutral");

  // Only schema_version is required.
  const auto minimal = endo::parse_cae_policy_config(R"({"schema_version": 1})");
  require(same(minimal, endo::CaePolicyConfig{}), "omitted fields must keep defaults");

  require_rejected(R"({"classes": {}})", "schema_version is required");
  require_rejected(R"({"schema_version": 2})", "schema_version must be 1");
  require_rejected(R"({"schema_version": 1, "extra": 1})", "unknown field");
  require_rejected(R"({"schema_version": 1, "classes": {"forceps": "protect"}})", "unknown class");
  require_rejected(R"({"schema_version": 1, "classes": {"7": "protect"}})", "unknown class");
  require_rejected(R"({"schema_version": 1, "classes": {"01": "protect"}})", "unknown class");
  require_rejected(R"({"schema_version": 1, "classes": {"hook": "01"}})", "must be one of");
  require_rejected(R"({"schema_version": 1, "classes": {"0": "protect", "grasper": "neutral"}})",
                   "more than once");
  require_rejected(R"({"schema_version": 1, "classes": {"hook": "protect", "hook": "neutral"}})",
                   "duplicate key");
  require_rejected(R"({"schema_version": 1, "classes": {"hook": "boost"}})", "must be one of");
  require_rejected(R"({"schema_version": 1, "classes": {"hook": 5}})", "must be one of");
  require_rejected(R"({"schema_version": 1, "classes": {"hook": 1.5}})", "must be one of");
  require_rejected(R"({"schema_version": 1, "background_policy": "loud"})", "must be one of");
  require_rejected(R"({"schema_version": 1, "class_threshold": 1.5})", "in [0, 1]");
  require_rejected(R"({"schema_version": 1, "mask_threshold": "0.5"})", "in [0, 1]");
  require_rejected(R"({"schema_version": 1, "classes": ["hook"]})", "arrays are not supported");
  require_rejected(R"({"schema_version": 1,})", "expected a string");
  require_rejected(R"({"schema_version": 1} trailing)", "trailing content");
  require_rejected(R"([1])", "arrays are not supported");
  require_rejected("\xEF\xBB\xBF{\"schema_version\": 1}", "byte order mark");
  require_rejected("1", "expected a JSON object");
  require_rejected(R"({"schema_version": 1, "classes": "hook"})", "classes must be an object");

  // Strings: control characters, escapes other than \" and \\, and missing closing quotes.
  require_rejected("{\"schema_version\": 1, \"classes\": {\"ho\x01ok\": 1}}", "control character");
  require_rejected(R"({"schema_version": 1, "classes": {"ho\nok": 1}})",
                   "unsupported string escape");
  require_rejected(R"({"schema_version": 1, "classes": {"hook)", "unterminated string");

  // Numbers follow the JSON grammar; whitespace is only space, tab, LF and CR.
  const auto exponent = endo::parse_cae_policy_config(
      R"({"schema_version": 1.0, "class_threshold": 5e-1, "mask_threshold": 2.5E-1})");
  require(exponent.class_threshold == 0.5F && exponent.mask_threshold == 0.25F, "exponents");
  require_rejected(R"({"schema_version": 01})", "expected a value");
  require_rejected(R"({"schema_version": 1, "class_threshold": .5})", "expected a value");
  require_rejected(R"({"schema_version": 1, "class_threshold": 1.})", "expected a value");
  require_rejected(R"({"schema_version": 1, "class_threshold": +0.5})", "expected a value");
  require_rejected(R"({"schema_version": 1, "class_threshold": 1e})", "expected a value");
  require_rejected("{\"schema_version\":\v1}", "expected a value");

  // Nesting: depth 8 parses (then fails as an unknown field), depth 9 is rejected.
  require_rejected(
      R"({"schema_version": 1, "x": {"a": {"b": {"c": {"d": {"e": {"f": {"g": {}}}}}}}}})",
      "unknown field");
  require_rejected(
      R"({"schema_version": 1, "x": {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {}}}}}}}}}})",
      "nesting is too deep");

  // A directory is not a configuration file.
  try {
    (void)endo::load_cae_policy_config(std::filesystem::path(argv[1]).parent_path().string());
    throw std::runtime_error("accepted a directory as a config file");
  } catch (const std::invalid_argument& error) {
    require(std::string_view(error.what()).find("not a readable regular file") !=
                std::string_view::npos,
            std::string("unexpected error for a directory: ") + error.what());
  }

  // One priority order shared by host code and the kernel; values outside 0..4 rank lowest.
  require(endo::policy_rank(2U) > endo::policy_rank(1U) &&
              endo::policy_rank(1U) > endo::policy_rank(0U) &&
              endo::policy_rank(0U) > endo::policy_rank(3U) &&
              endo::policy_rank(3U) > endo::policy_rank(4U),
          "policy priority order");
  require(endo::policy_rank(endo::kInvalidPolicyValue) < endo::policy_rank(4U),
          "non-policy values must rank below every policy");

  std::cout << "CAE policy configuration tests passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
