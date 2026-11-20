// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Content-adaptive encoding (CAE) policy vocabulary and its JSON configuration.
//
// A policy map assigns one codec-independent intent to each cell of the model's 60x107 mask grid.
// The application never names QP values or codec block sizes; a downstream converter translates
// these intents for a specific encoder profile.

#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace endoscopy_tool_tracking {

// Wire values of the U8 policy map. Numeric order carries no priority; see policy_rank().
enum class CaePolicy : std::uint8_t {
  kNeutral = 0,
  kProtect = 1,
  kProtectStrong = 2,
  kDeprioritize = 3,
  kDeprioritizeStrong = 4,
};

inline constexpr std::uint8_t kMaxPolicyValue = 4U;
// Written for every cell whose model output is invalid. Consumers must reject any value above
// kMaxPolicyValue rather than encode with a corrupted policy.
inline constexpr std::uint8_t kInvalidPolicyValue = 255U;
inline constexpr std::uint32_t kPolicySchemaVersion = 1U;

inline constexpr std::size_t kToolClasses = 7U;
inline constexpr std::size_t kPolicyHeight = 60U;
inline constexpr std::size_t kPolicyWidth = 107U;
inline constexpr std::size_t kPolicyCells = kPolicyHeight * kPolicyWidth;

// Model class order of tool_loc_convlstm.onnx; class numbers in the JSON configuration index it.
inline constexpr std::array<std::string_view, kToolClasses> kToolClassNames{
    "grasper", "bipolar",   "hook",        "scissors",
    "clipper", "irrigator", "specimen_bag"};  // NOLINT(whitespace/indent_namespace)

inline constexpr std::array<std::string_view, kMaxPolicyValue + 1U> kPolicyNames{
    "neutral", "protect", "protect_strong", "deprioritize",
    "deprioritize_strong"};  // NOLINT(whitespace/indent_namespace)

// Lets the CUDA kernel call the shared helpers below; plain C++ builds see no annotation.
#if defined(__CUDACC__)
#define ENDOSCOPY_CAE_HOST_DEVICE __host__ __device__
#else
#define ENDOSCOPY_CAE_HOST_DEVICE
#endif

// Overlap resolution: a higher rank wins, and protection always outranks every other intent:
// ProtectStrong > Protect > Neutral > Deprioritize > DeprioritizeStrong. This is the only
// definition of the order; the host code, the CUDA kernel and the tests all use it.
[[nodiscard]] ENDOSCOPY_CAE_HOST_DEVICE constexpr int policy_rank(std::uint8_t policy) {
  switch (static_cast<CaePolicy>(policy)) {
    case CaePolicy::kProtectStrong:
      return 4;
    case CaePolicy::kProtect:
      return 3;
    case CaePolicy::kNeutral:
      return 2;
    case CaePolicy::kDeprioritize:
      return 1;
    case CaePolicy::kDeprioritizeStrong:
      return 0;
  }
  return -1;  // not a policy value
}
static_assert(policy_rank(2U) > policy_rank(1U) && policy_rank(1U) > policy_rank(0U) &&
                  policy_rank(0U) > policy_rank(3U) && policy_rank(3U) > policy_rank(4U) &&
                  policy_rank(4U) > policy_rank(kMaxPolicyValue + 1U),
              "protection must outrank every other policy");

struct CaePolicyConfig {
  float class_threshold{0.5F};
  float mask_threshold{0.5F};
  std::uint8_t background_policy{static_cast<std::uint8_t>(CaePolicy::kDeprioritize)};
  std::array<std::uint8_t, kToolClasses> class_policies{};  // all Neutral
};

[[nodiscard]] inline std::string_view policy_name(std::uint8_t policy) {
  return policy <= kMaxPolicyValue ? kPolicyNames[policy] : std::string_view{"invalid"};
}

namespace cae_detail {

// Strict parser for the small JSON subset this configuration uses: objects, strings without
// escapes other than \" and \\, numbers, true, false and null. Arrays are rejected because the
// configuration has none.
struct JsonValue;
using JsonObject = std::vector<std::pair<std::string, JsonValue>>;
struct JsonValue {
  std::variant<std::nullptr_t, bool, double, std::string, JsonObject> value;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : text_(text) {}

  JsonValue parse_document() {
    JsonValue value = parse_value(0U);
    skip_space();
    if (pos_ != text_.size()) fail("unexpected trailing content");
    return value;
  }

 private:
  [[noreturn]] void fail(const std::string& message) const {
    std::size_t line = 1U;
    for (std::size_t i = 0; i < pos_ && i < text_.size(); ++i) {
      if (text_[i] == '\n') ++line;
    }
    throw std::invalid_argument("CAE policy config: " + message + " at line " +
                                std::to_string(line));
  }

  // JSON whitespace is exactly space, tab, line feed and carriage return.
  void skip_space() {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' ||
                                   text_[pos_] == '\n' || text_[pos_] == '\r')) {
      ++pos_;
    }
  }

  void expect(char c) {
    skip_space();
    if (pos_ >= text_.size() || text_[pos_] != c) fail(std::string("expected '") + c + "'");
    ++pos_;
  }

  bool consume_literal(std::string_view literal) {
    if (text_.substr(pos_, literal.size()) != literal) return false;
    pos_ += literal.size();
    return true;
  }

  JsonValue parse_value(std::size_t depth) {
    if (depth > 8U) fail("nesting is too deep");
    skip_space();
    if (pos_ >= text_.size()) fail("unexpected end of input");
    const char c = text_[pos_];
    if (c == '{') return JsonValue{parse_object(depth)};
    if (c == '"') return JsonValue{parse_string()};
    if (c == '[') fail("arrays are not supported");
    if (consume_literal("true")) return JsonValue{true};
    if (consume_literal("false")) return JsonValue{false};
    if (consume_literal("null")) return JsonValue{nullptr};
    return JsonValue{parse_number()};
  }

  JsonObject parse_object(std::size_t depth) {
    expect('{');
    JsonObject object;
    skip_space();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return object;
    }
    while (true) {
      skip_space();
      std::string key = parse_string();
      for (const auto& [existing, unused] : object) {
        if (existing == key) fail("duplicate key \"" + key + "\"");
      }
      expect(':');
      object.emplace_back(std::move(key), parse_value(depth + 1U));
      skip_space();
      if (pos_ < text_.size() && text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      expect('}');
      return object;
    }
  }

  std::string parse_string() {
    skip_space();
    if (pos_ >= text_.size() || text_[pos_] != '"') fail("expected a string");
    ++pos_;
    std::string result;
    while (pos_ < text_.size() && text_[pos_] != '"') {
      char c = text_[pos_++];
      if (static_cast<unsigned char>(c) < 0x20U) fail("control character in string");
      if (c == '\\') {
        if (pos_ >= text_.size()) break;
        c = text_[pos_++];
        if (c != '"' && c != '\\') fail("unsupported string escape");
      }
      result.push_back(c);
    }
    if (pos_ >= text_.size()) fail("unterminated string");
    ++pos_;
    return result;
  }

  // Count and skip decimal digits at the current position.
  std::size_t skip_digits() {
    const std::size_t first = pos_;
    while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
    return pos_ - first;
  }

  // JSON number grammar: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?
  double parse_number() {
    const std::size_t start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
    const std::size_t integer_start = pos_;
    const std::size_t integer_digits = skip_digits();
    bool valid = integer_digits == 1U || (integer_digits > 1U && text_[integer_start] != '0');
    if (valid && pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      valid = skip_digits() > 0U;
    }
    if (valid && pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      valid = skip_digits() > 0U;
    }
    double value{};
    const auto [end, error] = std::from_chars(text_.data() + start, text_.data() + pos_, value);
    if (!valid || error != std::errc() || end != text_.data() + pos_ || !std::isfinite(value)) {
      pos_ = start;
      fail("expected a value");
    }
    return value;
  }

  std::string_view text_;
  std::size_t pos_{};
};

[[nodiscard]] inline std::string lower(std::string_view text) {
  std::string result(text);
  for (char& c : result) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return result;
}

// Decimal digits only, without sign or leading zeros (as in the JSON number grammar).
[[nodiscard]] inline std::optional<std::uint32_t> exact_unsigned(std::string_view text) {
  if (text.size() > 1U && text.front() == '0') return std::nullopt;
  std::uint32_t value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || error != std::errc() || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] inline std::uint8_t parse_policy(const JsonValue& value, const std::string& where) {
  if (const auto* name = std::get_if<std::string>(&value.value)) {
    const std::string key = lower(*name);
    for (std::size_t i = 0; i < kPolicyNames.size(); ++i) {
      if (kPolicyNames[i] == key) return static_cast<std::uint8_t>(i);
    }
    if (const auto number = exact_unsigned(key); number && *number <= kMaxPolicyValue) {
      return static_cast<std::uint8_t>(*number);
    }
  } else if (const auto* number = std::get_if<double>(&value.value)) {
    if (*number >= 0.0 && *number <= kMaxPolicyValue && std::floor(*number) == *number) {
      return static_cast<std::uint8_t>(*number);
    }
  }
  throw std::invalid_argument("CAE policy config: " + where +
                              " must be one of neutral, protect, protect_strong, deprioritize, "
                              "deprioritize_strong, or 0..4");
}

[[nodiscard]] inline float parse_threshold(const JsonValue& value, const std::string& where) {
  const auto* number = std::get_if<double>(&value.value);
  if (number == nullptr || !(*number >= 0.0 && *number <= 1.0)) {
    throw std::invalid_argument("CAE policy config: " + where + " must be a number in [0, 1]");
  }
  return static_cast<float>(*number);
}

// Resolve a class key: a class name (case-insensitive) or a class number 0..6.
[[nodiscard]] inline std::size_t parse_class(std::string_view key) {
  const std::string name = lower(key);
  for (std::size_t i = 0; i < kToolClassNames.size(); ++i) {
    if (kToolClassNames[i] == name) return i;
  }
  if (const auto number = exact_unsigned(name); number && *number < kToolClasses) {
    return *number;
  }
  std::string names;
  for (const auto& known : kToolClassNames) {
    names += (names.empty() ? "" : ", ") + std::string(known);
  }
  throw std::invalid_argument("CAE policy config: unknown class \"" + std::string(key) +
                              "\"; use a class number 0..6 or one of " + names);
}

}  // namespace cae_detail

// Parse a configuration document. Omitted fields keep the defaults of CaePolicyConfig.
[[nodiscard]] inline CaePolicyConfig parse_cae_policy_config(std::string_view text) {
  using cae_detail::JsonObject;
  using cae_detail::JsonParser;
  using cae_detail::JsonValue;
  using cae_detail::parse_class;
  using cae_detail::parse_policy;
  using cae_detail::parse_threshold;
  if (text.starts_with("\xEF\xBB\xBF")) {
    throw std::invalid_argument(
        "CAE policy config: remove the UTF-8 byte order mark at the start of the file");
  }
  const JsonValue document = JsonParser(text).parse_document();
  const auto* root = std::get_if<JsonObject>(&document.value);
  if (root == nullptr) throw std::invalid_argument("CAE policy config: expected a JSON object");

  CaePolicyConfig config;
  bool version_seen = false;
  for (const auto& [key, value] : *root) {
    if (key == "schema_version") {
      const auto* number = std::get_if<double>(&value.value);
      if (number == nullptr || *number != static_cast<double>(kPolicySchemaVersion)) {
        throw std::invalid_argument("CAE policy config: schema_version must be " +
                                    std::to_string(kPolicySchemaVersion));
      }
      version_seen = true;
    } else if (key == "class_threshold") {
      config.class_threshold = parse_threshold(value, key);
    } else if (key == "mask_threshold") {
      config.mask_threshold = parse_threshold(value, key);
    } else if (key == "background_policy") {
      config.background_policy = parse_policy(value, key);
    } else if (key == "classes") {
      const auto* classes = std::get_if<JsonObject>(&value.value);
      if (classes == nullptr) {
        throw std::invalid_argument("CAE policy config: classes must be an object");
      }
      std::array<bool, kToolClasses> seen{};
      for (const auto& [class_key, policy] : *classes) {
        const std::size_t index = parse_class(class_key);
        if (seen[index]) {
          throw std::invalid_argument("CAE policy config: class " + std::to_string(index) + " (" +
                                      std::string(kToolClassNames[index]) +
                                      ") is assigned more than once");
        }
        seen[index] = true;
        config.class_policies[index] = parse_policy(policy, "classes." + class_key);
      }
    } else {
      throw std::invalid_argument("CAE policy config: unknown field \"" + key + "\"");
    }
  }
  if (!version_seen) throw std::invalid_argument("CAE policy config: schema_version is required");
  return config;
}

[[nodiscard]] inline CaePolicyConfig load_cae_policy_config(const std::string& path) {
  // Reject directories and other non-files explicitly; reading one would otherwise surface as a
  // misleading JSON syntax error.
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    throw std::invalid_argument("cannot read CAE policy config: " + path +
                                " is not a readable regular file");
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::invalid_argument("cannot read CAE policy config: " + path);
  std::ostringstream text;
  text << file.rdbuf();
  if (file.bad()) throw std::invalid_argument("cannot read CAE policy config: " + path);
  return parse_cae_policy_config(text.str());
}

// One-line summary for startup logs and dump manifests.
[[nodiscard]] inline std::string describe(const CaePolicyConfig& config) {
  std::ostringstream text;
  text << "class_threshold=" << config.class_threshold
       << " mask_threshold=" << config.mask_threshold
       << " background=" << policy_name(config.background_policy) << " classes=";
  for (std::size_t i = 0; i < kToolClasses; ++i) {
    text << (i ? "," : "") << kToolClassNames[i] << ':' << policy_name(config.class_policies[i]);
  }
  return text.str();
}

}  // namespace endoscopy_tool_tracking
