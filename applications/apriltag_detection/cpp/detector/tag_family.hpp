// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Owns one AprilTag 3 family (its code book and bit layout) by name.

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

struct apriltag_family;

namespace tag_detection {

struct TagFamilyDeleter {
  void (*destroy)(apriltag_family*) = nullptr;
  void operator()(apriltag_family* family) const noexcept;
};

// Every family the AprilTag 3 library ships, which is also the set VPI supports.
[[nodiscard]] std::span<const std::string_view> family_names() noexcept;

[[nodiscard]] bool is_known_family(std::string_view name) noexcept;

// Codes plus every variant within `max_hamming` bits: what a decoder must index.
// @throws std::invalid_argument for an unknown family.
[[nodiscard]] std::uint64_t decode_table_entries(std::string_view family, int max_hamming);

class TagFamily {
 public:
  // @throws std::invalid_argument if `name` is not one of family_names().
  explicit TagFamily(std::string_view name);
  ~TagFamily();
  TagFamily(const TagFamily&) = delete;
  TagFamily& operator=(const TagFamily&) = delete;
  TagFamily(TagFamily&&) noexcept = default;
  TagFamily& operator=(TagFamily&&) noexcept = default;

  [[nodiscard]] apriltag_family* get() const noexcept { return family_.get(); }
  [[nodiscard]] std::string_view name() const noexcept { return name_; }

 private:
  std::unique_ptr<apriltag_family, TagFamilyDeleter> family_;
  std::string_view name_;
};

}  // namespace tag_detection
