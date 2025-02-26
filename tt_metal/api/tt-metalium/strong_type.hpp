// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <ostream>

namespace tt::stl {

// `StrongType` provides a strongly-typed wrapper around a value to prevent accidental type conversions.
//
// This is useful when creating aliases that rely on a primitive type; for example instead of using `uint32_t` as
// `DeviceId` directly, wrap in `StrongType<uint32_t, struct DeviceIdTag>` to prevent accidental assignment
// from `uint32_t`. Here, the 'tag' is used to disambiguate the type, and to create distinct wrappers relying on
// `uint32_t`.
//
//
// Example usage:
//
// // Create strong types.
// // `struct`s for the tag can be supplied as shown, despite of being incomplete:
//
// using UserId = StrongType<uint32_t, struct UserIdTag>;
// using GroupId = StrongType<uint32_t, struct GroupIdTag>;
// using Username = StrongType<std::string, struct UsernameTag>;
//
//
// // The different types cannot be assigned to each other:
// UserId user_id(42);
// GroupId group_id(45);
// user_id = group_id;  // does not compile!
//
// Username name("john_doe");
// name = "jane_doe";            // does not compile!
// name = Username("jane_doe");  // instantiate explicitly.
//
// // Access the underlying value:
// uint32_t raw_user_id = *user_id;
// assert(*user_id < *group_id);
//
// // Strong types work with standard containers and the streaming operator, as long as the underlying type is
// // hashable and comparable.
//
// std::unordered_set<UserId> user_set;
// user_set.insert(UserId(1));
// user_set.insert(UserId(2));

// std::map<UserId, Username> user_map;
// user_map.emplace(UserId(1), Username("John Doe"));
//
// std::cout << user_map.at(UserId(1)) << std::endl;  // "John Doe"
//
template <typename T, typename Tag>
class StrongType {
public:
    constexpr explicit StrongType(T v) : value_(std::move(v)) {}

    StrongType(const StrongType&) = default;
    StrongType(StrongType&&) = default;
    StrongType& operator=(const StrongType&) = default;
    StrongType& operator=(StrongType&&) = default;

    const T& operator*() const { return value_; }

    auto operator<=>(const StrongType&) const = default;

private:
    T value_;
};

}  // namespace tt::stl

template <typename T, typename Tag>
std::ostream& operator<<(std::ostream& os, const tt::stl::StrongType<T, Tag>& h) {
    return os << *h;
}

template <typename T, typename Tag>
struct std::hash<tt::stl::StrongType<T, Tag>> {
    std::size_t operator()(const tt::stl::StrongType<T, Tag>& h) const noexcept { return std::hash<T>{}(*h); }
};
