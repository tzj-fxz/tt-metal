// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "shape_base.hpp"

namespace tt::tt_metal {

class SimpleShape final : protected ShapeBase {
public:
    using ShapeBase::ShapeBase;
    using ShapeBase::operator[];
    using ShapeBase::cbegin;
    using ShapeBase::cend;
    using ShapeBase::view;
    using ShapeBase::size;
    using ShapeBase::empty;

    template<std::size_t N>
    bool operator==(const std::array<uint32_t, N> &other) const {
        const bool sameSize = value_.size() == N;
        return sameSize && std::equal(value_.begin(), value_.end(), other.begin());
    }

    bool operator==(const SimpleShape &other) const;
    bool operator==(const ShapeBase::Container &other) const;


    [[nodiscard]] size_t rank() const;
    [[nodiscard]] uint64_t volume() const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(this->value_); }

    friend std::ostream &operator<<(std::ostream &os, const SimpleShape &shape);
};

std::ostream &operator<<(std::ostream &os, const tt::tt_metal::SimpleShape &shape);

} // namespace tt::tt_metal

namespace ttnn {
    using tt::tt_metal::SimpleShape;
} // namespace ttnn
