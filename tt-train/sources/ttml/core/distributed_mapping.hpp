// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_all_includes.hpp>
#include <unordered_map>

#include "core/xtensor_utils.hpp"

namespace ttml::core {
template <typename T>
std::vector<xt::xarray<T>> chunk(const xt::xarray<T>& tensor, int num_chunks, int dim) {
    if (num_chunks <= 0) {
        throw std::invalid_argument("num_chunks must be > 0");
    }
    if (dim < 0 || static_cast<std::size_t>(dim) >= tensor.dimension()) {
        throw std::invalid_argument("invalid dimension index");
    }

    int size_along_dim = static_cast<int>(tensor.shape()[dim]);
    if (num_chunks > size_along_dim) {
        throw std::invalid_argument("num_chunks cannot exceed the size of the tensor along the given dimension.");
    }

    if (num_chunks == 1) {
        return {tensor};
    }

    int chunk_size = (size_along_dim + num_chunks - 1) / num_chunks;
    int remaining_size = size_along_dim;

    std::vector<xt::xarray<T>> chunks;
    chunks.reserve(static_cast<std::size_t>(num_chunks));

    int start = 0;
    int end = 0;
    for (int i = 0; i < num_chunks && end < size_along_dim; ++i) {
        int current_chunk_size = std::min(chunk_size, remaining_size);
        remaining_size -= current_chunk_size;
        end = start + current_chunk_size;

        // Build indices for slicing
        xt::xstrided_slice_vector indices(tensor.dimension(), xt::all());
        indices[dim] = xt::range(start, end);

        auto chunk_view = xt::strided_view(tensor, indices);

        // Construct xarray from the view
        // This forces a copy of that slice into a new xarray
        chunks.push_back(xt::xarray<T>(chunk_view));
        start = end;
    }

    return chunks;
}

template <class Derived, typename T>
class XTensorToMesh {
public:
    XTensorToMesh(tt::tt_metal::distributed::MeshShape mesh_shape) : m_mesh_shape(std::move(mesh_shape)) {
    }

    std::vector<xt::xarray<T>> map(const xt::xarray<T>& tensor) const {
        return static_cast<Derived const*>(this)->map_impl(tensor);
    }

    std::unordered_map<std::string, std::string> config() const {
        return static_cast<Derived const*>(this)->config_impl();
    }

protected:
    tt::tt_metal::distributed::MeshShape m_mesh_shape;

    size_t get_num_devices() const {
        return m_mesh_shape.num_rows * m_mesh_shape.num_cols;
    }
};

template <class Derived, typename T>
class MeshToXTensor {
public:
    MeshToXTensor(tt::tt_metal::distributed::MeshShape mesh_shape) : m_mesh_shape(std::move(mesh_shape)) {
    }

    std::vector<xt::xarray<T>> compose(const std::vector<xt::xarray<T>>& tensors) const {
        return static_cast<Derived const*>(this)->compose_impl(tensors);
    }

protected:
    tt::tt_metal::distributed::MeshShape m_mesh_shape;
};

template <typename T>
class ShardXTensorToMesh : public XTensorToMesh<ShardXTensorToMesh<T>, T> {
public:
    using Base = XTensorToMesh<ShardXTensorToMesh<T>, T>;
    ShardXTensorToMesh(tt::tt_metal::distributed::MeshShape mesh_shape, int dim) :
        Base(std::move(mesh_shape)), m_shard_dim(dim) {
    }

    std::vector<xt::xarray<T>> map_impl(const xt::xarray<T>& tensor) const {
        int num_devices = Base::get_num_devices();
        auto sliced_tensors = chunk(tensor, num_devices, m_shard_dim);
        return sliced_tensors;
    }

    std::unordered_map<std::string, std::string> config_impl() const {
        return {{"strategy", "shard"}, {"shard_dim", std::to_string(m_shard_dim)}};
    }

private:
    int m_shard_dim = 0;
};

template <typename T>
class ShardTensor2dMesh : public XTensorToMesh<ShardTensor2dMesh<T>, T> {
public:
    using Base = XTensorToMesh<ShardTensor2dMesh<T>, T>;
    ShardTensor2dMesh(
        tt::tt_metal::distributed::MeshShape mesh_shape,
        const std::pair<std::optional<int>, std::optional<int>>& dims) :
        Base(std::move(mesh_shape)), m_dims(dims) {
        // We trust the provided mesh shape and do not validate against a MeshDevice.
    }

    std::vector<xt::xarray<T>> map_impl(const xt::xarray<T>& tensor) const {
        if (!m_dims.first.has_value() && !m_dims.second.has_value()) {
            throw std::invalid_argument("ShardTensor2dMesh requires at least one dimension to shard");
        }

        int rows = Base::m_mesh_shape.num_rows;
        int cols = Base::m_mesh_shape.num_cols;
        auto row_dim = m_dims.first;
        auto col_dim = m_dims.second;

        std::vector<xt::xarray<T>> row_tensors;

        // Shard along rows
        if (!row_dim.has_value()) {
            row_tensors.reserve(rows);
            for (int i = 0; i < rows; ++i) {
                row_tensors.push_back(tensor);
            }
        } else {
            row_tensors = chunk(tensor, rows, row_dim.value());
        }

        std::vector<xt::xarray<T>> tensor_shards;
        tensor_shards.reserve(static_cast<size_t>(rows * cols));
        // Shard along columns
        if (!col_dim.has_value()) {
            for (const auto& t : row_tensors) {
                for (int i = 0; i < cols; ++i) {
                    tensor_shards.push_back(t);
                }
            }
        } else {
            for (const auto& t : row_tensors) {
                auto col_chunks = chunk(t, cols, col_dim.value());
                tensor_shards.insert(tensor_shards.end(), col_chunks.begin(), col_chunks.end());
            }
        }

        if (static_cast<int>(tensor_shards.size()) != rows * cols) {
            throw std::runtime_error(fmt::format(
                "ShardTensor2dMesh: Sharding failed. Number of shards should match the product of the mesh "
                "dimensions. Size: {}, rows: {}, cols: {}",
                tensor_shards.size(),
                rows,
                cols));
        }

        return tensor_shards;
    }

    std::unordered_map<std::string, std::string> config_impl() const {
        return {
            {"strategy", "shard_2d"},
            {"mesh_shape_y", std::to_string(Base::m_mesh_shape.num_rows)},
            {"mesh_shape_x", std::to_string(Base::m_mesh_shape.num_cols)}};
    }

private:
    std::pair<std::optional<int>, std::optional<int>> m_dims;
};

template <typename T>
class ConcatMesh2dToTensor : public MeshToXTensor<ConcatMesh2dToTensor<T>, T> {
public:
    using Base = MeshToXTensor<ConcatMesh2dToTensor<T>, T>;
    ConcatMesh2dToTensor(
        tt::tt_metal::distributed::MeshShape mesh_shape, const tt::tt_metal::distributed::MeshShape& dims) :
        Base(std::move(mesh_shape)), m_dims(dims) {
        if (m_dims.num_rows == m_dims.num_cols) {
            throw std::invalid_argument("Dimensions in 'dims' must be different");
        }
    }

    std::vector<xt::xarray<T>> compose_impl(const std::vector<xt::xarray<T>>& tensors) const {
        int rows = Base::m_mesh_shape.num_rows;
        int cols = Base::m_mesh_shape.num_cols;
        size_t row_dim = m_dims.num_rows;
        size_t col_dim = m_dims.num_cols;

        std::vector<xt::xarray<T>> row_concatenated;
        row_concatenated.reserve(static_cast<size_t>(rows));

        for (int i = 0; i < rows; ++i) {
            auto row_start = tensors.begin() + i * cols;
            auto row_end = row_start + cols;
            std::vector<xt::xarray<T>> row_tensors(row_start, row_end);

            auto concatenated_row = core::concatenate(row_tensors, col_dim);
            row_concatenated.push_back(std::move(concatenated_row));
        }

        auto result = core::concatenate(row_concatenated, row_dim);
        return {result};
    }

private:
    tt::tt_metal::distributed::MeshShape m_dims;
};

template <typename T>
class ReplicateXTensorToMesh : public XTensorToMesh<ReplicateXTensorToMesh<T>, T> {
public:
    using Base = XTensorToMesh<ReplicateXTensorToMesh<T>, T>;
    ReplicateXTensorToMesh(tt::tt_metal::distributed::MeshShape mesh_shape) : Base(std::move(mesh_shape)) {
    }

    std::vector<xt::xarray<T>> map_impl(const xt::xarray<T>& tensor) const {
        int num_devices = Base::get_num_devices();
        std::vector<xt::xarray<T>> tensors;
        tensors.reserve(static_cast<size_t>(num_devices));
        for (int i = 0; i < num_devices; ++i) {
            tensors.push_back(tensor);  // Note: this copies the tensor
        }
        return tensors;
    }

    std::unordered_map<std::string, std::string> config_impl() const {
        int num_devices = Base::get_num_devices();
        return {{"strategy", "replicate"}, {"replication_factor", std::to_string(num_devices)}};
    }
};

template <typename T>
class ConcatMeshToXTensor : public MeshToXTensor<ConcatMeshToXTensor<T>, T> {
public:
    using Base = MeshToXTensor<ConcatMeshToXTensor<T>, T>;
    ConcatMeshToXTensor(tt::tt_metal::distributed::MeshShape mesh_shape, int dim) :
        Base(std::move(mesh_shape)), m_concat_dim(dim) {
    }

    std::vector<xt::xarray<T>> compose_impl(const std::vector<xt::xarray<T>>& tensors) const {
        return {core::concatenate(tensors, m_concat_dim)};
    }

private:
    int m_concat_dim = 0;
};

template <typename T>
class VectorMeshToXTensor : public MeshToXTensor<VectorMeshToXTensor<T>, T> {
public:
    using Base = MeshToXTensor<VectorMeshToXTensor<T>, T>;
    VectorMeshToXTensor([[maybe_unused]] tt::tt_metal::distributed::MeshShape mesh_shape) : Base(mesh_shape) {
    }
    std::vector<xt::xarray<T>> compose_impl(const std::vector<xt::xarray<T>>& tensors) const {
        return tensors;
    }
};

template <typename T>
using XTensorToMeshVariant = std::variant<ShardXTensorToMesh<T>, ShardTensor2dMesh<T>, ReplicateXTensorToMesh<T>>;

template <typename T>
using MeshToXTensorVariant = std::variant<ConcatMeshToXTensor<T>, ConcatMesh2dToTensor<T>, VectorMeshToXTensor<T>>;

}  // namespace ttml::core