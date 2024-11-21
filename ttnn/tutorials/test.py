import ttnn, torch


def main():
    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    torch.manual_seed(0)

    m, n, k = 1024, 1024, 1024

    torch_tensor_a = torch.randn((m, k), dtype=torch.float16)
    torch_tensor_b = torch.randn((k, n), dtype=torch.float16)

    ttnn_tensor_a = ttnn.from_torch(torch_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_tensor_b = ttnn.from_torch(torch_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    add_tensor = ttnn.add(ttnn_tensor_a, ttnn_tensor_b)

    gelu_tensor = ttnn.gelu(add_tensor)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
