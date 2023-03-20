import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from pymetal import ttmetal
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close

RSUM = ttmetal.tensor.ReduceOpMath.SUM
RW = ttmetal.tensor.ReduceOpDim.W
RH = ttmetal.tensor.ReduceOpDim.H
RHW = ttmetal.tensor.ReduceOpDim.HW

# Initialize the device
device = ttmetal.device.CreateDevice(ttmetal.device.Arch.GRAYSKULL, 0)
ttmetal.device.InitializeDevice(device)
host = ttmetal.device.GetHost()
ttmetal.device.StartDebugPrintServer(device)

if __name__ == "__main__":
    N = 2
    C = 3
    H = 32*5
    W = 32*3
    torch.manual_seed(123)
    x = (torch.randn((N,C,H,W))+0.01).to(torch.bfloat16)

    reduce_dims_tt = [RW, RH, RHW]
    reduce_dims_pyt = [[3], [2], [3,2]]
    reduce_shapes = [[N, C, H, 32], [N, C, 32, W], [N, C, 32, 32]]
    for rtype, expected_shape, rdims_pyt in zip(reduce_dims_tt, reduce_shapes, reduce_dims_pyt):
        xt = ttmetal.tensor.Tensor(tilize_to_list(x), [N, C, H, W], ttmetal.tensor.DataType.BFLOAT16, ttmetal.tensor.Layout.TILE, device)
        mul = 0.5
        if rtype == RHW:
            mul = 1.0
        tt_res = ttmetal.tensor.reduce(xt, RSUM, rtype, mul)
        assert(tt_res.shape() == expected_shape)
        tt_host_rm = tt_res.to(host).data()

        pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape(expected_shape)
        pyt_got_back_rm = untilize(pyt_got_back_rm)

        ref = x.to(torch.bfloat16).sum(rdims_pyt, keepdim=True)*mul
        if rtype == RW:
            ref_padded = torch.zeros(pyt_got_back_rm.shape)
            ref_padded[:,:,:,0:1] = ref
        elif rtype == RH:
            ref_padded = torch.zeros(pyt_got_back_rm.shape)
            ref_padded[:,:,0:1,:] = ref
        elif rtype == RHW:
            ref_padded = torch.zeros(pyt_got_back_rm.shape)
            ref_padded[:,:,0:1,0:1] = ref

        allok = is_close(pyt_got_back_rm, ref_padded, rtol=0.07, atol=0.3)
        if not allok:
            print_diff_argmax(pyt_got_back_rm, ref_padded)

        assert(allok)

ttmetal.device.CloseDevice(device)
