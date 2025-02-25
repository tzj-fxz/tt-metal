// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

namespace ttml::ops::tests {

class UnaryOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(UnaryOpsTest, GlobalMean) {
    std::vector<float> test_data = {1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F};

    auto shape = core::create_shape({2, 1, 1, 4});
    auto tensor = core::from_vector(test_data, shape, &autograd::ctx().get_device());

    auto tensor_ptr = autograd::create_tensor(tensor);

    auto result = mean(tensor_ptr);
    auto result_data = core::to_vector(result->get_value());

    ASSERT_EQ(result_data.size(), 1);
    EXPECT_FLOAT_EQ(result_data[0], 2.5F);

    result->backward();
    auto tensor_grad = core::to_vector(tensor_ptr->get_grad());
    ASSERT_EQ(tensor_grad.size(), test_data.size());
    for (float it : tensor_grad) {
        EXPECT_FLOAT_EQ(it, 0.125F);
    }
}

TEST_F(UnaryOpsTest, LogSoftmax) {
    auto* device = &autograd::ctx().get_device();
    std::vector<float> test_data = {-0.1F, -0.2F, -0.3F, -0.4F, 0.F, -0.2F, -0.3F, -0.4F};
    auto tensor = core::from_vector(test_data, core::create_shape({2, 1, 1, 4}), device);
    auto tensor_ptr = autograd::create_tensor(tensor);
    auto result = log_softmax_moreh(tensor_ptr, 3);
    auto result_data = core::to_vector(result->get_value());
    std::vector<float> expected_data = {
        -1.24253553F, -1.34253553F, -1.44253553F, -1.54253553F, -1.17244159F, -1.37244159F, -1.47244159F, -1.57244159F};
    EXPECT_EQ(result_data.size(), expected_data.size());
    for (uint32_t idx = 0; idx < result_data.size(); ++idx) {
        EXPECT_NEAR(result_data[idx], expected_data[idx], 2e-2F);
    }

    result->backward();
    auto tensor_grad = core::to_vector(tensor_ptr->get_grad());
    std::vector<float> expected_grad = {-0.156F, -0.03906F, 0.05078F, 0.1406F, -0.25F, -0.0156F, 0.07421F, 0.16406F};
    EXPECT_EQ(tensor_grad.size(), expected_grad.size());
    for (uint32_t idx = 0; idx < tensor_grad.size(); ++idx) {
        EXPECT_NEAR(tensor_grad[idx], expected_grad[idx], 2e-2F);
    }
}

TEST_F(UnaryOpsTest, Silu) {
    auto N = 4;
    auto C = 1;
    auto H = 20;
    auto W = 5;
    auto len = static_cast<float>(N * C * H * W);
    xt::random::seed(42);
    xt::xarray<float> a = xt::random::rand<float>({N, C, H, W}, -1.0F, 1.0F);
    xt::xarray<float> expected_silu = {
        {{{-0.10980F, 0.38199F, 0.64114F, -0.21957F, 0.28487F},
          {0.35594F, 0.10836F, 0.10620F, -0.23011F, -0.05124F},
          {-0.23012F, -0.24803F, -0.25842F, -0.03909F, 0.49457F},
          {-0.13889F, 0.11130F, -0.23475F, 0.25075F, 0.17348F},
          {-0.26570F, -0.25878F, 0.67579F, 0.27049F, 0.43906F},
          {0.61943F, -0.20712F, -0.26883F, -0.22022F, 0.71665F},
          {-0.21958F, 0.13122F, -0.15792F, 0.12407F, 0.02537F},
          {-0.26789F, -0.06343F, -0.26528F, -0.16581F, 0.02539F},
          {0.12431F, -0.09014F, -0.23589F, -0.26083F, -0.16526F},
          {0.68279F, -0.11588F, -0.19747F, -0.04200F, -0.25057F},
          {0.36437F, 0.13234F, -0.21275F, -0.10379F, 0.01444F},
          {0.70012F, 0.10093F, -0.03213F, -0.26088F, 0.48418F},
          {0.11907F, 0.21247F, -0.22469F, -0.04705F, -0.25686F},
          {-0.26692F, 0.63786F, 0.62592F, 0.66803F, 0.06729F},
          {0.40060F, -0.10151F, -0.15769F, -0.26648F, -0.24866F},
          {-0.19839F, 0.21780F, -0.19337F, -0.05627F, 0.21648F},
          {-0.24154F, 0.12205F, -0.00480F, 0.44028F, -0.26324F},
          {-0.22358F, 0.56809F, -0.09712F, -0.18414F, -0.22006F},
          {0.18871F, 0.31919F, -0.15325F, -0.06925F, 0.02047F},
          {-0.20911F, 0.04889F, 0.07228F, -0.21899F, -0.26381F}}},
        {{{0.67520F, 0.45507F, 0.34898F, -0.04772F, 0.62111F},
          {-0.09390F, 0.54309F, 0.59840F, 0.10745F, 0.27805F},
          {0.58999F, -0.14367F, -0.25112F, 0.07540F, -0.21434F},
          {0.02127F, -0.26112F, 0.65996F, -0.14447F, 0.45875F},
          {-0.09898F, 0.30727F, -0.17726F, 0.04127F, 0.43307F},
          {0.09426F, -0.12287F, 0.66734F, -0.17183F, 0.11845F},
          {0.04452F, -0.17465F, -0.23541F, -0.16279F, 0.39084F},
          {-0.22669F, -0.25463F, -0.26653F, 0.70683F, -0.07074F},
          {0.34458F, -0.09411F, -0.21316F, -0.16446F, -0.26812F},
          {-0.26678F, 0.41180F, -0.21311F, 0.24905F, 0.25535F},
          {0.28055F, 0.37209F, 0.34310F, 0.11715F, -0.25475F},
          {0.59777F, -0.12164F, 0.17373F, -0.24343F, 0.57790F},
          {0.48944F, 0.46779F, 0.13842F, -0.04800F, -0.14078F},
          {-0.24928F, -0.25720F, -0.11259F, -0.15371F, 0.19708F},
          {-0.14456F, 0.19320F, 0.28142F, 0.09961F, 0.15636F},
          {-0.17537F, 0.53008F, 0.06499F, -0.02701F, -0.10343F},
          {-0.24230F, 0.67907F, 0.25804F, 0.46594F, 0.32729F},
          {0.27010F, 0.06503F, -0.19589F, 0.34264F, -0.18558F},
          {-0.00617F, -0.26208F, 0.02325F, 0.25440F, -0.06722F},
          {-0.24491F, -0.26487F, -0.05699F, -0.24578F, -0.21186F}}},
        {{{-0.26378F, 0.54470F, 0.15490F, -0.02402F, -0.15157F},
          {0.06727F, 0.00864F, 0.23326F, 0.56505F, -0.23595F},
          {-0.18914F, 0.11528F, -0.08161F, 0.04143F, 0.31947F},
          {-0.21127F, -0.19940F, 0.62708F, -0.25404F, 0.10861F},
          {-0.16668F, 0.23225F, -0.22821F, 0.51862F, 0.60375F},
          {0.13974F, 0.40016F, -0.16317F, 0.15110F, -0.24647F},
          {0.50343F, -0.04158F, 0.39315F, -0.20431F, -0.21829F},
          {-0.07654F, 0.53920F, 0.52339F, 0.04089F, -0.14511F},
          {0.39909F, -0.24153F, 0.54526F, -0.12319F, -0.14923F},
          {0.56377F, -0.24515F, -0.17682F, -0.19982F, 0.16935F},
          {-0.06759F, -0.26887F, 0.41587F, -0.12585F, 0.48549F},
          {-0.15759F, -0.26791F, -0.22692F, 0.01086F, 0.03525F},
          {-0.07578F, -0.01494F, -0.20260F, 0.22902F, -0.24221F},
          {-0.17834F, -0.13625F, -0.19180F, 0.62718F, -0.22554F},
          {-0.14586F, -0.20416F, 0.01914F, 0.06147F, 0.24368F},
          {-0.08694F, -0.11789F, -0.25690F, 0.67920F, -0.18672F},
          {0.66226F, -0.19039F, -0.18784F, 0.23435F, -0.00274F},
          {0.25666F, -0.15999F, -0.23294F, -0.16957F, 0.72687F},
          {-0.26276F, -0.17979F, 0.12152F, 0.68801F, 0.00269F},
          {-0.08107F, -0.25984F, -0.26348F, -0.17314F, -0.13112F}}},
        {{{0.56626F, 0.15229F, -0.19410F, 0.21301F, -0.23405F},  {0.03189F, -0.01044F, -0.04949F, 0.70456F, 0.05569F},
          {-0.19285F, 0.10126F, 0.20148F, -0.25308F, 0.32854F},  {-0.11345F, -0.19507F, -0.19279F, 0.27941F, 0.39232F},
          {-0.11484F, -0.02882F, 0.14971F, 0.70047F, 0.15125F},  {-0.09097F, 0.03705F, 0.41335F, -0.25065F, 0.38480F},
          {0.44370F, -0.23201F, -0.14744F, 0.00827F, -0.21831F}, {0.23367F, -0.26201F, 0.48155F, 0.09913F, -0.14405F},
          {0.20877F, -0.20347F, -0.26637F, 0.25508F, 0.01224F},  {0.40235F, -0.20051F, -0.12861F, 0.16610F, -0.24907F},
          {-0.22319F, 0.62293F, 0.22696F, -0.09197F, -0.10049F}, {0.01807F, 0.61620F, 0.44761F, -0.23656F, 0.20624F},
          {-0.13388F, 0.28954F, -0.24414F, -0.20860F, 0.59494F}, {0.04316F, 0.51333F, 0.23363F, -0.18458F, -0.19952F},
          {0.18536F, -0.22296F, 0.41461F, 0.69817F, 0.05825F},   {0.01691F, 0.03053F, -0.18303F, -0.19295F, 0.72412F},
          {-0.24990F, 0.66764F, 0.54719F, 0.06169F, 0.55270F},   {0.52230F, 0.15071F, -0.21740F, -0.13528F, -0.17301F},
          {-0.12822F, 0.23997F, 0.27616F, 0.46224F, 0.54701F},   {0.47818F, 0.52986F, -0.08640F, 0.35622F, 0.53103F}}}};

    auto a_tensor = autograd::create_tensor(core::from_xtensor(a, &autograd::ctx().get_device()));
    auto computed_silu = silu(a_tensor);
    auto computed_silu_xtensor = core::to_xtensor(computed_silu->get_value());
    EXPECT_TRUE(xt::allclose(computed_silu_xtensor, expected_silu, 8e-3F, 4e-2F));

    xt::xarray<float> expected_silu_grad_ = {
        {{{-0.00021F, 0.00149F, 0.00287F, -0.00022F, 0.00103F},
          {0.00136F, 0.00032F, 0.00032F, -0.00021F, -0.00011F},
          {-0.00021F, -0.00017F, -0.00014F, -0.00009F, 0.00207F},
          {-0.00023F, 0.00033F, -0.00020F, 0.00088F, 0.00056F},
          {-0.00011F, -0.00014F, 0.00307F, 0.00097F, 0.00178F},
          {0.00275F, -0.00024F, -0.00010F, -0.00022F, 0.00331F},
          {-0.00022F, 0.00040F, -0.00024F, 0.00038F, 0.00007F},
          {-0.00010F, -0.00014F, -0.00011F, -0.00025F, 0.00007F},
          {0.00038F, -0.00018F, -0.00020F, -0.00013F, -0.00025F},
          {0.00311F, -0.00021F, -0.00024F, -0.00010F, -0.00017F},
          {0.00140F, 0.00041F, -0.00023F, -0.00020F, 0.00004F},
          {0.00321F, 0.00030F, -0.00007F, -0.00013F, 0.00201F},
          {0.00036F, 0.00072F, -0.00022F, -0.00011F, -0.00015F},
          {-0.00011F, 0.00285F, 0.00279F, 0.00303F, 0.00019F},
          {0.00158F, -0.00020F, -0.00024F, -0.00011F, -0.00017F},
          {-0.00024F, 0.00074F, -0.00024F, -0.00012F, 0.00074F},
          {-0.00019F, 0.00037F, -0.00001F, 0.00178F, -0.00012F},
          {-0.00022F, 0.00247F, -0.00019F, -0.00025F, -0.00022F},
          {0.00062F, 0.00119F, -0.00024F, -0.00015F, 0.00005F},
          {-0.00023F, 0.00013F, 0.00021F, -0.00022F, -0.00012F}}},
        {{{0.00307F, 0.00186F, 0.00133F, -0.00011F, 0.00276F},
          {-0.00019F, 0.00233F, 0.00263F, 0.00032F, 0.00100F},
          {0.00259F, -0.00024F, -0.00016F, 0.00021F, -0.00023F},
          {0.00006F, -0.00013F, 0.00298F, -0.00024F, 0.00188F},
          {-0.00019F, 0.00113F, -0.00025F, 0.00011F, 0.00175F},
          {0.00028F, -0.00022F, 0.00302F, -0.00025F, 0.00036F},
          {0.00012F, -0.00025F, -0.00020F, -0.00025F, 0.00153F},
          {-0.00021F, -0.00015F, -0.00011F, 0.00325F, -0.00015F},
          {0.00131F, -0.00019F, -0.00023F, -0.00025F, -0.00010F},
          {-0.00011F, 0.00164F, -0.00023F, 0.00087F, 0.00090F},
          {0.00101F, 0.00144F, 0.00130F, 0.00035F, -0.00015F},
          {0.00263F, -0.00022F, 0.00056F, -0.00018F, 0.00252F},
          {0.00204F, 0.00193F, 0.00043F, -0.00011F, -0.00024F},
          {-0.00017F, -0.00015F, -0.00021F, -0.00024F, 0.00066F},
          {-0.00024F, 0.00064F, 0.00102F, 0.00029F, 0.00050F},
          {-0.00025F, 0.00226F, 0.00018F, -0.00006F, -0.00020F},
          {-0.00019F, 0.00309F, 0.00091F, 0.00192F, 0.00123F},
          {0.00097F, 0.00018F, -0.00024F, 0.00130F, -0.00025F},
          {-0.00002F, -0.00013F, 0.00006F, 0.00090F, -0.00014F},
          {-0.00018F, -0.00012F, -0.00013F, -0.00018F, -0.00023F}}},
        {{{-0.00012F, 0.00234F, 0.00049F, -0.00006F, -0.00024F},
          {0.00019F, 0.00002F, 0.00081F, 0.00245F, -0.00020F},
          {-0.00025F, 0.00035F, -0.00017F, 0.00011F, 0.00119F},
          {-0.00023F, -0.00024F, 0.00279F, -0.00016F, 0.00032F},
          {-0.00025F, 0.00080F, -0.00021F, 0.00220F, 0.00266F},
          {0.00044F, 0.00158F, -0.00025F, 0.00048F, -0.00018F},
          {0.00211F, -0.00009F, 0.00155F, -0.00024F, -0.00022F},
          {-0.00016F, 0.00231F, 0.00222F, 0.00011F, -0.00024F},
          {0.00157F, -0.00019F, 0.00234F, -0.00022F, -0.00024F},
          {0.00244F, -0.00018F, -0.00025F, -0.00024F, 0.00055F},
          {-0.00014F, -0.00010F, 0.00166F, -0.00022F, 0.00202F},
          {-0.00024F, -0.00010F, -0.00021F, 0.00003F, 0.00009F},
          {-0.00016F, -0.00004F, -0.00024F, 0.00079F, -0.00019F},
          {-0.00025F, -0.00023F, -0.00024F, 0.00279F, -0.00022F},
          {-0.00024F, -0.00024F, 0.00005F, 0.00017F, 0.00085F},
          {-0.00018F, -0.00022F, -0.00015F, 0.00309F, -0.00025F},
          {0.00299F, -0.00024F, -0.00025F, 0.00081F, -0.00001F},
          {0.00091F, -0.00024F, -0.00020F, -0.00025F, 0.00337F},
          {-0.00013F, -0.00025F, 0.00037F, 0.00314F, 0.00001F},
          {-0.00017F, -0.00014F, -0.00012F, -0.00025F, -0.00023F}}},
        {{{0.00245F, 0.00048F, -0.00024F, 0.00072F, -0.00020F},  {0.00008F, -0.00003F, -0.00011F, 0.00324F, 0.00015F},
          {-0.00024F, 0.00030F, 0.00067F, -0.00016F, 0.00123F},  {-0.00021F, -0.00024F, -0.00024F, 0.00101F, 0.00154F},
          {-0.00021F, -0.00007F, 0.00047F, 0.00321F, 0.00048F},  {-0.00018F, 0.00010F, 0.00165F, -0.00017F, 0.00150F},
          {0.00180F, -0.00021F, -0.00024F, 0.00002F, -0.00022F}, {0.00081F, -0.00013F, 0.00200F, 0.00029F, -0.00024F},
          {0.00070F, -0.00024F, -0.00011F, 0.00090F, 0.00003F},  {0.00159F, -0.00024F, -0.00023F, 0.00053F, -0.00017F},
          {-0.00022F, 0.00277F, 0.00078F, -0.00018F, -0.00019F}, {0.00005F, 0.00273F, 0.00182F, -0.00020F, 0.00069F},
          {-0.00023F, 0.00105F, -0.00018F, -0.00023F, 0.00261F}, {0.00012F, 0.00217F, 0.00081F, -0.00025F, -0.00024F},
          {0.00061F, -0.00022F, 0.00165F, 0.00320F, 0.00016F},   {0.00004F, 0.00008F, -0.00025F, -0.00024F, 0.00335F},
          {-0.00017F, 0.00302F, 0.00235F, 0.00017F, 0.00238F},   {0.00222F, 0.00048F, -0.00023F, -0.00023F, -0.00025F},
          {-0.00023F, 0.00083F, 0.00099F, 0.00190F, 0.00235F},   {0.00198F, 0.00226F, -0.00017F, 0.00136F, 0.00226F}}}};
    xt::xarray<float> expected_silu_grad = expected_silu_grad_.reshape({N, C, H, W});

    auto target = autograd::create_tensor(core::zeros_like(computed_silu->get_value()));
    auto result = mse_loss(computed_silu, target);
    result->backward();
    auto computed_silu_grad = core::to_xtensor(computed_silu->get_grad());
    EXPECT_TRUE(xt::allclose(computed_silu_grad, expected_silu_grad, 8e-3F, 4e-2F));
}

}  // namespace ttml::ops::tests
