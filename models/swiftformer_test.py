import unittest
import torch
import torch.nn as nn
from swiftformer import SwiftFormer  # Import SwiftFormer from your module
from swiftformer import Embedding
from swiftformer import stem
from swiftformer import ConvEncoder
from swiftformer import Mlp
from swiftformer import EfficientAdditiveAttnetion
from swiftformer import SwiftFormerLocalRepresentation
from swiftformer import SwiftFormerEncoder
from swiftformer import Stage
from swiftformer import SwiftFormer_depth
from swiftformer import SwiftFormer_width
from swiftformer import SwiftFormer

class TestSwiftFormer(unittest.TestCase):
    def setUp(self):
        # Define the configuration for the SwiftFormer model
        self.model_config = {
            'layers': [3, 3, 6, 4],
            'embed_dims': [48, 96, 192, 384],
            'downsamples': [True, True, True, True],
            'num_classes': 1000,
            'down_patch_size': 3,
            'down_stride': 2,
            'down_pad': 1,
        }
    
    def test_conv2D(self):
        print("\n***test_conv2D")
        model = nn.Conv2d(in_channels=3, out_channels= 12, kernel_size=3, stride=4, padding=0, groups=3)
        input_data = torch.randn(64, 3, 32, 32)
        out = model(input_data)
        print(out.shape)

    def test_stem_layer(self):
        print("\n***test_stem_layer")
        in_chs = 3
        out_chs = 48  # 用于测试的输出通道数
        input_data = torch.randn(128, in_chs, 224, 224)  # 随机输入数据

        # 创建stem层
        stem_layer = stem(in_chs, out_chs)

        # 将输入数据传递给stem层
        output = stem_layer(input_data)

        print("output.shape", output.shape)

        # 检查输出的形状是否与预期相符
        expected_output_shape = (128, out_chs, 56, 56)  # 期望的输出形状
        self.assertEqual(output.shape, expected_output_shape)
    
    def test_Embedding(self):
        print("\n***test_Embedding")
        # 创建一个 Embedding 实例
        embedding = Embedding(patch_size=16, stride=16, padding=0,
                              in_chans=3, embed_dim=768)

        # 创建一个输入张量
        batch_size = 2
        in_chans = 3
        height = 32
        width = 32
        input_tensor = torch.randn(batch_size, in_chans, height, width)

        # 调用 forward 方法
        output = embedding(input_tensor)

        print(output.shape)

        # 检查输出张量的形状
        expected_shape = (batch_size, 768, 2, 2)  # 根据 patch_size 和 stride 计算
        self.assertEqual(output.shape, expected_shape)

    def test_ConvEncoder(self):
        print("\n***test_ConvEncoder")
        convencoder = ConvEncoder(dim=64, hidden_dim=128)
        # 创建一个输入张量，形状为 [B, C, H, W]
        input_tensor = torch.randn(1, 64, 32, 32)

        # 检查前向传播是否正常运行
        output = convencoder(input_tensor)
        print(output.shape)
        self.assertEqual(output.shape, input_tensor.shape)

    def test_Mlp(self):
        print("\n***test_Mlp")
        in_features = 64
        hidden_features = 128
        out_features = 64
        model = Mlp(in_features, hidden_features, out_features)
        input_data = torch.randn(1, in_features, 10, 10)
        output = model(input_data)
        print(output.shape)
        self.assertEqual(output.shape, (1, out_features, 10, 10))

    def test_EfficientAdditiveAttnetion(self):
        print("\n***test_EfficientAdditiveAttnetion")
        batch_size = 2
        seq_length = 4
        input_dim = 512
        model = EfficientAdditiveAttnetion(input_dim, token_dim=128)
        input_data = torch.randn(batch_size, seq_length, input_dim)
        output = model(input_data)
        print(output.shape)
        self.assertEqual(output.shape, (batch_size, seq_length, 128))
    
    def test_SwiftFormerLocalRepresentation(self):
        print("\n***test_SwiftFormerLocalRepresentation")
        batch_size = 2
        input_channels = 3
        height = 32
        width  = 32
        model = SwiftFormerLocalRepresentation(dim = input_channels)
        input_data = torch.randn(batch_size, input_channels, height, width)
        out = model(input_data)
        print(out.shape)
        self.assertEqual(out.shape, (batch_size, input_channels, height, width))
    
    def test_SwiftFormerEncoder(self):
        print("\n***test_SwiftFormerEncoder")
        batch_size = 2
        input_channels = 3
        height = 32
        width  = 32
        model  = SwiftFormerEncoder(dim = input_channels)
        input_data = torch.randn(batch_size, input_channels, height, width)
        out = model(input_data)
        print(out.shape)
        self.assertEqual(out.shape, (batch_size, input_channels, height, width))
    
    def test_stage(self):
        print("\n***test_stage")
        batch_size = 2
        input_channels = 3
        height = 32
        width  = 32
        model = Stage(dim=input_channels, index=0, layers=SwiftFormer_depth['XS'])
        input_data = torch.randn(batch_size, input_channels, height, width)
        out = model(input_data)
        print(out.shape)
        self.assertEqual(out.shape, (batch_size, input_channels, height, width))

    def test_SwiftFormer(self):
        print("\n***test_SwiftFormer")
        batch_size = 2
        input_channels = 3
        height = 32
        width  = 32
        model = SwiftFormer(layers=SwiftFormer_depth['XS'],
                            embed_dims=SwiftFormer_width['XS'], 
                            downsamples= [True, True, True, True],
                            vit_num=1, distillation=False)
        input_data = torch.randn(batch_size, input_channels, height, width)
        out = model(input_data)
        print(len(out))
        print(out.shape)
        #print(out[0].shape, out[1].shape)
        #self.assertEqual(out.shape, (batch_size, input_channels, height, width))



    



    

    # def test_forward_pass(self):
    #     # Create an instance of the SwiftFormer model
    #     model = SwiftFormer(**self.model_config)

    #     # Generate a random input tensor that matches the expected input size
    #     input_tensor = torch.randn(1, 3, 224, 224)

    #     # Perform a forward pass through the model
    #     output = model(input_tensor)

    #     # Ensure that the output tensor has the correct shape
    #     expected_shape = (1, self.model_config['num_classes'])
    #     self.assertEqual(output.shape, expected_shape)

    # def test_feature_extraction(self):
    #     # Create an instance of the SwiftFormer model with feature extraction mode
    #     model = SwiftFormer(fork_feat=True, **self.model_config)

    #     # Generate a random input tensor that matches the expected input size
    #     input_tensor = torch.randn(1, 3, 224, 224)

    #     # Perform a forward pass through the model for feature extraction
    #     features = model.forward_tokens(input_tensor)

    #     # Ensure that the features have the correct shape (4 feature maps)
    #     self.assertEqual(len(features), 4)
    #     for i, feature_map in enumerate(features):
    #         expected_shape = (1, self.model_config['embed_dims'][i], 14, 14)
    #         self.assertEqual(feature_map.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
