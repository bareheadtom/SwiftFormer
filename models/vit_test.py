import torch
import unittest
from vit import ViT
from vit import *

class TestViT(unittest.TestCase):
    def setUp(self):
        # 初始化测试数据和模型
        self.batch_size = 2
        self.image_channels = 3
        self.image_height = 32
        self.image_width = 32
        self.patch_size = 8
        self.num_classes = 10
        self.dim = 128
        self.depth = 6
        self.heads = 8
        self.mlp_dim = 512
        self.dim_head = 64
        self.dropout = 0.1
        self.emb_dropout = 0.1

        self.model = ViT(
            image_size=(self.image_height, self.image_width),
            patch_size=self.patch_size,
            num_classes=self.num_classes,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            channels=self.image_channels,
            dim_head=self.dim_head,
            dropout=self.dropout,
            emb_dropout=self.emb_dropout
        )

        self.input_data = torch.randn(
            2, 3, 32, 32
        )

    def test_FeedForward(self):
        print("\n***test_FeedForward")
        batch_size = 2
        seq_length = 10
        dim = 64
        hidden_dim = 128
        model = FeedForward(dim=dim, hidden_dim=hidden_dim)
        input_data = torch.randn(batch_size,seq_length, dim)
        out = model(input_data)
        print(out.shape)

    def test_Attention(self):
        print("\n***test_Attention")
        batch_size = 2
        seq_length = 10
        dim = 64
        heads = 8
        dim_head = 48
        model = Attention(dim=dim, heads=heads,dim_head=dim_head)
        input_data = torch.randn(batch_size, seq_length, dim)
        out = model(input_data)
        print(out.shape)
    
    def test_TransformerEncode(self):
        print("\n***test_TransformerEncode")
        batch_size = 2
        seq_length = 10
        dim = 64
        depth = 3
        heads = 8
        dim_head = 48
        mlp_dim = 128
        model = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
        input_data = torch.randn(batch_size, seq_length, dim)
        out = model(input_data)
        print(out.shape)
        self.assertEqual(out.shape, (batch_size, seq_length, dim))

    def test_Vit(self):
        print("\n***test_Vit")
        batch_size = 128
        image_channels = 3
        image_height = 224
        image_width = 224
        patch_size = 8
        num_classes = 10
        dim = 64
        depth = 3
        heads = 8
        mlp_dim = 512
        dim_head = 48
        model = ViT(image_size=(image_height, image_width), 
                    patch_size=patch_size, 
                    num_classes=num_classes, 
                    dim=dim, 
                    depth=depth,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    channels=image_channels,
                    dim_head=dim_head)
        input_data = torch.randn(batch_size, image_channels, image_height, image_width)
        out = model(input_data)
        print(out.shape)
        self.assertEqual(out.shape, (batch_size, num_classes))



        


    def test_forward_pass(self):
        print("\n***test_forward_pass")
        # 测试前向传播方法
        output = self.model(self.input_data)
        expected_shape = (self.batch_size, self.num_classes)
        print(output.shape)
        self.assertEqual(output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
