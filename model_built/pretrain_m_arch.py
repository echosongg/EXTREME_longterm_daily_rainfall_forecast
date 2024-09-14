import torch
import torch.nn as nn
import RRDBNet_arch as pre_arch
import torch.nn.functional as F
import functools

class ModifiedRRDBNet(nn.Module):
    def __init__(self, pretrained_model_path, in_nc, out_nc, nf, nb, gc=32):
        super(ModifiedRRDBNet, self).__init__()
        # 初始化原始 RRDBNetx4x2 模型
        self.original_model = pre_arch.RRDBNetx4x2(1, 1, 64, 23, gc=32)
        # 加载预训练模型的权重
        
        self.original_model.load_state_dict(torch.load(pretrained_model_path).module.state_dict(), strict=True)
        
        # 由于你的模型结构不是简单的Sequential，我们需要手动修改它的结构
        # 我们将保留original_model中的所有层，但不包括最后的卷积层self.conv_last
        # 因此，不需要物理删除原始模型中的层，而是在forward方法中改变数据流，绕过最后一层
        
        # 定义新的输出层以适应新的输出通道数
        self.new_output_conv = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    
    def forward(self, x):
        # 复制original_model的forward方法，但修改以绕过最后一层
        fea = self.original_model.conv_first(x)
        trunk = self.original_model.trunk_conv(self.original_model.RRDB_trunk(fea))
        fea = fea + trunk

        # x4 upsampling path
        fea = self.lrelu(self.original_model.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.original_model.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.original_model.HRconv(fea)

        trunk = self.original_model.trunk_conv2(self.original_model.RRDB_trunk2(fea))
        fea = fea + trunk

        # x2 upsampling path
        fea = self.lrelu(self.original_model.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.original_model.upconv4(fea)
        fea = self.lrelu(fea)
        fea = self.original_model.HRconv2(fea)
        #before conv torch.Size([1, 64, 128, 128])
        # 使用新的输出层代替原始模型的最后一层
        bg_output = self.new_output_conv(self.lrelu(fea))
        #bg_output torch.Size([1, 3, 128, 128])
        return bg_output