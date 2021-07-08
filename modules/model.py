from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from efficientnet_pytorch import EfficientNet
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
# from genet import GENet


class DeitForMetricLearning(nn.Module):
    def __init__(self, model, embedding_size, dropout=None, batchnorm=True):
        super().__init__()
        self.model = model
        self.embedding_size = embedding_size
        self.batchnorm = batchnorm
        num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(num_ftrs, embedding_size)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if self.batchnorm:
            self.embedding_bn = nn.BatchNorm1d(embedding_size)

    def forward(self, inputs):
        x = self.model.forward_features(inputs)
        if self.dropout:
            x = self.dropout(x)
        x = self.model.head(x)
        if self.batchnorm:
            x = self.embedding_bn(x)
        return x


class DeitForMetricLearning2head(nn.Module):
    def __init__(self, model, embedding_size, n_categories, cls_from_backbone, dropout=None, batchnorm=True):
        super().__init__()
        self.n_categories = n_categories
        self.model = model
        self.embedding_size = embedding_size
        self.batchnorm = batchnorm
        self.cls_from_backbone = cls_from_backbone
        num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(num_ftrs, embedding_size)
        if self.cls_from_backbone:
            self.cat_head = nn.Linear(num_ftrs, self.n_categories)
        else:
            self.cat_head = nn.Linear(embedding_size, self.n_categories)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if self.batchnorm:
            self.embedding_bn = nn.BatchNorm1d(embedding_size)

    def forward(self, inputs):
        x = self.model.forward_features(inputs)
        if self.dropout:
            x = self.dropout(x)
        if self.cls_from_backbone:
            y = self.cat_head(x)
        x = self.model.head(x)
        if self.batchnorm:
            x = self.embedding_bn(x)
        if not self.cls_from_backbone:
            y = self.cat_head(F.normalize(x))
        return x, y


class ResNextForMetricLearning(nn.Module):
    def __init__(self, model_name, backbone, embedding_size, pooling_type, dropout_p=None, batchnorm=True):
        super().__init__()
        self.dropout = None
        self.batchnorm = batchnorm

        last_channels_map = {
            512: ['t18', 't34'], 2048: ['t50', 't101', 't152']
        }
        for lc, model_sizes in last_channels_map.items():
            for model_size in model_sizes:
                if model_size in model_name:
                    last_channels = lc
                    break

        layers = [
            ('conv1', backbone.conv1),
            ('bn1', backbone.bn1),
            ('relu', backbone.relu),
            ('maxpool', backbone.maxpool),
            ('layer1', backbone.layer1),
            ('layer2', backbone.layer2),
            ('layer3', backbone.layer3),
            ('layer4', backbone.layer4)
        ]

        if pooling_type == 'gem':
            layers.append(('gem', GeM()))
            self.fc = nn.Linear(last_channels, embedding_size)
        elif pooling_type == 'avg_max':
            layers.append(('avg_max_pool', AdaptiveConcatPool2d((1, 1))))
            self.fc = nn.Linear(2 * last_channels, embedding_size)
        elif pooling_type == 'avg':
            layers.append(('avgpool', backbone.avgpool))
            self.fc = nn.Linear(last_channels, embedding_size)
        else:
            self.fc = nn.Linear(7 * 7 * last_channels, embedding_size)

        self.features = nn.Sequential(OrderedDict(layers))

        if dropout_p:
            self.dropout = nn.Dropout(dropout_p)
        if self.batchnorm:
            self.embedding_bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc(x)
        if self.batchnorm:
            x = self.embedding_bn(x)
        return x


class EfficientNetForMetricLearning(nn.Module):
    def __init__(self, model, pooling_type, embedding_size, batchnorm=True):
        super().__init__()
        self.model = model
        self.embedding_size = embedding_size
        self.batchnorm = batchnorm

        if pooling_type == 'gem':
            self.pooling = GeM()
        elif pooling_type == 'avg_max':
            self.pooling = AdaptiveConcatPool2d((1, 1))
        elif pooling_type == 'avg':
            self.pooling = model._avg_pooling

        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, embedding_size)
        if self.batchnorm:
            self.embedding_bn = nn.BatchNorm1d(embedding_size)

    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.model._fc(x)
        if self.batchnorm:
            x = self.embedding_bn(x)
        return x


class GENetForClassification(nn.Module):
    def __init__(self, genet_backbone):
        super().__init__()
        self.genet_backbone = genet_backbone
        in_features = self.genet_backbone.fc_linear.in_features
        self.genet_backbone.fc_linear = nn.Identity()

    def forward(self, x):
        x = self.genet_backbone(x)
        return x


class GeM(nn.Module):
    """
    https://amaarora.github.io/2020/08/30/gempool.html
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def load_model(model_name, embedding_size, pooling_type, img_size=224, batchnorm=True, n_categories=None,
               cls_from_backbone=False):
    if 'deit' in model_name:
        deit_size = model_name.split('-')[1]
        if img_size == 224:
            deit_model = torch.hub.load('facebookresearch/deit:main',
                                        f'deit_{deit_size}_patch16_{img_size}', pretrained=True)
        elif img_size == 384:
            deit_model = VisionTransformer(
                img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
            deit_model.default_cfg = _cfg()
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
                map_location="cpu", check_hash=True
            )
            deit_model.load_state_dict(checkpoint["model"])
        else:
            raise NotImplementedError
        if n_categories is None:
            deit_model = DeitForMetricLearning(deit_model, embedding_size, dropout=0.2, batchnorm=batchnorm)
        else:
            deit_model = DeitForMetricLearning2head(deit_model, embedding_size, n_categories, cls_from_backbone,
                                                    dropout=0.2, batchnorm=batchnorm)

        layers_to_freeze = {
            'deit-tiny': ['layer0', 'blocks.0', 'blocks.1'],
            'deit-small': ['layer0', 'blocks.0', 'blocks.1', 'blocks.2', 'blocks.3'],
            'deit-base': ['layer0', 'blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5']
        }

        for name, param in deit_model.named_parameters():
            # print(name)
            for layer_name in layers_to_freeze[model_name]:
                if layer_name == 'layer0':
                    if ('cls_token' in name) or ('pos_embed' in name) or ('patch_embed.proj.weight' in name) or (
                            'patch_embed.proj.bias' in name
                    ):
                        param.requires_grad = False
                if layer_name in name:
                    param.requires_grad = False

        return deit_model

    if 'efficientnet' in model_name:
        efficient_net_model = EfficientNet.from_pretrained(model_name, include_top=False)
        efficient_net_model = EfficientNetForMetricLearning(efficient_net_model, pooling_type, embedding_size,
                                                            batchnorm)

        layers_to_freeze = {
            'efficientnet-b0': ['layer0', '_blocks.0', '_blocks.1'],
            'efficientnet-b2': ['layer0', '_blocks.0', '_blocks.1', '_blocks.2', '_blocks.3', '_blocks.4', '_blocks.5',
                                '_blocks.6'],
            'efficientnet-b3': ['layer0', '_blocks.0', '_blocks.1', '_blocks.2', '_blocks.3', '_blocks.4',
                                '_blocks.5', '_blocks.6', '_blocks.7', '_blocks.8']
        }

        for name, param in efficient_net_model.named_parameters():
            # print(name)
            for layer_name in layers_to_freeze[model_name]:
                if layer_name == 'layer0':
                    if (name == 'efficient_net._conv_stem.weight') or (name == 'efficient_net._bn0.weight') or (
                            name == 'efficient_net._bn0.bias'):
                        param.requires_grad = False
                if layer_name in name:
                    param.requires_grad = False

        return efficient_net_model

    elif 'resnet50' in model_name:
        resnet_model = models.resnet50(pretrained=True)
        resnet_model = ResNextForMetricLearning(model_name, resnet_model, embedding_size, pooling_type,
                                                batchnorm=batchnorm)
        return resnet_model

    elif 'wsl' in model_name:
        resnext_model = torch.hub.load('facebookresearch/WSL-Images', model_name)
        resnext_model = ResNextForMetricLearning(model_name, resnext_model, embedding_size, pooling_type,
                                                 batchnorm=batchnorm)
        return resnext_model

    elif 'resnext' in model_name:
        resnext_model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
        resnext_model = ResNextForMetricLearning(model_name, resnext_model, embedding_size, pooling_type,
                                                 batchnorm=batchnorm)
        return resnext_model


def load_gnet_model(parameters_dir):
    genet_model = GENet.genet_normal(pretrained=True, root=parameters_dir)
    genet_model = GENetForClassification(genet_model)
    return genet_model


def load_style_models(mode):
    pretrained_models = []
    if "style" in mode:
        pretrained_models.append(load_style_model())
    if "content" in mode:
        pretrained_models.append(load_resnext_model())
    return [load_style_model(),]


def load_resnext_model():
    resnext_model = torch.hub.load('pytorch/vision:v0.6.0', "resnext50_32x4d", pretrained=True)
    resnext_model.fc = nn.Identity()
    return resnext_model


def load_style_model():
    vgg = models.vgg19(pretrained=True)
    extractor = StyleModel(vgg)
    return extractor


class StyleModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.vgg = backbone.features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def forward(self, image):
        features = get_features(image, self.vgg)
        return gram_matrix(features)


def get_features(image, model, layers=None):
    if layers is None:
        layers = {'10': 'conv3_1'}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            return x


def gram_matrix(tensor):
    _, d, h, w = tensor.size()

    tensor = tensor.view(d, h * w)

    gram = torch.mm(tensor, tensor.t())

    return gram


# freeze_layer(resnet_model, 'layer1')
def freeze_layer(model, layer_name):
    change_layer_freeze_state(model, layer_name, unfreeze=False)


# unfreeze_layer(resnet_model, 'layer1')
def unfreeze_layer(model, layer_name):
    change_layer_freeze_state(model, layer_name, unfreeze=True)


def change_layer_freeze_state(model, layer_name, unfreeze):
    for name, param in model.named_parameters():
        if layer_name == 'layer0':
            if (name == 'conv1.weight') or (name == 'bn1.weight') or (name == 'bn1.bias'):
                param.requires_grad = unfreeze
        if layer_name in name:
            param.requires_grad = unfreeze


class DeitForBinaryClassification(nn.Module):
    def __init__(self, model, dropout=None):
        super().__init__()
        self.model = model
        num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(num_ftrs, 1)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, inputs):
        x = self.model.forward_features(inputs)
        if self.dropout:
            x = self.dropout(x)
        x = self.model.head(x)
        return x


def load_binary_cls_model(model_name, img_size=224, dropout=0.2):
    deit_size = model_name.split('-')[1]
    if img_size == 224:
        deit_model = torch.hub.load('facebookresearch/deit:main',
                                    f'deit_{deit_size}_patch16_{img_size}', pretrained=True)
    elif img_size == 384:
        deit_model = VisionTransformer(
            img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        deit_model.default_cfg = _cfg()
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        deit_model.load_state_dict(checkpoint["model"])
    else:
        raise NotImplementedError
    deit_model = DeitForBinaryClassification(deit_model, dropout=dropout)

    layers_to_freeze = {
        'deit-tiny': ['layer0', 'blocks.0', 'blocks.1'],
        'deit-small': ['layer0', 'blocks.0', 'blocks.1', 'blocks.2', 'blocks.3'],
        'deit-base': ['layer0', 'blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5']
    }

    for name, param in deit_model.named_parameters():
        # print(name)
        for layer_name in layers_to_freeze[model_name]:
            if layer_name == 'layer0':
                if ('cls_token' in name) or ('pos_embed' in name) or ('patch_embed.proj.weight' in name) or (
                        'patch_embed.proj.bias' in name
                ):
                    param.requires_grad = False
            if layer_name in name:
                param.requires_grad = False

    return deit_model
