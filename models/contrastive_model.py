from torch import nn
import torch
from .generator import get_decoder as decoder

class ContrastiveModel(nn.Module):
    def __init__(self, init_base_model, repr_size):
        super(ContrastiveModel, self).__init__()
        self.base_model = init_base_model(num_classes=repr_size)
        self.base_model.fc = self.build_mlp(self.base_model.fc, repr_size)
        self.eval_mode = False

    def get_layers(self, endpoint='C', num_classes=60):
        layers = []
        in_features = self.base_model.fc[0].in_features
        if endpoint != 'A':
            layers.append(self.base_model.fc[0])
            in_features = self.base_model.fc[0].out_features
            if endpoint != 'B':
                layers.append(self.base_model.fc[1])  # ReLU
                layers.append(self.base_model.fc[2])
                in_features = self.base_model.fc[2].out_features

        # add a final classifying layer
        layers.append(nn.Linear(in_features, num_classes))
        return layers

    def eval_finetune(self, finetune=False, endpoint='C', num_classes=60):
        self.eval_mode=True
        self.finetune = finetune

        # copy layers from MLP depending on endpoint
        layers = self.get_layers(endpoint, num_classes)

        if not finetune:
            # if linear probe - freeze base_model layers
            for param in self.base_model.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(*layers)
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.base_model.fc = None

    def build_mlp(self, fc, repr_size):
        in_features = fc.in_features
        hidden = nn.Linear(in_features=in_features, out_features=repr_size)
        output_layer = nn.Linear(in_features=repr_size, out_features=repr_size)

        return nn.Sequential(hidden, nn.ReLU(), output_layer)

    def forward(self, x):
        x = self.base_model(x)
        if self.eval_mode == True:
            x = self.classifier(x)
        return x

class ContrastiveMultiTaskModel(nn.Module):
    def __init__(self, init_base_model, repr_size, action_classes_size=60):
        super(ContrastiveMultiTaskModel, self).__init__()
        self.base_model = init_base_model(num_classes=action_classes_size)
        self.classifier_head = nn.Linear(in_features=self.base_model.fc.in_features, out_features=action_classes_size)
        self.contrastive_head = self.build_mlp(self.base_model.fc.in_features, repr_size)
        self.base_model.fc=None

    def build_mlp(self, encoder_feature_size, repr_size):
        hidden = nn.Linear(in_features=encoder_feature_size, out_features=repr_size)
        output_layer = nn.Linear(in_features=repr_size, out_features=repr_size)
        return nn.Sequential(hidden, nn.ReLU(), output_layer)

    def forward(self, x, mode="contrastive"):
        clips = x.shape[1]
        x = self.base_model(x)
        x = (torch.sum(x, dim=1) / clips)
        if mode == "contrastive":
            x = self.contrastive_head(x)
        else:
            x = self.classifier_head(x)

        return x


class ContrastiveDecoderModel(nn.Module):
    def __init__(self, init_base_model, repr_size, action_classes_size=60):
        super(ContrastiveDecoderModel, self).__init__()
        self.encoder = init_base_model(num_classes=action_classes_size)
        self.contrastive_head = self._build_mlp(self.encoder.fc.in_features, repr_size)
        self.decoder = decoder(args={'feature_size': repr_size})
        self.encoder.fc=None
        self.devices=[0]

    def _build_mlp(self, encoder_feature_size, repr_size):
        hidden = nn.Linear(in_features=encoder_feature_size, out_features=repr_size)
        output_layer = nn.Linear(in_features=repr_size, out_features=repr_size)
        return nn.Sequential(hidden, nn.ReLU(), output_layer)

    def distribute_gpus(self, devices):
        if len(devices) > 1:
            self.encoder.cuda(devices[0])
            self.contrastive_head.cuda(devices[0])
            self.decoder(devices[1])
        else:
            self.cuda(devices[0])
        self.devices=devices

    def forward(self, x, eval_mode=False, detach=False):
        x = self.encoder(x)
        contrastive_repr = self.contrastive_head(x)
        if len(self.devices) == 1:
            if detach:
                compressed_repr, generated_output = self.decoder(x.detach())
            else:
                compressed_repr, generated_output = self.decoder(x)
        else:
            compressed_repr, generated_output = self.decoder(x.cuda(self.devices[1]))
            # switch back to main gpu device
            compressed_repr = compressed_repr.cuda(self.devices[0])
            generated_output = generated_output.cuda(self.devices[0])


        return compressed_repr, generated_output, contrastive_repr