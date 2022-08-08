import torch
from torch import nn

from cellshape_cloud.helpers.helper_modules import Flatten
from vendor.encoders import FoldNetEncoder, DGCNNEncoder
from cellshape_cloud.vendor.decoders import FoldNetDecoder, FoldingNetBasicDecoder


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std*eps


class CloudVAE(nn.Module):
    def __init__(self, num_features, k=20, encoder_type="dgcnn", decoder_type="foldingnet"):
        super(CloudVAE, self).__init__()
        self.k = k
        self.num_features = num_features

        assert encoder_type.lower() in [
            "foldingnet",
            "dgcnn",
        ], "Please select an encoder type from either foldingnet or dgcnn."

        assert decoder_type.lower() in [
            "foldingnet",
            "foldingnetbasic",
        ], "Please select an decoder type from either foldingnet."

        self.encoder_type = encoder_type.lower()
        self.decoder_type = decoder_type.lower()

        if self.encoder_type == "dgcnn":
            self.encoder = DGCNNEncoder(
                num_features=self.num_features, k=self.k
            )
        else:
            self.encoder = FoldNetEncoder(
                num_features=self.num_features, k=self.k
            )

        if self.decoder_type == "foldingnet":
            self.decoder = FoldNetDecoder(num_features=self.num_features)
        else:
            self.decoder = FoldingNetBasicDecoder(
                num_features=self.num_features
            )
        self.lin_features_len = 512
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            self.flatten = Flatten()
            self.fc_mu = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )
            self.fc_var = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )

    def forward(self, x):

        mu, log_var, feats = self._encode(x)
        z = reparametrize(mu, log_var)
        output = self._decode(z)
        return output, mu, log_var, z, feats

    def _encode(self, x):
        batch_size = x.size(0)
        feats = self.encoder(x)
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            x = self.flatten(feats)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
        else:
            mu = torch.reshape(torch.squeeze(feats), (batch_size, 512))
            log_var = torch.reshape(torch.squeeze(feats), (batch_size, 512))

        return mu, log_var, feats

    def _decode(self, z):
        return self.decoder(z)


if __name__ == "__main__":
    model = CloudVAE(num_features=128)
    inp = torch.rand((2, 2048, 3))
    out = model(inp)
    print(out[0].shape)
