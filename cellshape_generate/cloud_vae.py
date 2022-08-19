import torch
from torch import nn

from cellshape_cloud.helpers.helper_modules import Flatten
from vendor.encoders import FoldNetEncoder, DGCNNEncoder
from vendor.decoders import FoldNetDecoder


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std * eps


class CloudVAE(nn.Module):
    def __init__(
        self, num_features=512, k=20, encoder_type="dgcnn", decoder_type="foldingnet",
            num_features_ae=512
    ):
        super(CloudVAE, self).__init__()
        self.k = k
        self.num_features = num_features
        self.num_features_ae = num_features_ae

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
            self.encoder = DGCNNEncoder(num_features=self.num_features, k=self.k)
        else:
            self.encoder = FoldNetEncoder(num_features=self.num_features, k=self.k)

        if self.decoder_type == "foldingnet":
            self.decoder = FoldNetDecoder(num_features=self.num_features)

        if (self.num_features_ae < self.num_features) or (
            self.num_features_ae > self.num_features
        ):
            self.flatten = Flatten()
            self.fc_mu = nn.Linear(self.num_features, self.num_features_ae, bias=False)
            self.fc_var = nn.Linear(
                self.num_features, self.num_features_ae, bias=False
            )
            self.deembedding = nn.Linear(self.num_features_ae, self.num_features)

    def forward(self, x):

        mu, log_var, feats = self._encode(x)
        z = reparametrize(mu, log_var)
        output = self._decode(z)
        return output, mu, log_var, z, feats

    def _encode(self, x):
        batch_size = x.size(0)
        feats = self.encoder(x)
        if (self.num_features_ae < self.num_features) or (
            self.num_features_ae > self.num_features
        ):
            x = self.flatten(feats)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
        else:
            mu = torch.reshape(torch.squeeze(feats), (batch_size, self.num_features))
            log_var = torch.reshape(torch.squeeze(feats), (batch_size, self.num_features))

        return mu, log_var, feats

    def _decode(self, z):
        if (self.num_features_ae < self.num_features) or (
            self.num_features_ae > self.num_features
        ):
            z = self.deembedding(z)

        z = z.unsqueeze(1)
        output = self.decoder(z)
        return output


if __name__ == "__main__":
    model = CloudVAE(num_features_ae=128)
    inp = torch.rand((2, 2048, 3))
    out = model(inp)
    print(out[0].shape)
