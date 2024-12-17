import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, GCNConv, TransformerConv, global_mean_pool

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.context_vector = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        attention_scores = torch.tanh(self.fc(x))
        attention_weights = attention_scores * self.context_vector
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attended_features = x * attention_weights
        return attended_features, attention_weights

# Multi-Head Attention Layer
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            AttentionLayer(input_dim, output_dim) for _ in range(num_heads)
        ])
        self.fc = nn.Linear(output_dim * num_heads, output_dim)

    def forward(self, x):
        attention_outputs = []
        for attention in self.attention_heads:
            attended_x, _ = attention(x)
            attention_outputs.append(attended_x)
        concat_attended = torch.cat(attention_outputs, dim=-1)
        output = self.fc(concat_attended)
        return output

# Base GNN Class
class BaseGNN(nn.Module):
    def __init__(self, num_layers, emb_dim, input_dim, conv_type='GIN', drop_ratio=0.6):
        super(BaseGNN, self).__init__()
        self.atom_encoder = nn.Linear(input_dim, emb_dim)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_channels = emb_dim
        for _ in range(num_layers):
            if conv_type == 'GIN':
                nn_seq = nn.Sequential(
                    nn.Linear(in_channels, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim)
                )
                conv = GINConv(nn_seq)
                out_channels = emb_dim
            elif conv_type == 'GAT':
                conv = GATConv(in_channels, emb_dim // 4, heads=4, dropout=0.6, concat=True)
                out_channels = (emb_dim // 4) * 4
            elif conv_type == 'GCN':
                conv = GCNConv(in_channels, emb_dim)
                out_channels = emb_dim
            elif conv_type == 'Transformer':
                conv = TransformerConv(in_channels, emb_dim, heads=4, concat=False, dropout=0.5)
                out_channels = emb_dim
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")

            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        self.fc1 = nn.Linear(out_channels * num_layers, emb_dim // 2)
        self.fc2 = nn.Linear(emb_dim // 2, 1)
        self.drop_ratio = drop_ratio

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.atom_encoder(x)

        layer_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            pooled_x = global_mean_pool(x, batch)
            layer_outputs.append(pooled_x)

        x = torch.cat(layer_outputs, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Enhanced GNN Model with Variants
class EnhancedGNN(BaseGNN):
    def __init__(self, num_layers, emb_dim, input_dim, add_feat_dim, conv_type='GIN',
                 drop_ratio=0.6, attention_type='separate'):
        super(EnhancedGNN, self).__init__(num_layers, emb_dim, input_dim, conv_type, drop_ratio)
        self.fc2_input_dim = (emb_dim // 2) + add_feat_dim
        self.fc2 = nn.Linear(self.fc2_input_dim, 1)
        self.attention_type = attention_type

        if attention_type == 'separate':
            self.graph_attention = AttentionLayer(emb_dim // 2, emb_dim // 2)
            self.add_feat_attention = AttentionLayer(add_feat_dim, add_feat_dim)
        elif attention_type == 'unified':
            total_feature_dim = (emb_dim // 2) + add_feat_dim
            self.unified_attention = AttentionLayer(total_feature_dim, total_feature_dim)
        elif attention_type == 'multihead':
            num_heads = 4
            self.graph_attention = MultiHeadAttentionLayer(emb_dim // 2, emb_dim // 2, num_heads)
            self.add_feat_attention = MultiHeadAttentionLayer(add_feat_dim, add_feat_dim, num_heads)
        elif attention_type == 'none':
            pass  # No attention layers
        else:
            raise ValueError("Unsupported attention_type")

    def forward(self, data, return_attention_weights=False):
        x, edge_index, batch, add_feat = data.x, data.edge_index, data.batch, data.additional_feature
        x = self.atom_encoder(x)

        layer_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            pooled_x = global_mean_pool(x, batch)
            layer_outputs.append(pooled_x)

        x = torch.cat(layer_outputs, dim=1)
        x = F.relu(self.fc1(x))

        attention_weights = {}
        if self.attention_type == 'separate':
            x, weight_graph = self.graph_attention(x)
            attention_weights['graph'] = weight_graph
            if add_feat is not None:
                add_feat = add_feat.view(x.size(0), -1)
                add_feat, weight_desc = self.add_feat_attention(add_feat)
                attention_weights['descriptor'] = weight_desc
                x = torch.cat([x, add_feat], dim=1)
        elif self.attention_type == 'unified':
            if add_feat is not None:
                add_feat = add_feat.view(x.size(0), -1)
                x = torch.cat([x, add_feat], dim=1)
                x, weight_unified = self.unified_attention(x)
                attention_weights['unified'] = weight_unified
        elif self.attention_type == 'multihead':
            x = self.graph_attention(x)
            if add_feat is not None:
                add_feat = add_feat.view(x.size(0), -1)
                add_feat = self.add_feat_attention(add_feat)
                attention_weights['descriptor'] = add_feat  # Assuming MultiHeadAttentionLayer returns only features
                x = torch.cat([x, add_feat], dim=1)
        elif self.attention_type == 'none':
            if add_feat is not None:
                add_feat = add_feat.view(x.size(0), -1)
                x = torch.cat([x, add_feat], dim=1)

        x = self.fc2(x)

        if return_attention_weights:
            return x, attention_weights
        else:
            return x
