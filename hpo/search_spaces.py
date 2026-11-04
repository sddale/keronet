"""HPO search spaces, one per architecture and parameter budget.

SEARCH_SPACES[arch][budget] lists (hyperparameter, categorical choices) in suggestion order;
DEFAULT_SPACE_FOR_CAP maps --max_params to a budget name. tools/verify_search_spaces.py
checks the released study databases against these spaces.
"""

SEARCH_SPACES = {
    'keronet': {
        '100k': [
            ('L', [4, 8, 16, 32]),
            ('d_k', [8, 16, 32]),
            ('d_v', [8, 16, 32]),
            ('post_mlp_hidden', [8, 16, 32]),
            ('branch_hidden', [72, 96, 144]),
            ('trunk_hidden', [72, 96, 144]),
            ('branch_layers_n', [2, 3, 4]),
            ('trunk_layers_n', [2, 3, 4]),
            ('mapping_size', [24, 48, 96]),
        ],
        '200k': [
            ('L', [4, 8, 16, 32]),
            ('d_k', [16, 32, 64]),
            ('d_v', [16, 32, 64]),
            ('post_mlp_hidden', [16, 32, 64]),
            ('branch_hidden', [96, 128, 192]),
            ('trunk_hidden', [96, 128, 192]),
            ('branch_layers_n', [2, 3, 4]),
            ('trunk_layers_n', [2, 3, 4]),
            ('mapping_size', [32, 64, 128]),
        ],
        '300k': [
            ('L', [4, 8, 16, 32]),
            ('d_k', [20, 40, 80]),
            ('d_v', [20, 40, 80]),
            ('post_mlp_hidden', [16, 32, 64]),
            ('branch_hidden', [128, 160, 224]),
            ('trunk_hidden', [128, 160, 224]),
            ('branch_layers_n', [2, 3, 4]),
            ('trunk_layers_n', [2, 3, 4]),
            ('mapping_size', [32, 64, 128]),
        ],
        '500k': [
            ('L', [16, 32, 64, 128]),
            ('d_k', [16, 32, 64]),
            ('d_v', [16, 32, 64]),
            ('post_mlp_hidden', [16, 32, 64]),
            ('branch_hidden', [112, 144, 208]),
            ('trunk_hidden', [112, 144, 208]),
            ('branch_layers_n', [2, 3, 4]),
            ('trunk_layers_n', [2, 3, 4]),
            ('mapping_size', [32, 64, 128]),
        ],
    },
    'mkeronet': {
        '200k': [
            ('L', [4, 8, 16, 32]),
            ('d_k', [16, 32, 64]),
            ('branch_hidden', [96, 128, 192]),
            ('trunk_hidden', [96, 128, 192]),
            ('branch_layers_n', [2, 3, 4]),
            ('trunk_layers_n', [2, 3, 4]),
            ('mapping_size', [32, 64, 128]),
        ],
        '500k': [
            ('L', [16, 32, 64, 128]),
            ('d_k', [16, 32, 64]),
            ('branch_hidden', [112, 144, 208]),
            ('trunk_hidden', [112, 144, 208]),
            ('branch_layers_n', [2, 3, 4]),
            ('trunk_layers_n', [2, 3, 4]),
            ('mapping_size', [32, 64, 128]),
        ],
    },
    'pi_deeponet': {
        '100k': [
            ('p', [48, 96, 144, 192]),
            ('branch_hidden', [64, 96, 128, 192]),
            ('trunk_hidden', [64, 96, 128, 192]),
            ('branch_layers_n', [2, 3, 4, 5]),
            ('trunk_layers_n', [2, 3, 4, 5]),
            ('mapping_size', [16, 32, 64]),
        ],
        '200k': [
            ('branch_hidden', [96, 128, 192, 256]),
            ('trunk_hidden', [96, 128, 192, 256]),
            ('branch_layers_n', [2, 3, 4, 5]),
            ('trunk_layers_n', [2, 3, 4, 5]),
            ('p', [64, 128, 192, 256]),
            ('mapping_size', [32, 64, 128]),
        ],
        '300k': [
            ('p', [96, 192, 288, 384]),
            ('branch_hidden', [128, 160, 224, 320]),
            ('trunk_hidden', [128, 160, 224, 320]),
            ('branch_layers_n', [2, 3, 4, 5]),
            ('trunk_layers_n', [2, 3, 4, 5]),
            ('mapping_size', [32, 64, 128]),
        ],
        '500k': [
            ('p', [192, 384, 576, 768]),
            ('branch_hidden', [128, 192, 288, 384]),
            ('trunk_hidden', [128, 192, 288, 384]),
            ('branch_layers_n', [2, 3, 4, 5]),
            ('trunk_layers_n', [2, 3, 4, 5]),
            ('mapping_size', [32, 64, 128]),
        ],
    },
    'pit': {
        '100k': [
            ('n_blocks', [1, 2, 3, 4]),
            ('n_heads', [1, 2, 4]),
            ('d_model', [24, 48, 64, 96]),
            ('key_dim', [24, 48, 64, 96]),
            ('pe_dim', [16, 32, 64]),
        ],
        '200k': [
            ('d_model', [32, 64, 96, 128]),
            ('n_heads', [1, 2, 4]),
            ('key_dim', [32, 64, 96, 128]),
            ('n_blocks', [1, 2, 3, 4]),
            ('pe_dim', [16, 32, 64]),
        ],
        '300k': [
            ('n_blocks', [1, 2, 3, 4]),
            ('n_heads', [1, 2, 4]),
            ('d_model', [48, 80, 112, 160]),
            ('key_dim', [48, 80, 112, 160]),
            ('pe_dim', [16, 32, 64]),
        ],
        '500k': [
            ('n_blocks', [2, 3, 4, 5]),
            ('n_heads', [2, 4, 8]),
            ('d_model', [40, 72, 104, 136, 160]),
            ('key_dim', [40, 72, 104, 136, 160]),
            ('pe_dim', [16, 32, 64]),
        ],
    },
    'pinto': {
        '100k': [
            ('n_cross_attn', [1, 2, 3, 4]),
            ('n_heads', [1, 2, 4]),
            ('d_model', [24, 48, 64, 96]),
            ('key_dim', [24, 48, 64, 96]),
        ],
        '200k': [
            ('d_model', [32, 64, 96, 128]),
            ('n_heads', [1, 2, 4]),
            ('key_dim', [32, 64, 96, 128]),
            ('n_cross_attn', [1, 2, 3, 4]),
        ],
        '300k': [
            ('n_cross_attn', [1, 2, 3, 4]),
            ('n_heads', [1, 2, 4]),
            ('d_model', [48, 80, 112, 160]),
            ('key_dim', [48, 80, 112, 160]),
        ],
        '500k': [
            ('n_cross_attn', [2, 3, 4, 5]),
            ('n_heads', [2, 4, 8]),
            ('d_model', [40, 72, 104, 136, 160]),
            ('key_dim', [40, 72, 104, 136, 160]),
        ],
    },
}


DEFAULT_SPACE_FOR_CAP = {100_000: '100k', 200_000: '200k', 300_000: '300k', 500_000: '500k'}


def make_sampler(arch, space):
    """Return a sample_hparams(trial) that suggests SEARCH_SPACES[arch][space] in its listed order."""
    spec = SEARCH_SPACES[arch][space]

    def sample_hparams(trial):
        return {name: trial.suggest_categorical(name, choices) for name, choices in spec}

    return sample_hparams
