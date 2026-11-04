"""
Cosine LR schedule and the AdamW rebuild used by every architecture's train().
"""

import optax
import jax


def cosine_schedule(n_iter, init_lr=1e-3, end_lr_frac=0.01, warmup_frac=0.0):
    """Optax LR schedule: linear warmup from 0 to init_lr, then cosine decay.

    Warmup is int(warmup_frac*n_iter) steps; LR reaches init_lr*end_lr_frac at n_iter.
    """
    warmup_steps = int(warmup_frac * n_iter)
    decay_steps = n_iter - warmup_steps

    if warmup_steps == 0:
        return optax.cosine_decay_schedule(
            init_value=init_lr,
            decay_steps=decay_steps,
            alpha=end_lr_frac,
        )

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=init_lr,
        warmup_steps=warmup_steps,
        decay_steps=n_iter,  # total steps; warmup is included
        end_value=init_lr * end_lr_frac,
    )


def rebuild_optimizer(
    model, n_iter, init_lr=1e-3, end_lr_frac=0.01, warmup_frac=0.0, weight_decay=None
):
    """Replace model.optimizer/opt_state with AdamW on cosine_schedule(n_iter, ...).

    Decay is masked to 2D+ tensors; weight_decay=None means 1e-4.
    """
    sched = cosine_schedule(
        n_iter,
        init_lr=init_lr,
        end_lr_frac=end_lr_frac,
        warmup_frac=warmup_frac,
    )
    if weight_decay is None:
        weight_decay = 1e-4  # the models' default

    def decay_mask(p):
        return jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)

    model.optimizer = optax.adamw(
        learning_rate=sched, weight_decay=weight_decay, mask=decay_mask
    )

    model.opt_state = model.optimizer.init(model.params)
