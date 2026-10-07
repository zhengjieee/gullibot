"""Models the agent can run on, with request settings and prices.

Sampling is left at each model's default: Sonnet 5 and Opus 5.5 reject the
temperature parameter, so pinning one value across models isn't possible.
Prices are USD per million tokens (list price, 5-minute cache).
"""

MODELS = {
    "claude-haiku-4-5": {
        "thinking": None,  # Haiku 4.5 has no adaptive thinking; run it without
        "price": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
    },
    "claude-sonnet-5": {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "price": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    },
    "claude-opus-5-5": {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "price": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    },
}


def cost_usd(model, usage):
    """usage: dict with input, output, cache_write, cache_read token counts."""
    price = MODELS[model]["price"]
    return sum(usage[k] * price[k] for k in price) / 1e6
