import pytest

from prep.build_scenarios import build_one, check, priority_best, robust_best, weight_variants
from prep.scoring import is_feasible, rank, utilities, validate_rule


def laptop(pid, price, ram, weight, rating=4.0):
    return {"id": pid, "name": pid, "price": price, "rating": rating, "rating_count": 100,
            "attributes": {"ram_gb": ram, "storage_gb": 256, "screen_in": 14.0, "weight_lb": weight}}


RULE = {
    "constraints": [{"attr": "price", "op": "<=", "value": 1000}, {"attr": "ram_gb", "op": ">=", "value": 8}],
    "weights": {"ram_gb": 0.5, "price": 0.5},
}


def test_feasibility_checks_every_constraint():
    assert is_feasible(laptop("a", 900, 16, 3), RULE["constraints"])
    assert not is_feasible(laptop("b", 1100, 16, 3), RULE["constraints"])
    assert not is_feasible(laptop("c", 500, 4, 3), RULE["constraints"])


PRODUCTS = [laptop("cheap", 400, 8, 3, 4.0), laptop("mid", 700, 16, 3, 4.2), laptop("big", 1000, 32, 3, 4.4),
            laptop("over", 2000, 64, 3, 5.0)]


def with_weights(weights):
    return {"constraints": RULE["constraints"], "weights": weights}


def test_price_is_scored_against_the_budget():
    u = utilities(PRODUCTS, "laptops", with_weights({"price": 1.0}))
    assert u["over"] is None  # over the $1000 budget
    assert [u["cheap"], u["mid"], u["big"]] == pytest.approx([0.6, 0.3, 0.0])  # 1 - price / 1000


def test_ram_uses_fixed_log_scale():
    u = utilities(PRODUCTS, "laptops", with_weights({"ram_gb": 1.0}))
    assert [u["cheap"], u["mid"], u["big"]] == pytest.approx([1 / 3, 2 / 3, 1.0])  # 8, 16, 32 GB on log2 4-32


def test_rating_uses_fixed_range():
    u = utilities(PRODUCTS, "laptops", with_weights({"rating": 1.0}))
    assert [u["cheap"], u["mid"], u["big"]] == pytest.approx([0.5, 0.6, 0.7])  # 4.0, 4.2, 4.4 on 3-5 stars


def test_values_outside_a_scale_are_clamped():
    rule = {"constraints": [], "weights": {"ram_gb": 1.0}}
    u = utilities([laptop("tiny", 500, 2, 3), laptop("huge", 500, 128, 3)], "laptops", rule)
    assert u == {"tiny": 0.0, "huge": 1.0}


def test_score_does_not_depend_on_other_products():
    rule = with_weights({"ram_gb": 0.5, "price": 0.5})
    alone = utilities(PRODUCTS[:1], "laptops", rule)["cheap"]
    assert utilities(PRODUCTS, "laptops", rule)["cheap"] == pytest.approx(alone)


def test_booleans_score_one_when_true():
    rule = {"constraints": [], "weights": {"thermal_carafe": 1.0}}
    def cm(pid, thermal):
        return {"id": pid, "price": 50, "rating": 4.0, "attributes": {"thermal_carafe": thermal}}
    assert utilities([cm("a", True), cm("b", False)], "coffee_makers", rule) == {"a": 1.0, "b": 0.0}


def test_rank_puts_infeasible_last():
    products = [laptop("over", 2000, 64, 3), laptop("cheap", 400, 8, 3), laptop("big", 1000, 32, 3)]
    ranking = rank(products, "laptops", RULE)
    assert [r["id"] for r in ranking][-1] == "over"
    assert [r["rank"] for r in ranking] == [1, 2, 3]


def test_validate_rule_rejects_bad_weights():
    with pytest.raises(ValueError):
        validate_rule("laptops", {"constraints": [], "weights": {"ram_gb": 0.6, "price": 0.6}})
    with pytest.raises(ValueError):
        validate_rule("headphones", {"constraints": [], "weights": {"form_factor": 1.0}})
    with pytest.raises(ValueError):  # price weighted but no budget
        validate_rule("laptops", {"constraints": [], "weights": {"price": 1.0}})


def test_check_requires_gap_and_feasible_counts():
    def r(u):
        return {"id": "x", "utility": u, "feasible": u is not None, "rank": 0}
    assert check([r(0.9), r(0.8), r(0.5), r(0.4), r(None)])
    assert not check([r(0.9), r(0.88), r(0.5), r(0.4), r(None)])  # gap too small
    assert not check([r(0.9), r(0.8), r(0.5), r(0.4)])  # nothing infeasible
    assert not check([r(0.9), r(0.8), r(0.5), r(None), r(None)])  # only 3 feasible


def test_weight_variants_shift_each_weight_and_sum_to_one():
    variants = list(weight_variants({"ram_gb": 0.5, "price": 0.3, "rating": 0.2}))
    assert len(variants) == 6
    assert all(sum(v.values()) == pytest.approx(1.0) for v in variants)
    assert all(min(v.values()) > 0 for v in variants)


def test_robust_best_rejects_knife_edge_winners():
    brief = {"category": "laptops", **with_weights({"ram_gb": 0.5, "price": 0.5})}
    # "a" wins only because ram and price are weighted exactly equally
    close = [laptop("a", 650, 16, 3), laptop("b", 1000, 32, 3), laptop("c", 400, 8, 3)]
    assert not robust_best(close, brief, "a")
    clear = [laptop("a", 400, 32, 3), laptop("b", 900, 8, 3), laptop("c", 1000, 16, 3)]
    assert robust_best(clear, brief, "a")


def test_priority_reading_can_disagree_with_weighted_sum():
    # S13-like: "low price matters most, then ratings", budget $400
    brief = {"category": "laptops", "constraints": [{"attr": "price", "op": "<=", "value": 400}],
             "weights": {"price": 0.5, "rating": 0.3, "screen_in": 0.2}}
    cheap, rated = laptop("cheap", 199.99, 4, 3, 4.2), laptop("rated", 229, 4, 3, 4.7)
    assert rank([cheap, rated], "laptops", brief)[0]["id"] == "rated"  # weighted sum
    assert priority_best([cheap, rated], brief) == "cheap"  # $29 is more than a tie on price
    close = laptop("close", 205, 4, 3, 4.7)  # $5 apart counts as a tie, so ratings decide
    assert priority_best([cheap, close], brief) == "close"


def test_build_one_is_deterministic_and_passes_checks():
    pool = [laptop(f"p{i}", 300 + 60 * i, [4, 8, 16, 32][i % 4], 2 + 0.2 * i) for i in range(20)]
    brief = {"id": "T1", "category": "laptops", "brief": "", **RULE}
    a, b = build_one(brief, pool), build_one(brief, pool)
    assert a["products"] == b["products"]
    assert len(a["products"]) == 8
    assert check(a["ranking"])
