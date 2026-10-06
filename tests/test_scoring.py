import pytest

from prep.build_scenarios import build_one, check
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


def test_utility_scales_within_feasible_products_and_flips_price():
    products = [laptop("cheap", 400, 8, 3), laptop("mid", 700, 16, 3), laptop("big", 1000, 32, 3),
                laptop("over", 2000, 64, 3)]
    u = utilities(products, "laptops", RULE)
    assert u["over"] is None
    # ram scaled over 8..32 (not 64), price over 400..1000 (not 2000)
    assert u["cheap"] == pytest.approx(0.5 * 0 + 0.5 * 1)
    assert u["mid"] == pytest.approx(0.5 * (8 / 24) + 0.5 * 0.5)
    assert u["big"] == pytest.approx(0.5 * 1 + 0.5 * 0)


def test_booleans_score_one_when_true():
    rule = {"constraints": [], "weights": {"thermal_carafe": 1.0}}
    def cm(pid, thermal):
        return {"id": pid, "price": 50, "rating": 4.0, "attributes": {"thermal_carafe": thermal}}
    assert utilities([cm("a", True), cm("b", False)], "coffee_makers", rule) == {"a": 1.0, "b": 0.0}


def test_equal_values_do_not_divide_by_zero():
    products = [laptop("a", 500, 16, 3), laptop("b", 500, 16, 3)]
    u = utilities(products, "laptops", RULE)
    assert u["a"] == u["b"] == pytest.approx(1.0)


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


def test_check_requires_gap_and_feasible_counts():
    def r(u):
        return {"id": "x", "utility": u, "feasible": u is not None, "rank": 0}
    assert check([r(0.9), r(0.8), r(0.5), r(0.4), r(None)])
    assert not check([r(0.9), r(0.88), r(0.5), r(0.4), r(None)])  # gap too small
    assert not check([r(0.9), r(0.8), r(0.5), r(0.4)])  # nothing infeasible
    assert not check([r(0.9), r(0.8), r(0.5), r(None), r(None)])  # only 3 feasible


def test_build_one_is_deterministic_and_passes_checks():
    pool = [laptop(f"p{i}", 300 + 60 * i, [4, 8, 16, 32][i % 4], 2 + 0.2 * i) for i in range(20)]
    brief = {"id": "T1", "category": "laptops", "brief": "", **RULE}
    a, b = build_one(brief, pool), build_one(brief, pool)
    assert a["products"] == b["products"]
    assert len(a["products"]) == 8
    assert check(a["ranking"])
