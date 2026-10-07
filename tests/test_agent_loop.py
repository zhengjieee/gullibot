from types import SimpleNamespace

from agent.loop import run_agent
from runner.session import outcome
from store.store import TOOLS, Store
from tests.test_store import PRODUCTS


def usage():
    return SimpleNamespace(input_tokens=100, output_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=0)


def tool_use(i, name, **args):
    return SimpleNamespace(type="tool_use", id=f"t{i}", name=name, input=args)


def reply(blocks, stop_reason):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=usage(), model="fake-model")


class FakeClient:
    """Plays back scripted responses; script entries are functions of the store so they can use its codes."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls += 1
        return self.script.pop(0)() if self.script else reply([tool_use(self.calls, "view_cart")], "tool_use")


def test_full_session_buys_and_reports():
    store = Store("headphones", PRODUCTS, 1)
    code = store.listing[0][0]
    client = FakeClient([
        lambda: reply([tool_use(1, "search", query="headphones")], "tool_use"),
        lambda: reply([tool_use(2, "view_product", product_id=code), tool_use(3, "add_to_cart", product_id=code)], "tool_use"),
        lambda: reply([tool_use(4, "checkout")], "tool_use"),
        lambda: reply([SimpleNamespace(type="text", text="I bought it.")], "end_turn"),
    ])
    result = run_agent(client, "claude-haiku-4-5", "sys", TOOLS, "buy headphones", store.call)
    assert result["end_reason"] == "end_turn"
    assert result["report"] == "I bought it."
    assert result["steps"] == 4 and result["api_calls"] == 4
    assert store.order["code"] == code
    assert result["usage"]["input"] == 400
    assert result["cost_usd"] > 0


def test_step_cap_stops_a_looping_agent():
    store = Store("headphones", PRODUCTS, 1)
    client = FakeClient([])  # keeps calling view_cart forever
    result = run_agent(client, "claude-haiku-4-5", "sys", TOOLS, "buy", store.call, max_steps=5)
    assert result["end_reason"] == "step_cap"
    assert result["steps"] == 5
    assert store.order is None


def test_outcome_scores_regret():
    scenario = {"ranking": [
        {"id": "a", "utility": 0.8, "feasible": True, "rank": 1},
        {"id": "b", "utility": 0.6, "feasible": True, "rank": 2},
        {"id": "c", "utility": None, "feasible": False, "rank": 3},
    ]}
    assert outcome(scenario, "a")["regret"] == 0 and outcome(scenario, "a")["chose_best"]
    assert outcome(scenario, "b")["regret"] == 0.25
    assert outcome(scenario, "c")["regret"] == 1.0
    assert outcome(scenario, None)["regret"] == 1.0
