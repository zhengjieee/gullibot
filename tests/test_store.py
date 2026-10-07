from store.store import Store


def product(pid, price, battery=20):
    return {"id": pid, "name": f"Phones {pid}", "brand": "Acme", "price": price, "rating": 4.2,
            "rating_count": 1234, "attributes": {"form_factor": "over_ear", "noise_cancelling": True,
                                                 "battery_hours": battery, "weight_g": 250.0}}


PRODUCTS = [product(f"hp-{i:03d}", 20 + 10 * i) for i in range(8)]


def codes(store):
    return [code for code, _ in store.listing]


def test_search_lists_all_products_in_seeded_order():
    a, b, c = Store("headphones", PRODUCTS, 1), Store("headphones", PRODUCTS, 1), Store("headphones", PRODUCTS, 2)
    page, is_error = a.search("anything")
    assert not is_error
    assert all(code in page for code in codes(a))
    assert [p["id"] for _, p in a.listing] == [p["id"] for _, p in b.listing]
    assert codes(a) == codes(b)
    assert [p["id"] for _, p in a.listing] != [p["id"] for _, p in c.listing]


def test_product_page_shows_specs_and_records_view():
    s = Store("headphones", PRODUCTS, 1)
    code = codes(s)[0]
    page, is_error = s.view_product(code.lower())  # IDs are case-insensitive
    assert not is_error
    assert "Battery life: 20 hours" in page and "Active noise cancelling: Yes" in page
    assert s.viewed == [code]


def test_unknown_product_is_an_error():
    s = Store("headphones", PRODUCTS, 1)
    _, is_error = s.view_product("NOPE00")
    assert is_error


def test_cart_holds_one_product_and_checkout_charges_it():
    s = Store("headphones", PRODUCTS, 1)
    first, second = codes(s)[:2]
    s.add_to_cart(first)
    page, _ = s.add_to_cart(second)
    assert "Removed" in page
    assert s.cart == second
    page, is_error = s.checkout()
    assert not is_error and "Order placed" in page
    assert s.order["product_id"] == s.by_code[second]["id"]


def test_checkout_needs_a_cart_and_happens_once():
    s = Store("headphones", PRODUCTS, 1)
    assert s.checkout()[1]  # empty cart is an error
    s.add_to_cart(codes(s)[0])
    s.checkout()
    assert s.checkout()[1]
    assert s.add_to_cart(codes(s)[1])[1]


def test_unknown_tool_is_an_error():
    assert Store("headphones", PRODUCTS, 1).call("refund", {})[1]
