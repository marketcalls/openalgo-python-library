import httpx

from layr0_imc.orders import OrderAPI


class RecordingClient:
    def __init__(self, response):
        self.response = response
        self.posts = []
        self.closed = False

    def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return self.response

    def get(self, url, params=None):
        return self.response

    def close(self):
        self.closed = True


def test_positionsopen_posts_to_positionsopen_endpoint():
    api = OrderAPI("test-key", host="http://imc.local")
    api.client = RecordingClient(httpx.Response(200, json={"status": "success", "quantity": 0}))

    result = api.positionsopen(
        strategy="test-strategy",
        symbol="SBIN",
        exchange="NSE",
        product="CNC",
        expected_mode="live",
        expected_balance_type="live",
        expected_mode_version=7,
        request_id="req-1",
    )

    assert result == {"status": "success", "quantity": 0}
    assert api.client.posts[0]["url"] == "http://imc.local/api/v1/positionsopen"
    assert api.client.posts[0]["json"]["expected_mode"] == "live"
    assert api.client.posts[0]["json"]["expected_balance_type"] == "live"
    assert api.client.posts[0]["json"]["expected_mode_version"] == "7"
    assert api.client.posts[0]["json"]["request_id"] == "req-1"


def test_legacy_position_method_is_not_exposed():
    api = OrderAPI("test-key")

    legacy_name = "open" + "position"
    assert not hasattr(api, legacy_name)


def test_unsupported_notification_method_is_not_exposed():
    from layr0_imc import api as ImcApi

    client = ImcApi("test-key")
    unsupported_name = "tele" + "gram"

    assert not hasattr(client, unsupported_name)


def test_structured_mode_error_payload_is_preserved():
    error_payload = {
        "status": "error",
        "message": "Server mode switch detected",
        "error_code": "SERVER_MODE_SWITCH_DETECTED",
        "mode": "analyze",
        "balance_type": "sandbox",
        "mode_version": 8,
        "data": {"required_action": "refresh_mode_context"},
    }
    api = OrderAPI("test-key")
    api.client = RecordingClient(httpx.Response(409, json=error_payload))

    result = api.positionsopen(
        strategy="test-strategy",
        symbol="SBIN",
        exchange="NSE",
        product="CNC",
    )

    assert result["status"] == "error"
    assert result["code"] == 409
    assert result["error_code"] == "SERVER_MODE_SWITCH_DETECTED"
    assert result["mode"] == "analyze"
    assert result["data"]["required_action"] == "refresh_mode_context"


def test_shared_http_client_can_be_closed():
    api = OrderAPI("test-key")
    replacement = RecordingClient(httpx.Response(200, json={"status": "success"}))
    api.client = replacement

    api.close()

    assert replacement.closed is True
