import asyncio
import base64
import hashlib
import hmac
import json
from typing import Awaitable
from unittest import TestCase
from unittest.mock import MagicMock

from hummingbot.connector.derivative.kucoin_perpetual import kucoin_perpetual_constants as CONSTANTS
from hummingbot.connector.derivative.kucoin_perpetual.kucoin_perpetual_auth import KucoinPerpetualAuth
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, RESTRequest, WSJSONRequest


class KucoinPerpetualAuthTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.api_key = "testApiKey"
        self.secret_key = "testSecretKey"
        self.passphrase = "testPassphrase"
        self.subaccount_name = "test!?Subaccount"

        self.mock_time_provider = MagicMock()
        self.mock_time_provider.time.return_value = 1000000

        self.auth = KucoinPerpetualAuth(
            api_key=self.api_key,
            passphrase=self.passphrase,
            secret_key=self.secret_key,
            time_provider=self.mock_time_provider,
        )

    def async_run_with_timeout(self, coroutine: Awaitable, timeout: int = 1):
        ret = asyncio.get_event_loop().run_until_complete(asyncio.wait_for(coroutine, timeout))
        return ret

    def _sign(self, passphrase: str, key: str) -> str:
        signed_message = base64.b64encode(
            hmac.new(key.encode("utf-8"), passphrase.encode("utf-8"), hashlib.sha256).digest()
        )
        return signed_message.decode("utf-8")

    def test_add_auth_headers_to_get_request_without_params(self):
        request = RESTRequest(
            method=RESTMethod.GET,
            url="https://test.url/api/endpoint",
            is_auth_required=True,
            throttler_limit_id="/api/endpoint",
        )

        self.async_run_with_timeout(self.auth.rest_authenticate(request, use_time_provider=1))

        self.assertEqual(self.api_key, request.headers["KC-API-KEY"])
        self.assertEqual("1000000", request.headers["KC-API-TIMESTAMP"])
        self.assertEqual("2", request.headers["KC-API-KEY-VERSION"])
        expected_signature = self._sign("1000000" + "GET" + request.throttler_limit_id, key=self.secret_key)
        self.assertEqual(expected_signature, request.headers["KC-API-SIGN"])
        expected_passphrase = self._sign(self.passphrase, key=self.secret_key)
        self.assertEqual(expected_passphrase, request.headers["KC-API-PASSPHRASE"])

        self.assertEqual(CONSTANTS.HB_PARTNER_ID, request.headers["KC-API-PARTNER"])
        expected_partner_signature = self._sign(
            "1000000" + CONSTANTS.HB_PARTNER_ID + self.api_key, key=CONSTANTS.HB_PARTNER_KEY
        )
        self.assertEqual(expected_partner_signature, request.headers["KC-API-PARTNER-SIGN"])

    def test_add_auth_headers_to_get_request_with_params(self):
        request = RESTRequest(
            method=RESTMethod.GET,
            url="https://test.url/api/endpoint",
            params={"param1": "value1", "param2": "value2"},
            is_auth_required=True,
            throttler_limit_id="/api/endpoint",
        )

        self.async_run_with_timeout(self.auth.rest_authenticate(request, use_time_provider=1))

        self.assertEqual(self.api_key, request.headers["KC-API-KEY"])
        self.assertEqual("1000000", request.headers["KC-API-TIMESTAMP"])
        self.assertEqual("2", request.headers["KC-API-KEY-VERSION"])
        full_endpoint = f"{request.throttler_limit_id}?param1=value1&param2=value2"
        expected_signature = self._sign("1000000" + "GET" + full_endpoint, key=self.secret_key)
        self.assertEqual(expected_signature, request.headers["KC-API-SIGN"])
        expected_passphrase = self._sign(self.passphrase, key=self.secret_key)
        self.assertEqual(expected_passphrase, request.headers["KC-API-PASSPHRASE"])

        self.assertEqual(CONSTANTS.HB_PARTNER_ID, request.headers["KC-API-PARTNER"])
        expected_partner_signature = self._sign(
            "1000000" + CONSTANTS.HB_PARTNER_ID + self.api_key, key=CONSTANTS.HB_PARTNER_KEY
        )
        self.assertEqual(expected_partner_signature, request.headers["KC-API-PARTNER-SIGN"])

    def test_add_auth_headers_to_post_request(self):
        body = {"param_z": "value_param_z", "param_a": "value_param_a"}
        request = RESTRequest(
            method=RESTMethod.POST,
            url="https://test.url/api/endpoint",
            data=json.dumps(body),
            is_auth_required=True,
            throttler_limit_id="/api/endpoint",
        )

        self.async_run_with_timeout(self.auth.rest_authenticate(request, use_time_provider=1))

        self.assertEqual(self.api_key, request.headers["KC-API-KEY"])
        self.assertEqual("1000000", request.headers["KC-API-TIMESTAMP"])
        self.assertEqual("2", request.headers["KC-API-KEY-VERSION"])
        expected_signature = self._sign(
            "1000000" + "POST" + request.throttler_limit_id + json.dumps(body), key=self.secret_key
        )
        self.assertEqual(expected_signature, request.headers["KC-API-SIGN"])
        expected_passphrase = self._sign(self.passphrase, key=self.secret_key)
        self.assertEqual(expected_passphrase, request.headers["KC-API-PASSPHRASE"])

        self.assertEqual(CONSTANTS.HB_PARTNER_ID, request.headers["KC-API-PARTNER"])
        expected_partner_signature = self._sign(
            "1000000" + CONSTANTS.HB_PARTNER_ID + self.api_key, key=CONSTANTS.HB_PARTNER_KEY
        )
        self.assertEqual(expected_partner_signature, request.headers["KC-API-PARTNER-SIGN"])

    def test_no_auth_added_to_wsrequest(self):
        payload = {"param1": "value_param_1"}
        request = WSJSONRequest(payload=payload, is_auth_required=True)

        self.async_run_with_timeout(self.auth.ws_authenticate(request))

        self.assertEqual(payload, request.payload)

    def test_ws_auth_payload(self):
        expires = self.auth._get_expiration_timestamp()
        self.mock_time_provider.return_value = expires
        payload = self.auth.get_ws_auth_payload()

        raw_signature = "GET/realtime" + expires
        expected_signature = hmac.new(
            self.secret_key.encode("utf-8"), raw_signature.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        self.assertEqual(3, len(payload))
        self.assertEqual(self.api_key, payload[0])
        self.assertEqual(expires, payload[1])
        self.assertEqual(expected_signature, payload[2])
