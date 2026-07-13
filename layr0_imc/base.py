# -*- coding: utf-8 -*-
"""
layr0_imc REST API Documentation - Base API Class
    https://docs.layr0.org
"""

import httpx

class BaseAPI:
    """
    Base class to handle all the API calls to layr0_imc.
    """

    def __init__(self, api_key, host="http://127.0.0.1:5000", version="v1", timeout=120.0):
        """
        Initialize the api object with an API key and optionally a host URL and API version.

        Attributes:
        - api_key (str): User's API key.
        - host (str): Base URL for the API endpoints. Defaults to localhost.
        - version (str): API version. Defaults to "v1".
        - timeout (float): Request timeout in seconds. Defaults to 120.0 seconds.
        """
        self.api_key = api_key
        self.base_url = f"{host}/api/{version}/"
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.timeout = timeout
        self.client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=120.0
            )
        )

    def close(self):
        """Close the shared HTTP client and any pooled connections."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _post(self, endpoint, payload):
        """Make a POST request with shared error handling."""
        url = self.base_url + endpoint
        try:
            response = self.client.post(url, json=payload, headers=self.headers)
            return self._handle_response(response)
        except httpx.TimeoutException:
            return {
                'status': 'error',
                'message': 'Request timed out. The server took too long to respond.',
                'error_type': 'timeout_error'
            }
        except httpx.ConnectError:
            return {
                'status': 'error',
                'message': 'Failed to connect to the server. Please check if the server is running.',
                'error_type': 'connection_error'
            }
        except httpx.HTTPError as e:
            return {
                'status': 'error',
                'message': f'HTTP error occurred: {str(e)}',
                'error_type': 'http_error'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'An unexpected error occurred: {str(e)}',
                'error_type': 'unknown_error'
            }

    def _get(self, endpoint, params):
        """Make a GET request with shared error handling."""
        url = self.base_url + endpoint
        try:
            response = self.client.get(url, params=params)
            return self._handle_response(response)
        except httpx.TimeoutException:
            return {
                'status': 'error',
                'message': 'Request timed out. The server took too long to respond.',
                'error_type': 'timeout_error'
            }
        except httpx.ConnectError:
            return {
                'status': 'error',
                'message': 'Failed to connect to the server. Please check if the server is running.',
                'error_type': 'connection_error'
            }
        except httpx.HTTPError as e:
            return {
                'status': 'error',
                'message': f'HTTP error occurred: {str(e)}',
                'error_type': 'http_error'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'An unexpected error occurred: {str(e)}',
                'error_type': 'unknown_error'
            }

    def _make_request(self, endpoint, payload):
        """Backward-internal helper for existing mixins."""
        return self._post(endpoint, payload)

    def _handle_response(self, response):
        """Return successful payloads and preserve structured IMC error payloads."""
        try:
            data = response.json()
        except ValueError:
            return {
                'status': 'error',
                'message': 'Invalid JSON response from server',
                'code': response.status_code,
                'raw_response': response.text,
                'error_type': 'json_error'
            }

        if not isinstance(data, dict):
            return data

        if response.status_code != 200:
            result = dict(data)
            result.setdefault('status', 'error')
            result.setdefault('message', f'HTTP {response.status_code}: {response.text}')
            result.setdefault('code', response.status_code)
            result.setdefault('error_type', 'http_error')
            return result

        if data.get('status') == 'error':
            result = dict(data)
            result.setdefault('code', response.status_code)
            result.setdefault('error_type', 'api_error')
            return result

        return data
