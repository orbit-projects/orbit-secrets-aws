# Copyright 2026-present Orbit Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AWS adapter request mapping, failure redaction, and resource lifecycle tests."""

import asyncio

import pytest
from orbit_secrets import (
    SecretAccessDeniedError,
    SecretProviderClosedError,
    SecretReference,
    SecretUnavailableError,
)

from orbit_secrets_aws import AWSSecretsProvider


class FakeClient:
    def __init__(self, response=None, error=None, gate=None):
        self.response = response or {"SecretString": "local-only", "VersionId": "v4"}
        self.error = error
        self.gate = gate
        self.request = None

    async def get_secret_value(self, **kwargs):
        self.request = kwargs
        if self.gate is not None:
            await self.gate.wait()
        if self.error:
            raise self.error
        return self.response


class ClientContext:
    def __init__(self, client):
        self.client = client
        self.closed = False

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *_):
        self.closed = True


@pytest.mark.asyncio
async def test_resolve_preserves_binary_and_version_and_closes_client():
    client = FakeClient({"SecretBinary": b"binary", "VersionId": "ver-9"})
    context = ClientContext(client)
    provider = AWSSecretsProvider(client_factory=lambda: context)
    resolved = await provider.resolve(SecretReference(name="local/test"))
    assert resolved.reveal() == b"binary"
    assert resolved.version == "ver-9"
    assert client.request == {"SecretId": "local/test"}
    await provider.aclose()
    await provider.aclose()
    assert context.closed
    with pytest.raises(SecretProviderClosedError):
        await provider.resolve(SecretReference(name="local/test"))


@pytest.mark.asyncio
async def test_explicit_version_is_sent_and_provider_message_is_discarded():
    class Denied(Exception):
        response = {"Error": {"Code": "AccessDeniedException", "Message": "secret token"}}

    client = FakeClient(error=Denied("secret token"))
    provider = AWSSecretsProvider(client_factory=lambda: ClientContext(client))
    with pytest.raises(SecretAccessDeniedError) as caught:
        await provider.resolve(SecretReference(name="local/test", version="v5"))
    assert client.request == {"SecretId": "local/test", "VersionId": "v5"}
    assert "secret token" not in str(caught.value)
    assert caught.value.__cause__ is None
    await provider.aclose()


@pytest.mark.asyncio
async def test_cancellation_propagates_and_close_waits_for_inflight_read():
    gate = asyncio.Event()
    client = FakeClient(gate=gate)
    context = ClientContext(client)
    provider = AWSSecretsProvider(client_factory=lambda: context)
    read = asyncio.create_task(provider.resolve(SecretReference(name="local/test")))
    while client.request is None:
        await asyncio.sleep(0)
    close = asyncio.create_task(provider.aclose())
    await asyncio.sleep(0)
    assert not close.done()
    gate.set()
    await read
    await close
    assert context.closed


@pytest.mark.asyncio
async def test_timeout_is_sanitized_and_retryable():
    class Slow(FakeClient):
        async def get_secret_value(self, **kwargs):
            await asyncio.sleep(1)

    from orbit_secrets_aws.config import AWSSecretsConfig

    provider = AWSSecretsProvider(
        AWSSecretsConfig(request_timeout_seconds=0.01),
        client_factory=lambda: ClientContext(Slow()),
    )
    with pytest.raises(SecretUnavailableError) as caught:
        await provider.resolve(SecretReference(name="local/test"))
    assert caught.value.retryable
    await provider.aclose()
