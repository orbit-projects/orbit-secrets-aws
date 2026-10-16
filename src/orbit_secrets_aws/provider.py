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
"""Bounded asynchronous AWS Secrets Manager provider with explicit client ownership."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Protocol, cast

from orbit_secrets import (
    MAX_SECRET_BYTES,
    SecretAccessDeniedError,
    SecretError,
    SecretNotFoundError,
    SecretOperationError,
    SecretProviderClosedError,
    SecretReference,
    SecretUnavailableError,
    SecretValue,
    SecretVersionNotFoundError,
)

from orbit_secrets_aws.config import AWSSecretsConfig


class _AWSClient(Protocol):
    async def get_secret_value(self, **kwargs: Any) -> Mapping[str, Any]:
        """Fetch a secret value from AWS Secrets Manager."""


class _ClientContext(Protocol):
    async def __aenter__(self) -> _AWSClient: ...

    async def __aexit__(self, *args: object) -> None: ...


class AWSSecretsProvider:
    """Resolve secret values with the AWS SDK's default credential chain.

    The asynchronous client is created lazily, shared by a bounded number of requests, and
    closed only after active requests drain. The provider never caches values. ``refresh`` thus
    performs a new read and returns AWS's current version unless an exact version was requested.
    """

    def __init__(
        self,
        config: AWSSecretsConfig | None = None,
        *,
        client_factory: Callable[[], _ClientContext] | None = None,
    ) -> None:
        """Create the provider without resolving credentials or opening network resources."""
        self.config = config or AWSSecretsConfig()
        self._client_factory = client_factory or self._default_client_factory
        self._client: _AWSClient | None = None
        self._client_context: _ClientContext | None = None
        self._client_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._condition = asyncio.Condition()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._active = 0
        self._closing = False
        self._closed = False

    def _default_client_factory(self) -> _ClientContext:
        """Create an aiobotocore Secrets Manager client using standard AWS credentials."""
        from aiobotocore.config import AioConfig
        from aiobotocore.session import get_session

        return cast(
            _ClientContext,
            get_session().create_client(
                "secretsmanager",
                region_name=self.config.region_name,
                config=AioConfig(
                    connect_timeout=self.config.connect_timeout_seconds,
                    read_timeout=self.config.request_timeout_seconds,
                    retries={"mode": "standard", "total_max_attempts": self.config.max_attempts},
                ),
            ),
        )

    @asynccontextmanager
    async def _use_client(self) -> AsyncIterator[_AWSClient]:
        """Gate new work during shutdown and hold a slot until a request completes."""
        async with self._condition:
            if self._closing or self._closed:
                raise SecretProviderClosedError()
            self._active += 1
        try:
            async with self._semaphore:
                async with self._client_lock:
                    if self._client is None:
                        context = self._client_factory()
                        client = await context.__aenter__()
                        self._client_context = context
                        self._client = client
                    client = self._client
                yield client
        finally:
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()

    async def resolve(self, reference: SecretReference) -> SecretValue:
        """Fetch one current or explicitly versioned AWS secret, without exposing SDK errors."""
        if not isinstance(reference, SecretReference):
            raise TypeError("reference must be a SecretReference.")
        request: dict[str, str] = {"SecretId": reference.name}
        if reference.version is not None:
            request["VersionId"] = reference.version
        try:
            async with asyncio.timeout(self.config.request_timeout_seconds):
                async with self._use_client() as client:
                    response = await client.get_secret_value(**request)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise SecretUnavailableError() from None
        except SecretError:
            raise
        except Exception as exc:
            raise _map_error(exc, requested_version=reference.version is not None) from None
        return _decode_response(response)

    async def refresh(self, reference: SecretReference) -> SecretValue:
        """Fetch AWS's current version; an explicit version remains pinned by contract."""
        if not isinstance(reference, SecretReference):
            raise TypeError("reference must be a SecretReference.")
        if reference.version is not None:
            return await self.resolve(reference)
        return await self.resolve(reference)

    async def aclose(self) -> None:
        """Stop new reads, drain in-flight requests and close the owned AWS client context."""
        async with self._close_lock:
            if self._closed:
                return
            async with self._condition:
                self._closing = True
                await self._condition.wait_for(lambda: self._active == 0)
            try:
                if self._client_context is not None:
                    await self._client_context.__aexit__(None, None, None)
            except asyncio.CancelledError:
                raise
            except Exception:
                raise SecretOperationError("client-close-failed") from None
            else:
                self._closed = True

    async def __aenter__(self) -> AWSSecretsProvider:
        """Return this provider for ``async with`` ownership."""
        if self._closing or self._closed:
            raise SecretProviderClosedError()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close provider resources when leaving an async context."""
        await self.aclose()


def _decode_response(response: Mapping[str, Any]) -> SecretValue:
    """Validate bounded AWS response fields and discard every unrelated response attribute."""
    try:
        if not isinstance(response, Mapping):
            raise ValueError
        version = response.get("VersionId")
        if not isinstance(version, str):
            raise ValueError
        text = response.get("SecretString")
        binary = response.get("SecretBinary")
        if isinstance(text, str) and binary is None:
            value = text.encode("utf-8", errors="strict")
        elif isinstance(binary, bytes) and text is None:
            value = binary
        elif isinstance(binary, bytearray) and text is None:
            value = bytes(binary)
        else:
            raise ValueError
        if len(value) > MAX_SECRET_BYTES:
            raise ValueError
        return SecretValue(value, version)
    except (TypeError, ValueError, UnicodeError):
        raise SecretOperationError("invalid-provider-response") from None


def _map_error(exc: Exception, *, requested_version: bool) -> Exception:
    """Map AWS error codes to stable safe categories without retaining provider text."""
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping):
            candidate = error.get("Code")
            if isinstance(candidate, str):
                code = candidate
    if code in {"ResourceNotFoundException", "InvalidRequestException"}:
        return SecretVersionNotFoundError() if requested_version else SecretNotFoundError()
    if code in {"AccessDeniedException", "UnrecognizedClientException", "InvalidClientTokenId"}:
        return SecretAccessDeniedError()
    if code in {"ThrottlingException", "InternalServiceError", "ServiceUnavailableException"}:
        return SecretUnavailableError()
    if exc.__class__.__name__ in {
        "EndpointConnectionError",
        "ConnectTimeoutError",
        "ReadTimeoutError",
    }:
        return SecretUnavailableError()
    return SecretOperationError("provider-failure")


__all__ = ["AWSSecretsProvider"]
