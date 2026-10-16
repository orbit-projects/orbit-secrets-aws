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
"""Validated AWS Secrets Manager client and request limits."""

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator


class AWSSecretsConfig(BaseModel):
    """Bounded options for one asynchronous AWS Secrets Manager client."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, validate_default=True)

    region_name: StrictStr | None = None
    request_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    connect_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    max_concurrency: StrictInt = Field(default=8, ge=1, le=128)
    max_attempts: StrictInt = Field(default=3, ge=1, le=5)

    @field_validator("region_name")
    @classmethod
    def validate_region(cls, value: str | None) -> str | None:
        """Reject whitespace and control characters in an optional region override."""
        if value is not None and (
            not value.strip()
            or len(value) > 64
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError("region_name must be bounded printable text.")
        return value
