"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Running tests: pytest -xvs tests/driver/test_drevo_version_check.py
#
# Covers the drevo server-version compatibility check (graphiti option "D"):
# DrevoDriver probes `CALL drevo.info()` (drevo#303/#304, v0.0.18+) at setup and
# fails fast when the connected drevo is older than MINIMUM_DREVO_VERSION, while
# tolerating a drevo too old to expose drevo.info at all (can't verify -> warn).

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from graphiti_core.driver.drevo_driver import (
        MINIMUM_DREVO_VERSION,
        DrevoDriver,
        _parse_semver,
    )

    HAS_NEO4J = True
except ImportError:
    DrevoDriver = None
    MINIMUM_DREVO_VERSION = None
    _parse_semver = None
    HAS_NEO4J = False


# ---------------------------------------------------------------------------
# Version parsing / comparison
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_NEO4J, reason='neo4j driver package is not installed')
class TestParseSemver:
    def test_plain_semver(self):
        assert _parse_semver('0.0.18') == (0, 0, 18)

    def test_strips_v_prefix(self):
        assert _parse_semver('v0.0.16') == (0, 0, 16)

    def test_ordering_patch(self):
        assert _parse_semver('0.0.15') < _parse_semver('0.0.16')

    def test_ordering_minor_beats_patch(self):
        assert _parse_semver('0.1.0') > _parse_semver('0.0.99')

    def test_tolerates_prerelease_suffix(self):
        # a build/pre-release suffix degrades to the numeric prefix, never raises
        assert _parse_semver('0.0.18-rc1') == (0, 0, 18)

    def test_minimum_constant_is_parseable(self):
        assert _parse_semver(MINIMUM_DREVO_VERSION) >= (0, 0, 16)


# ---------------------------------------------------------------------------
# Negotiation on the DrevoDriver
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_NEO4J, reason='neo4j driver package is not installed')
class TestDrevoVersionNegotiation:
    def _make_driver(self) -> 'DrevoDriver':
        with patch('graphiti_core.driver.neo4j_driver.AsyncGraphDatabase') as mock_gdb:
            mock_gdb.driver.return_value = MagicMock()
            return DrevoDriver(uri='bolt://localhost:7687', user='neo4j', password='password')

    def _info(self, version: str) -> dict:
        return {
            'version': version,
            'git_sha': '20c263d4',
            'build_date': '2026-08-14T19:32:26Z',
            'protocol': 1,
        }

    @pytest.mark.asyncio
    async def test_accepts_when_version_meets_minimum(self):
        """drevo.info reports a version >= minimum -> setup proceeds, no raise."""
        driver = self._make_driver()
        exec_mock = AsyncMock(return_value=([self._info('0.0.18')], None, None))
        with patch.object(driver, 'execute_query', exec_mock):
            await driver.build_indices_and_constraints()  # must not raise

        emitted = ' '.join(str(call.args[0]) for call in exec_mock.await_args_list)
        assert 'drevo.info' in emitted

    @pytest.mark.asyncio
    async def test_raises_when_version_below_minimum(self):
        """drevo.info reports a version < minimum -> fail fast with upgrade message."""
        driver = self._make_driver()
        exec_mock = AsyncMock(return_value=([self._info('0.0.14')], None, None))
        with (
            patch.object(driver, 'execute_query', exec_mock),
            pytest.raises(RuntimeError, match=str(MINIMUM_DREVO_VERSION)),
        ):
            await driver.build_indices_and_constraints()

    @pytest.mark.asyncio
    async def test_warns_and_continues_when_info_procedure_absent(self, caplog):
        """drevo older than v0.0.18 has no drevo.info -> cannot verify, warn, continue."""
        driver = self._make_driver()
        exec_mock = AsyncMock(side_effect=Exception('no such procedure `drevo.info`'))
        with patch.object(driver, 'execute_query', exec_mock), caplog.at_level('WARNING'):
            await driver.build_indices_and_constraints()  # must not raise

        assert any('drevo' in rec.message.lower() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_warns_when_version_field_missing(self, caplog):
        """drevo.info present but no version field -> cannot verify, warn, no raise."""
        driver = self._make_driver()
        exec_mock = AsyncMock(return_value=([{'git_sha': 'abc', 'protocol': 1}], None, None))
        with patch.object(driver, 'execute_query', exec_mock), caplog.at_level('WARNING'):
            await driver.build_indices_and_constraints()  # must not raise

        assert any('version' in rec.message.lower() for rec in caplog.records)
