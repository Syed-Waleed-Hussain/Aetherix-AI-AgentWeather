"""Opt-in, read-only verification of a previously uploaded Day 5B sample."""

import os
import unittest

from aetherix_weather.hopsworks_features import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    read_feature_frame,
    select_sync_rows,
    verify_secret_ignored,
)
from scripts import sync_hopsworks_features as sync


@unittest.skipUnless(
    os.environ.get("AETHERIX_HOPSWORKS_LIVE_READ") == "1",
    "Set AETHERIX_HOPSWORKS_LIVE_READ=1 after an authorized sample upload.",
)
class HopsworksLiveReadTest(unittest.TestCase):
    def test_existing_sample_matches_local_data(self) -> None:
        verify_secret_ignored(sync.PROJECT_ROOT)
        frame, _ = read_feature_frame(
            sync.PROJECT_ROOT / "data" / "processed" / "features.csv"
        )
        sample = select_sync_rows(frame, sample=True)
        feature_store = sync._connect_to_feature_store()
        try:
            with sync._quiet_sdk_output():
                group = feature_store.get_feature_group(
                    name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION
                )
                sync._verify_feature_group_schema(group)
        except Exception as exc:
            self.fail(f"Existing Feature Group could not be inspected ({type(exc).__name__}).")
        sync._wait_and_verify_read_back(group, sample)
