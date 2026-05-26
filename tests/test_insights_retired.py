#!/usr/bin/env python3
# pyre-strict

from __future__ import annotations

import unittest


class TestInsightsRetiredStub(unittest.TestCase):
    def test_import_insights_module(self) -> None:
        """Importing captum.insights should succeed (stub module exists)."""
        import captum.insights  # noqa: F811

        self.assertIsNotNone(captum.insights)

    def test_attribution_visualizer_raises(self) -> None:
        """Instantiating AttributionVisualizer should raise ImportError."""
        # Import is the subject under test
        from captum.insights import AttributionVisualizer

        with self.assertRaises(ImportError) as ctx:
            AttributionVisualizer()
        self.assertIn("retired", str(ctx.exception).lower())
        self.assertIn("v0.8.0", str(ctx.exception))

    def test_batch_raises(self) -> None:
        """Instantiating Batch should raise ImportError."""
        # Import is the subject under test
        from captum.insights import Batch

        with self.assertRaises(ImportError) as ctx:
            Batch()
        self.assertIn("retired", str(ctx.exception).lower())

    def test_unknown_attr_raises(self) -> None:
        """Accessing an unknown attribute should raise ImportError."""
        # Import is the subject under test
        import captum.insights

        with self.assertRaises(ImportError) as ctx:
            captum.insights.SomeNonexistentClass  # noqa: B018
        self.assertIn("SomeNonexistentClass", str(ctx.exception))
        self.assertIn("retired", str(ctx.exception).lower())

    def test_error_message_includes_migration_guidance(self) -> None:
        """Error message should include migration guidance to new API."""
        # Import is the subject under test
        from captum.insights import AttributionVisualizer

        with self.assertRaises(ImportError) as ctx:
            AttributionVisualizer()
        msg = str(ctx.exception)
        self.assertIn("visualize_image_attr", msg)
        self.assertIn("visualize_text", msg)
        self.assertIn("captum.ai", msg)
