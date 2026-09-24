#!/usr/bin/env python3

# pyre-unsafe

import unittest
import warnings

import torch
from captum.concept._utils.classifier import FastCAV


class TestFastCAV(unittest.TestCase):
    def setUp(self):
        """Set up simple, deterministic data for tests."""
        # Balanced, well-separated, non-deterministic data
        self.x_train_balanced_randn = torch.cat(
            [
                torch.randn(10, 5),  # Class 0
                torch.randn(10, 5),  # Class 1
            ]
        )
        self.y_train_balanced_randn = torch.cat(
            [
                torch.zeros(10),
                torch.ones(10),
            ]
        ).int()

        # Imbalanced data (triggers warning)
        self.x_train_imbalanced = torch.cat(
            [
                torch.randn(15, 5),  # Class 0
                torch.randn(5, 5),  # Class 1
            ]
        )
        self.y_train_imbalanced = torch.cat(
            [
                torch.zeros(15),
                torch.ones(5),
            ]
        ).int()

        # Simple, deterministic data for predictable results
        self.x_train_simple = torch.tensor(
            [[-1.0, -1.0], [-2.0, -2.0], [1.0, 1.0], [2.0, 2.0]]
        )
        self.y_train_simple = torch.tensor([0, 0, 1, 1]).int()

        # Test data for simple model
        self.x_test_simple = torch.tensor(
            [
                [-10.0, -10.0],  # Should be class 0
                [10.0, 10.0],  # Should be class 1
                [0.0, 0.0],  # Should be class 0 (on boundary)
            ]
        )
        self.expected_pred_simple = torch.tensor([[0], [1], [0]])

    def test_init(self):
        """Test FastCAV initialization."""
        cav = FastCAV()
        self.assertIsNone(cav.coef_)
        self.assertIsNone(cav.intercept_)
        self.assertIsNone(cav.mean)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FastCAV(foo="bar", baz=123)
            self.assertTrue(
                any(
                    "FastCAV does not support any additional parameters"
                    in str(warn.message)
                    for warn in w
                )
            )

    def test_fit_balanced(self):
        """Test fitting with balanced, deterministic data."""
        cav = FastCAV()
        cav.fit(self.x_train_simple, self.y_train_simple)

        self.assertIsNotNone(cav.coef_)
        self.assertIsNotNone(cav.intercept_)
        self.assertIsNotNone(cav.mean)
        self.assertIsNotNone(cav.classes_)

        self.assertEqual(cav.coef_.shape, (1, 2))
        self.assertEqual(cav.intercept_.shape, (1, 1))
        self.assertEqual(cav.mean.shape, (2,))
        self.assertEqual(cav.classes_.shape, (2,))

        expected_mean = torch.tensor([0.0, 0.0])
        torch.testing.assert_close(cav.mean, expected_mean)

        expected_coef = torch.tensor([[1.5, 1.5]])
        torch.testing.assert_close(cav.coef_, expected_coef)

        expected_intercept = torch.tensor([[0.0]])
        torch.testing.assert_close(cav.intercept_, expected_intercept)

        self.assertTrue(torch.equal(cav.classes_, torch.tensor([0, 1])))

    def test_fit_imbalanced_warns(self):
        """Test that fitting with imbalanced data raises a warning."""
        cav = FastCAV()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cav.fit(self.x_train_imbalanced, self.y_train_imbalanced)
            self.assertTrue(
                any("Classes are imbalanced" in str(warn.message) for warn in w)
            )

        self.assertIsNotNone(cav.coef_)
        self.assertIsNotNone(cav.intercept_)

    def test_fit_assertions(self):
        """Test assertions for invalid input shapes and labels."""
        cav = FastCAV()
        with self.assertRaises(AssertionError) as cm:
            cav.fit(torch.randn(20), self.y_train_balanced_randn)
        self.assertIn("Input tensor must be 2D", str(cm.exception))

        with self.assertRaises(AssertionError) as cm:
            cav.fit(self.x_train_balanced_randn, torch.randn(20, 1))
        self.assertIn("Labels tensor must be 1D", str(cm.exception))

        with self.assertRaises(AssertionError) as cm:
            cav.fit(self.x_train_balanced_randn, torch.zeros(5))
        self.assertIn("Input and labels must have same batch size", str(cm.exception))

        y_multi = self.y_train_balanced_randn.clone()
        y_multi[0] = 2
        with self.assertRaises(AssertionError) as cm:
            cav.fit(self.x_train_balanced_randn, y_multi)
        self.assertIn("Only binary classification is supported", str(cm.exception))

    def test_predict_before_fit(self):
        """Test that predict raises an error if called before fit."""
        cav = FastCAV()
        with self.assertRaises(ValueError) as cm:
            cav.predict(self.x_test_simple)
        self.assertIn("Model not trained. Call fit() first.", str(cm.exception))

    def test_predict_after_fit(self):
        """Test prediction on single and batch inputs after fitting."""
        cav = FastCAV()
        cav.fit(self.x_train_simple, self.y_train_simple)

        predictions = cav.predict(self.x_test_simple)
        self.assertTrue(torch.equal(predictions, self.expected_pred_simple))

        prediction_single_0 = cav.predict(self.x_test_simple[0])
        self.assertEqual(
            prediction_single_0.item(), self.expected_pred_simple[0].item()
        )

        prediction_single_1 = cav.predict(self.x_test_simple[1])
        self.assertEqual(
            prediction_single_1.item(), self.expected_pred_simple[1].item()
        )

    def test_fit_zero_mean_difference(self):
        """Test fitting when class means are identical."""
        cav = FastCAV()
        x_train = torch.tensor([[-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]])
        y_train = torch.tensor([0, 0, 1, 1]).int()

        cav.fit(x_train, y_train)

        torch.testing.assert_close(cav.coef_, torch.zeros_like(cav.coef_))
        torch.testing.assert_close(cav.intercept_, torch.zeros_like(cav.intercept_))

        predictions = cav.predict(torch.randn(5, 2))
        self.assertTrue(torch.all(predictions == cav.classes_[0]))

    def test_classes_before_fit(self):
        """Test that classes raises an error if called before fit."""
        cav = FastCAV()
        with self.assertRaises(ValueError) as cm:
            cav.classes()
        self.assertIn("Please call `fit` to train the model first.", str(cm.exception))

    def test_classes_after_fit(self):
        """Test the classes method after fitting."""
        cav = FastCAV()
        cav.fit(self.x_train_simple, self.y_train_simple)
        classes = cav.classes()
        self.assertTrue(torch.equal(classes, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(classes, cav.classes_))

        y_train_custom_labels = torch.tensor([2, 2, 5, 5]).int()
        cav.fit(self.x_train_simple, y_train_custom_labels)
        classes = cav.classes()
        self.assertTrue(torch.equal(classes, torch.tensor([2, 5])))
