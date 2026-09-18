"""Small dependency-light tests for the InterSelf action channel."""

import unittest

import numpy as np


def channel(alpha, destination):
  alpha = np.asarray(alpha, dtype=np.float64)
  destination = np.asarray(destination, dtype=np.float64)
  identity = np.eye(destination.shape[-1], dtype=np.float64)
  return (
      (1.0 - alpha[..., :, None]) * identity
      + alpha[..., :, None] * destination)


def effective_action(intended, previous, matrix, rho):
  mapped = np.asarray(intended) @ np.asarray(matrix)
  return (1.0 - rho) * mapped + rho * np.asarray(previous)


class ChannelMathTest(unittest.TestCase):

  def setUp(self):
    self.destination = np.array([
        [0.10, 0.70, 0.20],
        [0.25, 0.25, 0.50],
        [0.60, 0.15, 0.25],
    ])

  def test_rows_remain_probabilities(self):
    matrix = channel(np.array([0.2, 0.4, 0.8]), self.destination)
    self.assertTrue(np.all(matrix >= 0.0))
    np.testing.assert_allclose(matrix.sum(-1), np.ones(3))

  def test_identity_limit(self):
    matrix = channel(np.zeros(3), self.destination)
    np.testing.assert_allclose(matrix, np.eye(3))
    intended = np.array([0.0, 1.0, 0.0])
    previous = np.array([1.0, 0.0, 0.0])
    output = effective_action(intended, previous, matrix, rho=0.0)
    np.testing.assert_allclose(output, intended)

  def test_temporal_persistence(self):
    matrix = np.eye(3)
    intended = np.array([0.0, 1.0, 0.0])
    previous = np.array([1.0, 0.0, 0.0])
    output = effective_action(intended, previous, matrix, rho=0.35)
    expected = 0.65 * intended + 0.35 * previous
    np.testing.assert_allclose(output, expected)
    self.assertAlmostEqual(float(output.sum()), 1.0)

  def test_identity_control_changes_only_transition_input(self):
    matrix = channel(np.full(3, 0.7), self.destination)
    intended = np.array([0.0, 0.0, 1.0])
    previous = np.array([1.0, 0.0, 0.0])
    inferred = effective_action(intended, previous, matrix, rho=0.25)
    full_transition_action = inferred
    identity_transition_action = intended
    self.assertFalse(np.allclose(full_transition_action, intended))
    np.testing.assert_allclose(identity_transition_action, intended)


if __name__ == '__main__':
  unittest.main()
