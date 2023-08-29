"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

# daily_mean() tests
# def test_daily_mean_zeros():
#     """Test that mean function works for an array of zeros."""
#     from inflammation.models import daily_mean
#
#     test_input = np.array([[0, 0],
#                            [0, 0],
#                            [0, 0]])
#     test_result = np.array([0, 0])
#
#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_mean(test_input), test_result)
#
#
# def test_daily_mean_integers():
#     """Test that mean function works for an array of positive integers."""
#     from inflammation.models import daily_mean
#
#     test_input = np.array([[1, 2],
#                            [3, 4],
#                            [5, 6]])
#     test_result = np.array([3, 4])
#
#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_mean(test_input), test_result)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))



# daily_max() tests
# def test_daily_max_integers():
#     """Test that max function works for an array of positive integers."""
#     from inflammation.models import daily_max
#
#     test_input = np.array([[4, 2, 5],
#                            [1, 6, 2],
#                            [4, 1, 9]])
#     test_result = np.array([4, 6, 9])
#
#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_max(test_input), test_result)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [4, 2, 5], [1, 6, 2], [4, 1, 9] ], [4, 6, 9]),
        ([ [4, -2, 5], [1, -6, 2], [-4, -1, 9] ], [4, -1, 9]),
    ])
def test_daily_max(test, expected):
    """Test max function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


# daily_min() tests
# def test_daily_min():
#     """Test that min function works for an array of positive and negative integers."""
#     from inflammation.models import daily_min
#
#     test_input = np.array([[4, -2, 5],
#                            [1, -6, 2],
#                            [-4, -1, 9]])
#     test_result = np.array([-4, -6, 2])
#
#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_min(test_input), test_result)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [4, 2, 5], [1, 6, 2], [4, 1, 9] ], [1, 1, 2]),
        ([ [4, -2, 5], [1, -6, 2], [-4, -1, 9] ], [-4, -6, 2]),
    ])
def test_daily_min(test, expected):
    """Test min function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

# @pytest.mark.parametrize(
#     "test, expected",
#     [
#         ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
#         ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
#         ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]]),
#     ])
# def test_patient_normalise(test, expected):
#     """Test normalisation works for arrays of one and positive integers.
#        Assumption that test accuracy of two decimal places is sufficient."""
#     from inflammation.models import patient_normalise
#     npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)

# @pytest.mark.parametrize(
#     "test, expected, expect_raises",
#     [
#         ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#          [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#          None,
#          ),
#         ([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
#          [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
#          None,
#          ),
#         ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#          [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
#          None,
#          ),
#         ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
#          [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
#          ValueError,
#          ),
#         ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
#          [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
#          ValueError,
#          ),
#         ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#          [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
#          None,
#          ),
#     ])
# def test_patient_normalise(test, expected, expect_raises):
#     """Test normalisation works for arrays of one and positive integers."""
#     from inflammation.models import patient_normalise
#     if expect_raises is not None:
#         with pytest.raises(expect_raises):
#             npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
#     else:
#         npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)

@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         None,
         ),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
         [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
         None,
         ),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
         None,
         ),
        ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
         ValueError,
         ),
        ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
         ValueError,
         ),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
         None,
         ),
        (
            'hello',
            None,
            TypeError,
        ),
        (
            3,
            None,
            TypeError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        )
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""
    from inflammation.models import patient_normalise
    if isinstance(test, list):
        test = np.array(test)
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_almost_equal(patient_normalise(test), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalise(test), np.array(expected), decimal=2)

def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])
