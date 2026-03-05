"""Unit tests for the revenue optimizer."""

import pytest

from src.models.optimizer import (
    OptimizationResult,
    compute_revenue_curve,
    demand_function,
    get_optimal_price,
    revenue_function,
)


class TestDemandFunction:
    def test_demand_at_base_price_equals_base_demand(self):
        assert demand_function(100, 0.7, 100, -1.5) == pytest.approx(0.7)

    def test_higher_price_lower_demand(self):
        d_low = demand_function(100, 0.7, 100, -1.5)
        d_high = demand_function(150, 0.7, 100, -1.5)
        assert d_high < d_low

    def test_lower_price_higher_demand(self):
        d_base = demand_function(100, 0.7, 100, -1.5)
        d_low = demand_function(80, 0.7, 100, -1.5)
        assert d_low > d_base

    def test_zero_price_returns_zero(self):
        assert demand_function(0, 0.7, 100, -1.5) == 0.0

    def test_zero_base_price_returns_zero(self):
        assert demand_function(100, 0.7, 0, -1.5) == 0.0


class TestRevenueFunction:
    def test_positive_revenue(self):
        rev = revenue_function(100, 0.7, 100, -1.5)
        assert rev > 0

    def test_revenue_equals_price_times_demand(self):
        price = 120
        demand = demand_function(price, 0.7, 100, -1.5)
        rev = revenue_function(price, 0.7, 100, -1.5)
        assert rev == pytest.approx(price * demand)


class TestGetOptimalPrice:
    def test_returns_optimization_result(self):
        result = get_optimal_price(0.7, 150, -1.5)
        assert isinstance(result, OptimizationResult)

    def test_optimal_within_bounds(self):
        result = get_optimal_price(0.7, 150, -1.5, floor_price=100, ceiling_price=200)
        assert result.optimal_price >= 100
        assert result.optimal_price <= 200

    def test_revenue_at_optimal_greater_than_floor(self):
        result = get_optimal_price(0.7, 150, -1.5, floor_price=50, ceiling_price=300)
        rev_floor = revenue_function(50, 0.7, 150, -1.5)
        assert result.expected_revenue >= rev_floor

    def test_revenue_at_optimal_greater_than_ceiling(self):
        result = get_optimal_price(0.7, 150, -1.5, floor_price=50, ceiling_price=300)
        rev_ceil = revenue_function(300, 0.7, 150, -1.5)
        assert result.expected_revenue >= rev_ceil

    def test_handles_zero_demand(self):
        result = get_optimal_price(0.0, 150, -1.5)
        assert result.expected_revenue == 0.0

    def test_handles_positive_elasticity(self):
        # Should still return a result (fallback behavior)
        result = get_optimal_price(0.7, 150, 0.5)
        assert result.optimal_price > 0

    def test_price_range_ordered(self):
        result = get_optimal_price(0.7, 150, -1.5)
        assert result.price_range[0] <= result.price_range[1]

    def test_revenue_lift_computed(self):
        result = get_optimal_price(0.7, 150, -1.5)
        assert isinstance(result.revenue_lift_pct, float)

    def test_highly_elastic_market(self):
        # With very elastic demand, optimal price should be lower
        result = get_optimal_price(0.7, 150, -3.0)
        assert result.optimal_price <= 150

    def test_inelastic_market(self):
        # With inelastic demand, optimal price should be at ceiling
        result = get_optimal_price(0.7, 150, -0.3)
        assert result.optimal_price >= 150


class TestRevenueCurve:
    def test_returns_correct_keys(self):
        curve = compute_revenue_curve(0.7, 150, -1.5)
        assert "prices" in curve
        assert "revenues" in curve
        assert "demands" in curve

    def test_correct_length(self):
        curve = compute_revenue_curve(0.7, 150, -1.5, n_points=50)
        assert len(curve["prices"]) == 50
        assert len(curve["revenues"]) == 50
        assert len(curve["demands"]) == 50

    def test_prices_are_sorted(self):
        curve = compute_revenue_curve(0.7, 150, -1.5)
        assert curve["prices"] == sorted(curve["prices"])
