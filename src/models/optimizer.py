"""Revenue Optimization Layer.

Given a demand forecast and price elasticity estimate, computes
the price that maximizes expected revenue:

    Revenue(P) = P * Demand(P)
    Demand(P) = D0 * (P / P0) ^ elasticity

Uses scipy.optimize.minimize_scalar with bounded search.
This is the key differentiator — most portfolios predict, we optimize.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result of price optimization."""
    optimal_price: float
    expected_revenue: float
    expected_demand: float
    price_range: tuple[float, float]  # (lower, upper) at 90% CI
    base_price: float
    base_demand: float
    elasticity: float
    revenue_at_base: float
    revenue_lift_pct: float


def demand_function(price: float, base_demand: float, base_price: float, elasticity: float) -> float:
    """Constant-elasticity demand function.

    D(P) = D0 * (P / P0) ^ elasticity

    Args:
        price: Price to evaluate.
        base_demand: Demand at the base price (D0).
        base_price: Reference/market price (P0).
        elasticity: Price elasticity of demand (typically negative).

    Returns:
        Expected demand at the given price.
    """
    if base_price <= 0 or price <= 0:
        return 0.0
    return base_demand * (price / base_price) ** elasticity


def revenue_function(price: float, base_demand: float, base_price: float, elasticity: float) -> float:
    """Revenue = Price * Demand(Price).

    Args:
        price: Price to evaluate.
        base_demand: Demand at base price.
        base_price: Reference price.
        elasticity: Price elasticity.

    Returns:
        Expected revenue at the given price.
    """
    demand = demand_function(price, base_demand, base_price, elasticity)
    return price * demand


def negative_revenue(price: float, base_demand: float, base_price: float, elasticity: float) -> float:
    """Negative revenue for minimization (scipy minimizes, we want to maximize)."""
    return -revenue_function(price, base_demand, base_price, elasticity)


def get_optimal_price(
    base_demand: float,
    base_price: float,
    elasticity: float,
    floor_price: float | None = None,
    ceiling_price: float | None = None,
    config: dict | None = None,
) -> OptimizationResult:
    """Find the price that maximizes revenue.

    Args:
        base_demand: Expected demand at the current market price (0-1 for probability).
        base_price: Current market/reference price.
        elasticity: Price elasticity coefficient (should be negative).
        floor_price: Minimum allowed price. Defaults to base_price * 0.7.
        ceiling_price: Maximum allowed price. Defaults to base_price * 1.5.
        config: Configuration dict.

    Returns:
        OptimizationResult with optimal price and supporting metrics.
    """
    if config is None:
        config = load_config()

    opt_cfg = config["optimization"]

    if floor_price is None:
        floor_price = base_price * opt_cfg["price_floor_multiplier"]
    if ceiling_price is None:
        ceiling_price = base_price * opt_cfg["price_ceiling_multiplier"]

    # Ensure valid bounds
    floor_price = max(floor_price, 1.0)  # Never go below $1
    ceiling_price = max(ceiling_price, floor_price + 1.0)

    # Handle edge cases
    if base_demand <= 0:
        logger.warning("Base demand is 0 or negative. Returning floor price.")
        return OptimizationResult(
            optimal_price=floor_price,
            expected_revenue=0.0,
            expected_demand=0.0,
            price_range=(floor_price, floor_price),
            base_price=base_price,
            base_demand=base_demand,
            elasticity=elasticity,
            revenue_at_base=0.0,
            revenue_lift_pct=0.0,
        )

    if elasticity >= 0:
        # Inelastic or Giffen good — edge case, cap at ceiling
        logger.warning(f"Non-negative elasticity ({elasticity}). Capping at ceiling price.")
        elasticity = -0.1  # Use small negative as fallback

    # Optimize
    result = minimize_scalar(
        negative_revenue,
        bounds=(floor_price, ceiling_price),
        method="bounded",
        args=(base_demand, base_price, elasticity),
    )

    optimal_price = result.x
    optimal_revenue = -result.fun
    optimal_demand = demand_function(optimal_price, base_demand, base_price, elasticity)

    # Revenue at base price for comparison
    revenue_at_base = revenue_function(base_price, base_demand, base_price, elasticity)
    revenue_lift = ((optimal_revenue - revenue_at_base) / revenue_at_base * 100) if revenue_at_base > 0 else 0.0

    # Confidence interval (heuristic: +/- 10% of optimal price)
    ci_lower = max(floor_price, optimal_price * 0.9)
    ci_upper = min(ceiling_price, optimal_price * 1.1)

    result_obj = OptimizationResult(
        optimal_price=round(optimal_price, 2),
        expected_revenue=round(optimal_revenue, 2),
        expected_demand=round(optimal_demand, 4),
        price_range=(round(ci_lower, 2), round(ci_upper, 2)),
        base_price=round(base_price, 2),
        base_demand=round(base_demand, 4),
        elasticity=round(elasticity, 4),
        revenue_at_base=round(revenue_at_base, 2),
        revenue_lift_pct=round(revenue_lift, 2),
    )

    logger.info(
        f"Optimization: base=${base_price:.2f} -> optimal=${optimal_price:.2f} "
        f"(+{revenue_lift:.1f}% revenue, elasticity={elasticity:.3f})"
    )
    return result_obj


def compute_revenue_curve(
    base_demand: float,
    base_price: float,
    elasticity: float,
    n_points: int = 100,
    floor_price: float | None = None,
    ceiling_price: float | None = None,
    config: dict | None = None,
) -> dict[str, list[float]]:
    """Compute price vs revenue curve for visualization.

    Args:
        base_demand: Demand at base price.
        base_price: Reference price.
        elasticity: Price elasticity.
        n_points: Number of points on the curve.
        floor_price: Min price for curve.
        ceiling_price: Max price for curve.
        config: Configuration dict.

    Returns:
        Dict with 'prices', 'revenues', and 'demands' lists.
    """
    if config is None:
        config = load_config()

    opt_cfg = config["optimization"]
    if floor_price is None:
        floor_price = base_price * opt_cfg["price_floor_multiplier"]
    if ceiling_price is None:
        ceiling_price = base_price * opt_cfg["price_ceiling_multiplier"]

    prices = np.linspace(max(floor_price, 1.0), ceiling_price, n_points)
    revenues = [revenue_function(p, base_demand, base_price, elasticity) for p in prices]
    demands = [demand_function(p, base_demand, base_price, elasticity) for p in prices]

    return {
        "prices": prices.tolist(),
        "revenues": revenues,
        "demands": demands,
    }
