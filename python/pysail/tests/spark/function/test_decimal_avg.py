"""Exact Spark decimal AVG rounding, including the partial/window paths."""

from decimal import Decimal

import pytest
from pyspark.sql.types import DecimalType


@pytest.mark.parametrize("function", ["avg", "mean"])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("numerator,count,expected", [(2, 3, "0.006667"), (1, 32, "0.000313"), (1, 3, "0.003333")])
def test_decimal_avg_rounding(spark, function, sign, numerator, count, expected):
    frame = spark.sql(
        f"SELECT {function}(CAST(CASE WHEN id=0 THEN {sign * numerator}/100.0 ELSE 0 END "
        f"AS DECIMAL(18,2))) a FROM range({count})"
    )
    assert frame.schema["a"].dataType == DecimalType(22, 6)
    assert frame.first().a == Decimal(expected) * sign


@pytest.mark.parametrize("function", ["avg", "mean"])
def test_decimal_avg_distinct_and_grouped(spark, function):
    frame = spark.sql(
        f"SELECT s, {function}(DISTINCT CAST(s*v AS DECIMAL(18,2))) a "
        "FROM VALUES (0.01),(0.01),(0.00),(0.04),(NULL) t(v) "
        "CROSS JOIN VALUES (1),(-1) signs(s) GROUP BY s ORDER BY s"
    )
    assert frame.schema["a"].dataType == DecimalType(22, 6)
    assert [row.a for row in frame.collect()] == [Decimal("-0.016667"), Decimal("0.016667")]


@pytest.mark.parametrize("function", ["avg", "mean"])
def test_decimal_avg_sliding_window(spark, function):
    frame = spark.sql(
        f"SELECT id, {function}(CAST(CASE WHEN id%3=0 THEN -0.02 ELSE 0 END AS DECIMAL(18,2))) "
        "OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) a FROM range(7) ORDER BY id"
    )
    assert frame.schema["a"].dataType == DecimalType(22, 6)
    assert [row.a for row in frame.collect()] == [Decimal("-0.020000"), Decimal("-0.010000")] + [
        Decimal("-0.006667")
    ] * 5


@pytest.mark.parametrize("rows", [0, 3])
def test_decimal_avg_empty_or_null(spark, rows):
    frame = spark.sql(f"SELECT avg(CAST(NULL AS DECIMAL(18,2))) a FROM range({rows})")
    assert frame.schema["a"].dataType == DecimalType(22, 6)
    assert frame.first().a is None
