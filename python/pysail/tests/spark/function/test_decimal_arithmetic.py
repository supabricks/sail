"""Spark decimal result types and exact values (TPC-DS #199 reductions)."""

from decimal import Decimal

import pytest
from pyspark.sql.types import DecimalType


@pytest.mark.parametrize(
    "precision,scale,result_precision,result_scale,value",
    [
        (15, 4, 35, 20, "0.33333333333333333333"),
        (17, 2, 37, 20, "0.33333333333333333333"),
        (7, 2, 17, 10, "0.3333333333"),
        (38, 18, 38, 6, "0.333333"),
    ],
)
def test_decimal_division(spark, precision, scale, result_precision, result_scale, value):
    frame = spark.sql(
        f"SELECT CAST(v AS DECIMAL({precision},{scale}))/CAST(3 AS DECIMAL({precision},{scale})) a "
        "FROM VALUES (-1),(1),(NULL) t(v) ORDER BY v"
    )
    assert frame.schema["a"].dataType == DecimalType(result_precision, result_scale)
    assert [r.a for r in frame.collect()] == [None, -Decimal(value), Decimal(value)]


@pytest.mark.parametrize(
    "expression,precision,scale,value",
    [
        ("v*100", 21, 2, "100.00"),
        ("v/3", 21, 6, "0.333333"),
        ("v+1", 18, 2, "2.00"),
        ("v-1", 18, 2, "0.00"),
        ("round(v/v,2)", 20, 2, "1.00"),
        ("v/(v+v+v)/3*100", 38, 17, "11.11111111111111111"),
    ],
)
def test_decimal_expression_types(spark, expression, precision, scale, value):
    frame = spark.sql(f"SELECT {expression} a FROM (SELECT CAST(1 AS DECIMAL(17,2)) v)")
    assert frame.schema["a"].dataType == DecimalType(precision, scale)
    assert frame.first().a == Decimal(value)


def test_decimal_nested_window_sum(spark):
    frame = spark.sql(
        "SELECT sum(v)*100/sum(sum(v)) OVER () a FROM "
        "(SELECT id%2 g, CAST(id+1 AS DECIMAL(7,2)) v FROM range(5)) GROUP BY g ORDER BY g"
    )
    assert frame.schema["a"].dataType == DecimalType(38, 17)
    assert [r.a for r in frame.collect()] == [Decimal("60.00000000000000000"), Decimal("40.00000000000000000")]


@pytest.mark.parametrize("ansi", ["true", "false"])
@pytest.mark.parametrize(
    "expression",
    [
        "CAST(1 AS DECIMAL(15,4))/CAST(0 AS DECIMAL(15,4))",
        "CAST('99999999999999999999999999999999999999' AS DECIMAL(38,0))*CAST(2 AS DECIMAL(1,0))",
    ],
)
def test_decimal_errors(spark, ansi, expression):
    original = spark.conf.get("spark.sql.ansi.enabled")
    try:
        spark.conf.set("spark.sql.ansi.enabled", ansi)
        if ansi == "true":
            with pytest.raises(Exception, match="(?i)(zero|overflow|out.of.range|cannot be represented)"):
                spark.sql(f"SELECT {expression} a").collect()
        else:
            frame = spark.sql(f"SELECT {expression} a")
            assert isinstance(frame.schema["a"].dataType, DecimalType)
            assert frame.first().a is None
    finally:
        spark.conf.set("spark.sql.ansi.enabled", original)


@pytest.mark.parametrize("allow_loss,scale", [("true", 6), ("false", 18)])
def test_decimal_precision_loss_setting(spark, allow_loss, scale):
    key = "spark.sql.decimalOperations.allowPrecisionLoss"
    original = spark.conf.get(key)
    try:
        spark.conf.set(key, allow_loss)
        frame = spark.sql("SELECT CAST(1 AS DECIMAL(38,18))/CAST(3 AS DECIMAL(38,18)) a")
        assert frame.schema["a"].dataType == DecimalType(38, scale)
        assert frame.first().a == Decimal("0." + "3" * scale)
    finally:
        spark.conf.set(key, original)


@pytest.mark.parametrize("sign", [1, -1])
def test_decimal_scale_reduction_rounds_half_up(spark, sign):
    frame = spark.sql(f"SELECT CAST({sign} AS DECIMAL(38,18))/CAST(128 AS DECIMAL(38,18)) a")
    assert frame.schema["a"].dataType == DecimalType(38, 6)
    assert frame.first().a == Decimal("0.007813") * sign


@pytest.mark.parametrize(
    "expression,precision,scale,value",
    [
        ("round(CAST(9.99 AS DECIMAL(3,2)),0)", 2, 0, "10"),
        ("round(CAST(99 AS DECIMAL(2,0)),-1)", 3, 0, "100"),
        ("round(CAST(99.99 AS DECIMAL(4,2)),1+1)", 5, 2, "99.99"),
        (
            (
                "CAST('99999999999999999999999999999999999999' AS DECIMAL(38,0)) / "
                "CAST('99999999999999999999999999999999999999' AS DECIMAL(38,0))"
            ),
            38,
            6,
            "1.000000",
        ),
        (
            (
                "CAST('0.99999999999999999999999999999999999999' AS DECIMAL(38,38)) * "
                "CAST('0.99999999999999999999999999999999999999' AS DECIMAL(38,38))"
            ),
            38,
            37,
            "1.0000000000000000000000000000000000000",
        ),
    ],
)
def test_decimal_rounding_and_wide_intermediates(spark, expression, precision, scale, value):
    # Check constant expressions with and without an input relation.
    for sql in [f"SELECT {expression} a", f"SELECT {expression} a FROM range(2)"]:
        frame = spark.sql(sql)
        assert frame.schema["a"].dataType == DecimalType(precision, scale)
        assert all(r.a == Decimal(value) for r in frame.collect())


@pytest.mark.parametrize("op,precision,scale", [("+", 16, 4), ("-", 16, 4), ("*", 31, 8), ("/", 35, 20)])
def test_decimal_untyped_null(spark, op, precision, scale):
    frame = spark.sql(f"SELECT CAST(1 AS DECIMAL(15,4)) {op} NULL a")
    assert frame.schema["a"].dataType == DecimalType(precision, scale)
    assert frame.first().a is None


def test_decimal_wide_array_intermediate(spark):
    frame = spark.sql(
        "SELECT CAST(v AS DECIMAL(38,38))*CAST(v AS DECIMAL(38,38)) a FROM VALUES "
        "('0.99999999999999999999999999999999999999'),"
        "('0.99999999999999999999999999999999999998') t(v)"
    )
    assert frame.schema["a"].dataType == DecimalType(38, 37)
    assert [r.a for r in frame.collect()] == [Decimal("1.0000000000000000000000000000000000000")] * 2
