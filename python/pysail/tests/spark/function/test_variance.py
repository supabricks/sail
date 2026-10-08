"""Central moments follow Spark arithmetic and statistical NULL semantics."""

import math

import pytest


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ("(1),(2),(3)", [1.0, 2.0 / 3.0, 1.0, math.sqrt(2.0 / 3.0)]),
        ("(1),(NULL),(2),(3)", [1.0, 2.0 / 3.0, 1.0, math.sqrt(2.0 / 3.0)]),
        ("(7)", [None, 0.0, None, 0.0]),
        ("(7),(7),(7)", [0.0, 0.0, 0.0, 0.0]),
        ("(CAST(NULL AS DOUBLE)),(NULL)", [None, None, None, None]),
        ("(CAST('NaN' AS DOUBLE)),(1.)", [math.nan] * 4),
        ("(CAST('Infinity' AS DOUBLE))", [None, math.nan, None, math.nan]),
        ("(CAST('-Infinity' AS DOUBLE))", [None, math.nan, None, math.nan]),
    ],
)
def test_variance_edge_cases(spark, values, expected):
    row = spark.sql(
        "SELECT var_samp(v), var_pop(v), stddev_samp(v), stddev_pop(v) " + f"FROM VALUES {values} t(v)"
    ).first()
    for actual, value in zip(row, expected):
        if value is not None and math.isnan(value):
            assert math.isnan(actual)
        else:
            assert actual == value


def test_variance_empty_and_distinct(spark):
    row = spark.sql(
        "SELECT var_samp(v), var_pop(v), stddev_samp(v), stddev_pop(v) FROM VALUES (1.) t(v) WHERE false"
    ).first()
    assert list(row) == [None] * 4
    row = spark.sql(
        "SELECT var_samp(DISTINCT v), stddev_samp(DISTINCT v) FROM VALUES (1),(2),(2),(3),(NULL) t(v)"
    ).first()
    assert list(row) == [1.0, 1.0]


def test_variance_group_filter_and_aliases(spark):
    rows = spark.sql(
        "SELECT k, var_samp(v), stddev_samp(v), var_pop(v) FILTER (WHERE v<3) "
        "FROM VALUES (1,1),(1,2),(1,3),(2,8),(2,NULL) t(k,v) GROUP BY k ORDER BY k"
    ).collect()
    assert [list(row) for row in rows] == [[1, 1.0, 1.0, 0.25], [2, None, None, None]]
    row = spark.sql(
        "SELECT std(v), stddev(v), stddev_samp(v), variance(v), var_samp(v) FROM VALUES (1),(2),(3) t(v)"
    ).first()
    assert list(row) == [1.0] * 5


def test_variance_sliding_window_recomputes_spark_moments(spark):
    rows = spark.sql(
        "SELECT id, var_samp(v) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) "
        "FROM VALUES (1,10),(2,404),(3,13),(4,814),(5,NULL) t(id,v) ORDER BY id"
    ).collect()
    assert [row[1] for row in rows] == [None, 77618.0, 51354.33333333333, 160430.3333333333, 320800.5]
