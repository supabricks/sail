"""Correlated scalar aggregates retain CHAR semantics through internal aliases."""

import json
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType


@pytest.fixture(params=["table", "view"])
def correlated_char_table(spark, tmp_path, request):
    raw = "__CHAR_VARCHAR_TYPE_STRING"
    schema = StructType(
        [
            StructField("id", IntegerType()),
            StructField("c", StringType(), metadata={raw: "char(4)"}),
            StructField("d", StringType(), metadata={raw: "char(8)"}),
            StructField("v", DecimalType(7, 2)),
        ]
    )
    values = [(1, "x", "1"), (2, "x", "3"), (3, "y", "5"), (4, None, "7"), (5, "🧱é", "9"), (6, "", "11")]
    rows = [
        (i, c.ljust(4) if c is not None else None, c.ljust(8) if c is not None else None, Decimal(v))
        for i, c, v in values
    ]
    source = "correlated_char_source" if request.param == "view" else "correlated_char"
    # The independent JVM qualification can use a native DataFrame with the
    # identical metadata/values; the product path always imports real Delta.
    dataframe = request.config.getoption("--correlated-source", default="delta") == "dataframe"
    if dataframe:
        spark.createDataFrame(rows, schema).createOrReplaceTempView(source)
    else:
        arrow = pa.schema(
            [
                pa.field("id", pa.int32()),
                pa.field("c", pa.string()),
                pa.field("d", pa.string()),
                pa.field("v", pa.decimal128(7, 2)),
            ]
        )
        data = pa.Table.from_pylist([dict(zip(schema.names, row)) for row in rows], schema=arrow)
        file = tmp_path / "data.parquet"
        pq.write_table(data, file)
        log = tmp_path / "_delta_log"
        log.mkdir()
        actions = [
            {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
            {
                "metaData": {
                    "id": "correlated-char",
                    "format": {"provider": "parquet", "options": {}},
                    "schemaString": json.dumps(schema.jsonValue()),
                    "partitionColumns": [],
                    "configuration": {},
                }
            },
            {
                "add": {
                    "path": file.name,
                    "size": file.stat().st_size,
                    "modificationTime": 0,
                    "partitionValues": {},
                    "dataChange": True,
                }
            },
        ]
        (log / "00000000000000000000.json").write_text("\n".join(map(json.dumps, actions)) + "\n")
        spark.sql(f"CREATE TABLE {source} USING delta LOCATION '{tmp_path}'").collect()
    if request.param == "view":
        temporal = "" if dataframe else " VERSION AS OF 0"
        temporary = "TEMPORARY " if dataframe else ""
        spark.sql(f"CREATE {temporary}VIEW correlated_char AS SELECT * FROM {source}{temporal}").collect()
    try:
        yield
    finally:
        if request.param == "view":
            if dataframe:
                spark.catalog.dropTempView("correlated_char")
            else:
                spark.sql("DROP VIEW correlated_char").collect()
        if dataframe:
            spark.catalog.dropTempView(source)
        else:
            spark.sql(f"DROP TABLE {source}").collect()


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        ("a.v > (SELECT avg(b.v) FROM correlated_char b WHERE b.c=a.c)", [2]),
        ("a.v > (SELECT avg(b.v) FROM correlated_char b WHERE a.c=b.c)", [2]),
        ("a.v > (SELECT avg(b.v) FROM correlated_char b WHERE b.c=a.d)", [2]),
        ("(SELECT count(*) FROM correlated_char b WHERE b.c=a.c)>0", [1, 2, 3, 5, 6]),
        ("(SELECT count(*) FROM correlated_char b WHERE b.c=a.c)=0", [4]),
        ("(SELECT count(*) FROM correlated_char b WHERE b.c=a.c AND b.id<0)=0", [1, 2, 3, 4, 5, 6]),
        ("EXISTS (SELECT 1 FROM correlated_char b WHERE b.c=a.c)", [1, 2, 3, 5, 6]),
        ("NOT EXISTS (SELECT 1 FROM correlated_char b WHERE b.c=a.c)", [4]),
        ("(SELECT count(*) FROM correlated_char b WHERE (b.c=a.c AND b.id=1) OR (b.c=a.c AND b.id=3))>0", [1, 2, 3]),
        ("(SELECT count(*) FROM correlated_char b WHERE b.c=a.c AND b.d=a.d)>0", [1, 2, 3, 5, 6]),
        ("(SELECT count(*) FROM correlated_char b WHERE CAST(b.c AS STRING)=CAST(a.d AS STRING))>0", []),
    ],
)
def test_correlated_char_predicates(spark, correlated_char_table, predicate, expected):
    result = spark.sql(f"SELECT a.id FROM correlated_char a WHERE {predicate} ORDER BY a.id")
    assert [row.id for row in result.collect()] == expected
    assert result.columns == ["id"]


def test_correlated_char_projection_keeps_count_empty_and_output_schema(spark, correlated_char_table):
    result = spark.sql(
        "SELECT a.*, (SELECT count(*) FROM correlated_char b WHERE b.c=a.d) n FROM correlated_char a ORDER BY a.id"
    )
    assert result.columns == ["id", "c", "d", "v", "n"]
    assert [row.n for row in result.collect()] == [2, 2, 1, 0, 1, 1]


def test_correlated_char_grouped_cte(spark, correlated_char_table):
    result = spark.sql(
        "WITH grouped AS (SELECT c, sum(v) total FROM correlated_char GROUP BY c) "
        "SELECT a.c FROM grouped a WHERE a.total > "
        "(SELECT avg(b.total) FROM grouped b WHERE b.c=a.c) ORDER BY a.c"
    )
    assert result.collect() == []
