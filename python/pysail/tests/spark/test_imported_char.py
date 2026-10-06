"""Spark-compatible comparisons for imported, already padded Delta CHAR data."""

import json

import pyarrow as pa
import pyarrow.parquet as pq


def test_imported_char_comparisons(spark, tmp_path):
    # Write a minimal portable Delta table without requiring a second engine.
    raw = "__CHAR_VARCHAR_TYPE_STRING"
    fields = [
        {"name": "id", "type": "integer", "nullable": True, "metadata": {}},
        {"name": "c", "type": "string", "nullable": True, "metadata": {raw: "char(4)"}},
        {"name": "d", "type": "string", "nullable": True, "metadata": {raw: "char(8)"}},
        {"name": "s", "type": "string", "nullable": True, "metadata": {}},
    ]
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("c", pa.string()),
            pa.field("d", pa.string()),
            pa.field("s", pa.string()),
        ]
    )
    values = ["x", "", None, "🧱é"]
    data = pa.Table.from_pylist(
        [
            {"id": i, "c": v.ljust(4) if v is not None else None, "d": v.ljust(8) if v is not None else None, "s": v}
            for i, v in enumerate(values)
        ],
        schema=schema,
    )
    file = tmp_path / "data.parquet"
    pq.write_table(data, file)
    log = tmp_path / "_delta_log"
    log.mkdir()
    actions = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": "char-comparison-test",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps({"type": "struct", "fields": fields}),
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
    (log / "00000000000000000000.json").write_text("\n".join(map(json.dumps, actions)))
    spark.sql(f"CREATE TABLE imported_char USING delta LOCATION '{tmp_path}'").collect()
    try:

        def rows(sql):
            return [tuple(r) for r in spark.sql(sql).collect()]

        assert rows(
            "SELECT id,c='x',c=d,c=s,length(c),c LIKE 'x',CAST(c AS STRING)='x' FROM imported_char ORDER BY id"
        ) == [
            (0, True, True, False, 4, False, False),
            (1, False, True, False, 4, False, False),
            (2, None, None, None, None, None, None),
            (3, False, True, False, 4, False, False),
        ]
        assert rows(
            "SELECT id,c=concat('x',''),c IN ('x','y'),c IN ('x',NULL),"
            "c <=> d,c='🧱é',c='x     ' FROM imported_char ORDER BY id"
        ) == [
            (0, True, True, None, True, False, True),
            (1, False, False, None, True, False, False),
            (2, None, None, None, True, None, None),
            (3, False, False, None, True, True, False),
        ]
        assert rows("SELECT a.id FROM imported_char a JOIN imported_char b ON a.c=b.d ORDER BY a.id") == [
            (0,),
            (1,),
            (3,),
        ]
        assert rows("SELECT id FROM imported_char WHERE c IN (SELECT d FROM imported_char)") == []
        assert rows("SELECT x='x' FROM (SELECT c AS x,id FROM imported_char) ORDER BY id") == [
            (True,),
            (False,),
            (None,),
            (False,),
        ]
        assert rows("SELECT x='x' FROM (SELECT CAST(c AS STRING) AS x,id FROM imported_char) ORDER BY id") == [
            (False,),
            (False,),
            (None,),
            (False,),
        ]
    finally:
        spark.sql("DROP TABLE imported_char").collect()
