//! Materialize CHAR padding keys before DataFusion decorrelates scalar aggregates.
//!
//! DataFusion only pulls equality predicates over aggregates when the local key
//! is a column (or a cast of one). Leaving rpad(column) in the predicate delays
//! decorrelation until projection pushdown has changed its qualified names.
use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{Column, DFSchema, Result, ScalarValue};
use datafusion_expr::{Expr, Filter, LogicalPlan, Operator, Projection};

#[derive(Debug)]
pub struct MaterializeCorrelatedCharKeys;

// Only total, deterministic padding and STRING casts introduced by CHAR
// analysis are admitted. Do not move arbitrary functions, casts, or volatile expressions
// ahead of a filter, where they could raise new errors or change evaluation.
fn is_local_key(expr: &Expr, schema: &DFSchema) -> bool {
    // CHAR analysis marks an explicit STRING cast with metadata so that parent
    // expressions do not inherit CHAR padding. Preserve that marker on the
    // computed key while avoiding an Alias(Cast(Column)) decorrelation barrier.
    if let Expr::Alias(alias) = expr {
        let string_marker = alias.metadata.as_ref().is_some_and(|metadata| {
            metadata
                .inner()
                .get("__CHAR_VARCHAR_TYPE_STRING")
                .is_some_and(|raw| raw == "string")
        });
        if string_marker
            && let Expr::Cast(cast) = alias.expr.as_ref()
            && matches!(
                cast.field.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            )
            && let Expr::Column(column) = cast.expr.as_ref()
        {
            return schema
                .qualified_field_from_column(column)
                .is_ok_and(|(_, field)| {
                    matches!(
                        field.data_type(),
                        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                    )
                });
        }
        return false;
    }
    let Expr::ScalarFunction(function) = expr else {
        return false;
    };
    function.func.name() == "rpad"
        && matches!(function.args.as_slice(), [Expr::Column(_), Expr::Literal(ScalarValue::Int64(Some(1..=10_485_760)), _), Expr::Literal(ScalarValue::Utf8(Some(fill)), _)] if fill == " ")
}

fn is_outer_only(expr: &Expr) -> bool {
    expr.contains_outer() && !expr.any_column_refs() && !expr.is_volatile()
}

impl OptimizerRule for MaterializeCorrelatedCharKeys {
    fn name(&self) -> &str {
        "materialize_correlated_char_keys"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Filter(filter) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let original: Vec<_> = filter
            .input
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        let mut keys: Vec<(Expr, String)> = Vec::new();
        let predicate = filter
            .predicate
            .clone()
            .transform_up(|expr| {
                let Expr::BinaryExpr(mut binary) = expr else {
                    return Ok(Transformed::no(expr));
                };
                if binary.op != Operator::Eq {
                    return Ok(Transformed::no(Expr::BinaryExpr(binary)));
                }
                let local = if is_local_key(&binary.left, filter.input.schema())
                    && is_outer_only(&binary.right)
                {
                    &mut binary.left
                } else if is_local_key(&binary.right, filter.input.schema())
                    && is_outer_only(&binary.left)
                {
                    &mut binary.right
                } else {
                    return Ok(Transformed::no(Expr::BinaryExpr(binary)));
                };
                let name =
                    if let Some((_, name)) = keys.iter().find(|(expr, _)| expr == local.as_ref()) {
                        name.clone()
                    } else {
                        let name = loop {
                            let candidate = config.alias_generator().next("__sail_char_key");
                            if !filter
                                .input
                                .schema()
                                .fields()
                                .iter()
                                .any(|f| f.name() == &candidate)
                            {
                                break candidate;
                            }
                        };
                        keys.push((local.as_ref().clone(), name.clone()));
                        name
                    };
                **local = Expr::Column(Column::new_unqualified(name));
                Ok(Transformed::yes(Expr::BinaryExpr(binary)))
            })
            .data()?;
        if keys.is_empty() {
            return Ok(Transformed::no(plan));
        }
        let mut computed = original.clone();
        computed.extend(keys.into_iter().map(|(expr, name)| match expr {
            // Keep the cast's metadata on the outermost alias: the physical
            // projection only reads that alias's metadata, not nested aliases.
            Expr::Alias(mut alias) => {
                alias.name = name;
                alias.relation = None;
                Expr::Alias(alias)
            }
            expr => expr.alias(name),
        }));
        let input =
            LogicalPlan::Projection(Projection::try_new(computed, Arc::clone(&filter.input))?);
        let filtered = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input))?);
        // Keep hidden keys out of the observable Filter schema, including raw
        // Spark Connect filter plans. Decorrelation can pull them through this
        // projection when it builds the aggregate's grouping and join keys.
        Ok(Transformed::yes(LogicalPlan::Projection(
            Projection::try_new(original, Arc::new(filtered))?,
        )))
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, clippy::unwrap_used)]
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::functions::unicode::expr_fn::rpad;
    use datafusion::optimizer::OptimizerContext;
    use datafusion_common::DFSchema;
    use datafusion_expr::{EmptyRelation, col, lit};

    use super::*;

    fn input() -> LogicalPlan {
        LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(
                DFSchema::try_from(Schema::new(vec![Field::new("c", DataType::Utf8, true)]))
                    .unwrap(),
            ),
        })
    }

    fn padded(expr: Expr) -> Expr {
        rpad(vec![expr, lit(8_i64), lit(" ")])
    }

    fn outer() -> Expr {
        Expr::OuterReferenceColumn(
            Arc::new(Field::new("outer_c", DataType::Utf8, true)),
            Column::new_unqualified("outer_c"),
        )
    }

    #[test]
    fn repeated_and_reversed_keys_are_shared_and_hidden() {
        let local = padded(col("c"));
        let predicate = local
            .clone()
            .eq(padded(outer()))
            .or(padded(outer()).eq(local));
        let original = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input())).unwrap());
        let result = MaterializeCorrelatedCharKeys
            .rewrite(original.clone(), &OptimizerContext::new())
            .unwrap();
        assert!(result.transformed);
        assert_eq!(result.data.schema(), original.schema());
        let LogicalPlan::Projection(restore) = result.data else {
            panic!("missing output projection")
        };
        let LogicalPlan::Filter(filter) = restore.input.as_ref() else {
            panic!("missing filter")
        };
        let LogicalPlan::Projection(compute) = filter.input.as_ref() else {
            panic!("missing key projection")
        };
        assert_eq!(
            compute.expr.len(),
            2,
            "one shared computed key, not one per occurrence"
        );
        assert!(filter.predicate.contains_outer());
        assert_eq!(filter.predicate.column_refs().len(), 1);
        assert!(
            filter
                .predicate
                .column_refs()
                .iter()
                .all(|c| c.name.starts_with("__sail_char_key"))
        );
        assert!(
            !MaterializeCorrelatedCharKeys
                .rewrite(
                    LogicalPlan::Filter(filter.clone()),
                    &OptimizerContext::new()
                )
                .unwrap()
                .transformed
        );
    }

    #[test]
    fn string_cast_marker_is_preserved_but_fallible_casts_are_not_lifted() {
        use std::collections::HashMap;

        use datafusion_expr::expr_fn::cast;
        let metadata = HashMap::from([(
            "__CHAR_VARCHAR_TYPE_STRING".to_string(),
            "string".to_string(),
        )]);
        for (data_type, expected) in [(DataType::Utf8, true), (DataType::Int64, false)] {
            let key = cast(col("c"), data_type)
                .alias_with_metadata("string_key", Some((&metadata).into()));
            let predicate = key.eq(outer());
            let plan = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input())).unwrap());
            let result = MaterializeCorrelatedCharKeys
                .rewrite(plan.clone(), &OptimizerContext::new())
                .unwrap();
            assert_eq!(result.transformed, expected);
            assert_eq!(result.data.schema(), plan.schema());
            if expected {
                let LogicalPlan::Projection(restore) = result.data else {
                    panic!("missing restore")
                };
                let LogicalPlan::Filter(filter) = restore.input.as_ref() else {
                    panic!("missing filter")
                };
                let LogicalPlan::Projection(compute) = filter.input.as_ref() else {
                    panic!("missing computed key")
                };
                let Expr::Alias(alias) = &compute.expr[1] else {
                    panic!("missing key alias")
                };
                assert!(
                    matches!(alias.expr.as_ref(), Expr::Cast(_)),
                    "do not nest the metadata alias"
                );
                assert_eq!(
                    alias
                        .metadata
                        .as_ref()
                        .unwrap()
                        .inner()
                        .get("__CHAR_VARCHAR_TYPE_STRING")
                        .unwrap(),
                    "string"
                );
            }
        }
    }

    #[test]
    fn ordinary_keys_inequalities_and_uncorrelated_padding_are_unchanged() {
        let predicates = [
            col("c").eq(outer()),
            padded(col("c")).gt(padded(outer())),
            padded(col("c")).eq(lit("x       ")),
        ];
        for predicate in predicates {
            let plan = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input())).unwrap());
            let result = MaterializeCorrelatedCharKeys
                .rewrite(plan.clone(), &OptimizerContext::new())
                .unwrap();
            assert!(!result.transformed);
            assert_eq!(result.data, plan);
        }
    }
}
