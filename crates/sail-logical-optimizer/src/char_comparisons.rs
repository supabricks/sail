//! Spark CHAR comparison semantics for imported, already-padded columns.
//!
//! Delta stores CHAR as strings plus Spark's raw-type field metadata. Keep the
//! stored bytes (including spaces) and ordinary STRING/LIKE/cast behavior. Only
//! CHAR/CHAR and CHAR/string-literal comparisons require length equalization.
use std::collections::HashMap;
use std::ops::Not;

use datafusion::functions::unicode::expr_fn::rpad;
use datafusion::optimizer::AnalyzerRule;
use datafusion::optimizer::simplify_expressions::ExprSimplifier;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{DFSchema, Result, ScalarValue};
use datafusion_expr::Volatility;
use datafusion_expr::expr_rewriter::NamePreserver;
use datafusion_expr::simplify::SimplifyContextBuilder;
use datafusion_expr::utils::merge_schema;
use datafusion_expr::{BinaryExpr, Expr, LogicalPlan, Operator, lit};

const RAW_TYPE: &str = "__CHAR_VARCHAR_TYPE_STRING";

#[derive(Debug)]
pub struct CharComparisons;

fn field_width(field: &datafusion::arrow::datatypes::Field) -> Option<i64> {
    use datafusion::arrow::datatypes::DataType;
    if !matches!(
        field.data_type(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    ) {
        return None;
    }
    let raw = field.metadata().get(RAW_TYPE)?;
    let n = raw.strip_prefix("char(")?.strip_suffix(')')?.parse().ok()?;
    (n > 0 && n <= 10_485_760).then_some(n)
}

fn char_width(expr: &Expr, schema: &DFSchema) -> Option<i64> {
    match expr {
        Expr::Column(c) => field_width(schema.qualified_field_from_column(c).ok()?.1),
        Expr::OuterReferenceColumn(f, _) => field_width(f),
        Expr::Alias(a) => char_width(&a.expr, schema),
        // An explicit cast/function result is STRING, not CHAR.
        _ => None,
    }
}

fn literal_width(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Literal(
            ScalarValue::Utf8(Some(s))
            | ScalarValue::LargeUtf8(Some(s))
            | ScalarValue::Utf8View(Some(s)),
            _,
        ) => Some(s.chars().count() as i64),
        Expr::Literal(
            ScalarValue::Utf8(None) | ScalarValue::LargeUtf8(None) | ScalarValue::Utf8View(None),
            _,
        ) => Some(0),
        _ => None,
    }
}

// Spark comparisons accept foldable string expressions. Never fold a column
// cast: it deliberately removes CHAR.
fn foldable(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(_, _) => true,
        Expr::Cast(c) => foldable(&c.expr),
        Expr::ScalarFunction(f) => {
            f.func.signature().volatility == Volatility::Immutable && f.args.iter().all(foldable)
        }
        _ => false,
    }
}

fn fold_constant(expr: &Expr) -> Expr {
    if foldable(expr) {
        let context = SimplifyContextBuilder::default().build();
        if let Ok(result) = ExprSimplifier::new(context).simplify(expr.clone()) {
            if literal_width(&result).is_some() {
                return result;
            }
        }
    }
    expr.clone()
}

fn pair_width(left: &Expr, right: &Expr, schema: &DFSchema) -> Option<i64> {
    match (char_width(left, schema), char_width(right, schema)) {
        (Some(l), Some(r)) => Some(l.max(r)),
        (Some(l), None) => literal_width(right).map(|r| l.max(r)),
        (None, Some(r)) => literal_width(left).map(|l| l.max(r)),
        _ => None,
    }
}

fn pad(expr: Expr, width: i64) -> Expr {
    rpad(vec![expr, lit(width), lit(" ")])
}

fn comparison(left: Expr, op: Operator, right: Expr, schema: &DFSchema) -> Option<Expr> {
    let width = pair_width(&left, &right, schema)?;
    Some(Expr::BinaryExpr(BinaryExpr::new(
        Box::new(pad(left, width)),
        op,
        Box::new(pad(right, width)),
    )))
}

fn rewrite(expr: Expr, schema: &DFSchema) -> Result<Transformed<Expr>> {
    // DataFusion inherits source metadata through casts. Spark drops CHAR
    // semantics there, including when a parent query refers to a cast alias.
    let cast_source = match &expr {
        Expr::Cast(c) => Some(c.expr.as_ref()),
        Expr::TryCast(c) => Some(c.expr.as_ref()),
        _ => None,
    };
    if cast_source.and_then(|e| char_width(e, schema)).is_some() {
        let name = expr.schema_name().to_string();
        let metadata = HashMap::from([(RAW_TYPE.to_string(), "string".to_string())]);
        return Ok(Transformed::yes(
            expr.alias_with_metadata(name, Some((&metadata).into())),
        ));
    }
    match &expr {
        Expr::BinaryExpr(b)
            if matches!(
                b.op,
                Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq
                    | Operator::IsDistinctFrom
                    | Operator::IsNotDistinctFrom
            ) =>
        {
            let left = if char_width(&b.right, schema).is_some() {
                fold_constant(&b.left)
            } else {
                *b.left.clone()
            };
            let right = if char_width(&b.left, schema).is_some() {
                fold_constant(&b.right)
            } else {
                *b.right.clone()
            };
            if let Some(e) = comparison(left, b.op, right, schema) {
                return Ok(Transformed::yes(e));
            }
        }
        Expr::InList(list) => {
            // Match Spark's analysis-time behavior: pad homogeneous CHAR
            // attributes or resolved foldable strings. An untyped NULL keeps
            // an IN list unresolved at this stage, so it receives no padding.
            if let Some(width) = char_width(&list.expr, schema) {
                let folded: Vec<_> = list.list.iter().map(fold_constant).collect();
                let widths: Option<Vec<_>> = folded.iter().map(literal_width).collect();
                let widths =
                    widths.or_else(|| list.list.iter().map(|e| char_width(e, schema)).collect());
                if let Some(widths) = widths {
                    let width = widths.into_iter().fold(width, i64::max);
                    let mut result = list.clone();
                    result.expr = Box::new(pad(*result.expr, width));
                    result.list = folded.into_iter().map(|e| pad(e, width)).collect();
                    return Ok(Transformed::yes(Expr::InList(result)));
                }
            }
        }
        Expr::Between(b) => {
            let low = comparison(*b.expr.clone(), Operator::GtEq, *b.low.clone(), schema);
            let high = comparison(*b.expr.clone(), Operator::LtEq, *b.high.clone(), schema);
            if low.is_some() || high.is_some() {
                let result = low
                    .unwrap_or_else(|| b.expr.as_ref().clone().gt_eq(*b.low.clone()))
                    .and(high.unwrap_or_else(|| b.expr.as_ref().clone().lt_eq(*b.high.clone())));
                return Ok(Transformed::yes(if b.negated {
                    result.not()
                } else {
                    result
                }));
            }
        }
        _ => {}
    }
    Ok(Transformed::no(expr))
}

impl AnalyzerRule for CharComparisons {
    fn analyze(&self, plan: LogicalPlan, _: &ConfigOptions) -> Result<LogicalPlan> {
        Ok(plan
            .transform_up_with_subqueries(|plan| {
                let plan = plan.recompute_schema()?;
                let mut schema = merge_schema(&plan.inputs());
                if let LogicalPlan::TableScan(scan) = &plan {
                    schema.merge(&DFSchema::try_from_qualified_schema(
                        scan.table_name.clone(),
                        &scan.source.schema(),
                    )?);
                }
                let names = NamePreserver::new(&plan);
                let mut transformed = plan.map_expressions(|expr| {
                    let name = names.save(&expr);
                    Ok(expr
                        .transform_up(|e| rewrite(e, &schema))?
                        .update_data(|e| name.restore(e)))
                })?;
                // Equijoin keys are pairs, not BinaryExpr nodes in map_expressions.
                if let LogicalPlan::Join(join) = &mut transformed.data {
                    for (left, right) in &mut join.on {
                        if let Some(width) = pair_width(left, right, &schema) {
                            *left = pad(left.clone(), width);
                            *right = pad(right.clone(), width);
                            transformed.transformed = true;
                        }
                    }
                }
                transformed.data = transformed.data.recompute_schema()?;
                Ok(transformed)
            })?
            .data)
    }

    fn name(&self) -> &str {
        "imported_char_comparisons"
    }
}
