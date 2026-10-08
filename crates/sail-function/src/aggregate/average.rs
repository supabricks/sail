//! Spark decimal AVG finalization. Keep DataFusion's accumulation and state layout,
//! but round the final quotient HALF_UP (ties away from zero), not toward zero.
use std::sync::{Arc, LazyLock};

use arrow::array::{ArrayRef, AsArray, BooleanArray, Decimal128Array};
use arrow::datatypes::{DataType, Decimal128Type, FieldRef, UInt64Type};
use datafusion::functions_aggregate::average::Avg;
use datafusion_common::{Result, ScalarValue, exec_datafusion_err, internal_err};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::{AggregateUDF, AggregateUDFImpl, ReversedUDAF};
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_expr_common::groups_accumulator::{EmitTo, GroupsAccumulator};
use datafusion_expr_common::signature::Signature;
use datafusion_functions_aggregate_common::aggregate::sum_distinct::DistinctSumAccumulator;

pub fn avg_udaf() -> Arc<AggregateUDF> {
    static AVG: LazyLock<Arc<AggregateUDF>> =
        LazyLock::new(|| Arc::new(AggregateUDF::from(SparkAvg::default())));
    Arc::clone(&AVG)
}

#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub struct SparkAvg(Avg);

impl AggregateUDFImpl for SparkAvg {
    fn name(&self) -> &str {
        "avg"
    }
    fn signature(&self) -> &Signature {
        self.0.signature()
    }
    fn aliases(&self) -> &[String] {
        self.0.aliases()
    }
    fn return_type(&self, args: &[DataType]) -> Result<DataType> {
        self.0.return_type(args)
    }
    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        self.0.state_fields(args)
    }
    fn reverse_expr(&self) -> ReversedUDAF {
        ReversedUDAF::Identical
    }

    fn accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let Some(rounding) = DecimalAverage::from_args(&args) else {
            return self.0.accumulator(args);
        };
        let inner = if args.is_distinct {
            DecimalState::Distinct(DistinctSumAccumulator::new(&DataType::Decimal128(
                38,
                rounding.sum_scale,
            )))
        } else {
            DecimalState::Plain(self.0.accumulator(args)?)
        };
        Ok(Box::new(DecimalAccumulator { inner, rounding }))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        self.0.groups_accumulator_supported(args)
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        let rounding = DecimalAverage::from_args(&args);
        let inner = self.0.create_groups_accumulator(args)?;
        match rounding {
            Some(rounding) => Ok(Box::new(DecimalGroupsAccumulator { inner, rounding })),
            None => Ok(inner),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DecimalAverage {
    sum_scale: i8,
    precision: u8,
    scale: i8,
}

impl DecimalAverage {
    fn from_args(args: &AccumulatorArgs) -> Option<Self> {
        match (args.expr_fields[0].data_type(), args.return_type()) {
            (DataType::Decimal128(_, sum_scale), DataType::Decimal128(precision, scale)) => {
                Some(Self {
                    sum_scale: *sum_scale,
                    precision: *precision,
                    scale: *scale,
                })
            }
            _ => None,
        }
    }

    fn value(self, sum: Option<i128>, count: u64) -> Result<Option<i128>> {
        let Some(sum) = sum.filter(|_| count != 0) else {
            return Ok(None);
        };
        let overflow = || exec_datafusion_err!("Arithmetic overflow in decimal AVG");
        let shift = u32::try_from(i16::from(self.scale) - i16::from(self.sum_scale))
            .map_err(|_| overflow())?;
        // Preserve the existing checked rescaling/overflow contract. No floating
        // point intermediate or rounding of partial averages is involved.
        let scaled = sum
            .checked_mul(10_i128.checked_pow(shift).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        let count = i128::from(count);
        let quotient = scaled / count;
        let remainder = scaled % count;
        let rounded = if remainder.abs() * 2 >= count {
            quotient.checked_add(scaled.signum()).ok_or_else(overflow)?
        } else {
            quotient
        };
        let limit = 10_i128
            .checked_pow(u32::from(self.precision))
            .ok_or_else(overflow)?;
        if rounded <= -limit || rounded >= limit {
            return Err(overflow());
        }
        Ok(Some(rounded))
    }

    fn scalar(self, sum: Option<i128>, count: u64) -> Result<ScalarValue> {
        Ok(ScalarValue::Decimal128(
            self.value(sum, count)?,
            self.precision,
            self.scale,
        ))
    }
}

#[derive(Debug)]
enum DecimalState {
    Plain(Box<dyn Accumulator>),
    Distinct(DistinctSumAccumulator<Decimal128Type>),
}

impl DecimalState {
    fn accumulator(&mut self) -> &mut dyn Accumulator {
        match self {
            Self::Plain(inner) => inner.as_mut(),
            Self::Distinct(inner) => inner,
        }
    }
}

#[derive(Debug)]
struct DecimalAccumulator {
    inner: DecimalState,
    rounding: DecimalAverage,
}

impl Accumulator for DecimalAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.inner.accumulator().update_batch(values)
    }
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.inner.accumulator().merge_batch(states)
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.inner.accumulator().state()
    }
    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.inner.accumulator().retract_batch(values)
    }
    fn supports_retract_batch(&self) -> bool {
        match &self.inner {
            DecimalState::Plain(inner) => inner.supports_retract_batch(),
            DecimalState::Distinct(_) => false,
        }
    }
    fn size(&self) -> usize {
        size_of_val(self)
            + match &self.inner {
                DecimalState::Plain(inner) => inner.size(),
                DecimalState::Distinct(inner) => inner.size(),
            }
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        match &mut self.inner {
            DecimalState::Plain(inner) => {
                let state = inner.state()?;
                match state.as_slice() {
                    [
                        ScalarValue::UInt64(Some(count)),
                        ScalarValue::Decimal128(sum, _, _),
                    ] => self.rounding.scalar(*sum, *count),
                    _ => internal_err!("Unexpected decimal AVG state"),
                }
            }
            DecimalState::Distinct(inner) => {
                let count = inner.distinct_count() as u64;
                match inner.evaluate()? {
                    ScalarValue::Decimal128(sum, _, _) => self.rounding.scalar(sum, count),
                    _ => internal_err!("Unexpected decimal AVG DISTINCT sum"),
                }
            }
        }
    }
}

struct DecimalGroupsAccumulator {
    inner: Box<dyn GroupsAccumulator>,
    rounding: DecimalAverage,
}

impl GroupsAccumulator for DecimalGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        indices: &[usize],
        filter: Option<&BooleanArray>,
        groups: usize,
    ) -> Result<()> {
        self.inner.update_batch(values, indices, filter, groups)
    }
    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        indices: &[usize],
        filter: Option<&BooleanArray>,
        groups: usize,
    ) -> Result<()> {
        self.inner.merge_batch(values, indices, filter, groups)
    }
    fn state(&mut self, emit: EmitTo) -> Result<Vec<ArrayRef>> {
        self.inner.state(emit)
    }
    fn size(&self) -> usize {
        size_of_val(self) + self.inner.size()
    }
    fn supports_convert_to_state(&self) -> bool {
        self.inner.supports_convert_to_state()
    }
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        self.inner.convert_to_state(values, filter)
    }
    fn evaluate(&mut self, emit: EmitTo) -> Result<ArrayRef> {
        // Taking state preserves EmitTo's removal semantics and the vectorized
        // accumulation path; the sum/count arrays are moved, not rebuilt.
        let state = self.inner.state(emit)?;
        let counts = state[0].as_primitive::<UInt64Type>();
        let sums = state[1].as_primitive::<Decimal128Type>();
        let values = sums
            .iter()
            .zip(counts.iter())
            .map(|(sum, count)| self.rounding.value(sum, count.unwrap_or(0)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(
            Decimal128Array::from(values)
                .with_precision_and_scale(self.rounding.precision, self.rounding.scale)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::Array;
    use arrow::datatypes::{Field, Schema};

    use super::*;

    fn args<'a>(schema: &'a Schema, fields: &'a [FieldRef], distinct: bool) -> AccumulatorArgs<'a> {
        AccumulatorArgs {
            return_field: Arc::new(Field::new("avg", DataType::Decimal128(22, 6), true)),
            schema,
            ignore_nulls: false,
            order_bys: &[],
            is_reversed: false,
            name: "avg",
            is_distinct: distinct,
            exprs: &[],
            expr_fields: fields,
        }
    }
    fn values(values: Vec<Option<i128>>) -> Result<ArrayRef> {
        Ok(Arc::new(
            Decimal128Array::from(values).with_precision_and_scale(18, 2)?,
        ))
    }
    fn state_arrays(acc: &mut dyn Accumulator) -> Result<Vec<ArrayRef>> {
        acc.state()?.iter().map(|v| v.to_array_of_size(1)).collect()
    }

    #[test]
    fn decimal_avg_rounding_and_bounds() -> Result<()> {
        let round = DecimalAverage {
            sum_scale: 2,
            precision: 22,
            scale: 6,
        };
        for (sum, count, expected) in [
            (2, 3, 6667),
            (1, 3, 3333),
            (1, 32, 313),
            (3, 32, 938),
            (0, 3, 0),
            (3, 3, 10000),
        ] {
            for sign in [1, -1] {
                assert_eq!(round.value(Some(sign * sum), count)?, Some(sign * expected));
            }
        }
        assert_eq!(round.value(None, 3)?, None);
        assert_eq!(round.value(Some(1), 0)?, None);
        assert!(round.value(Some(i128::MAX), 1).is_err());
        let capped = DecimalAverage {
            sum_scale: 38,
            precision: 38,
            scale: 38,
        };
        assert_eq!(capped.value(Some(1), 2)?, Some(1));
        assert_eq!(capped.value(Some(-1), 2)?, Some(-1));
        assert_eq!(
            capped.value(Some(10_i128.pow(38) - 1), 1)?,
            Some(10_i128.pow(38) - 1)
        );
        let small = DecimalAverage {
            sum_scale: 2,
            precision: 2,
            scale: 2,
        };
        assert_eq!(small.value(Some(198), 2)?, Some(99));
        assert!(small.value(Some(199), 2).is_err()); // rounding itself overflows
        Ok(())
    }

    #[test]
    fn decimal_avg_partial_merge_and_retraction() -> Result<()> {
        let schema = Schema::empty();
        let fields = [Arc::new(Field::new("v", DataType::Decimal128(18, 2), true))];
        let udf = SparkAvg::default();
        let mut first = udf.accumulator(args(&schema, &fields, false))?;
        let mut second = udf.accumulator(args(&schema, &fields, false))?;
        assert_eq!(first.evaluate()?, ScalarValue::Decimal128(None, 22, 6));
        first.update_batch(&[values(vec![Some(2), None])?])?;
        second.update_batch(&[values(vec![Some(0), Some(0)])?])?;
        let mut merged = udf.create_sliding_accumulator(args(&schema, &fields, false))?;
        merged.merge_batch(&state_arrays(first.as_mut())?)?;
        merged.merge_batch(&state_arrays(second.as_mut())?)?;
        assert_eq!(
            merged.evaluate()?,
            ScalarValue::Decimal128(Some(6667), 22, 6)
        );
        assert_eq!(
            merged.evaluate()?,
            ScalarValue::Decimal128(Some(6667), 22, 6)
        );
        assert!(merged.supports_retract_batch());
        merged.retract_batch(&[values(vec![Some(2), Some(0), Some(0)])?])?;
        assert_eq!(merged.evaluate()?, ScalarValue::Decimal128(None, 22, 6));
        merged.update_batch(&[values(vec![Some(-2), Some(0), Some(0), None])?])?;
        assert_eq!(
            merged.evaluate()?,
            ScalarValue::Decimal128(Some(-6667), 22, 6)
        );
        Ok(())
    }

    #[test]
    fn decimal_avg_distinct_merge_deduplicates_before_rounding() -> Result<()> {
        let schema = Schema::empty();
        let fields = [Arc::new(Field::new("v", DataType::Decimal128(18, 2), true))];
        let udf = SparkAvg::default();
        let mut first = udf.accumulator(args(&schema, &fields, true))?;
        let mut second = udf.accumulator(args(&schema, &fields, true))?;
        first.update_batch(&[values(vec![None, None])?])?;
        assert_eq!(first.evaluate()?, ScalarValue::Decimal128(None, 22, 6));
        first.update_batch(&[values(vec![Some(1), Some(1), Some(0)])?])?;
        second.update_batch(&[values(vec![Some(0), Some(4), None])?])?;
        let mut merged = udf.accumulator(args(&schema, &fields, true))?;
        merged.merge_batch(&state_arrays(first.as_mut())?)?;
        merged.merge_batch(&state_arrays(second.as_mut())?)?;
        assert_eq!(
            merged.evaluate()?,
            ScalarValue::Decimal128(Some(16667), 22, 6)
        );
        assert!(!udf.groups_accumulator_supported(args(&schema, &fields, true)));
        Ok(())
    }

    #[test]
    fn decimal_avg_group_filter_emit_and_convert() -> Result<()> {
        let schema = Schema::empty();
        let fields = [Arc::new(Field::new("v", DataType::Decimal128(18, 2), true))];
        let udf = SparkAvg::default();
        let mut groups = udf.create_groups_accumulator(args(&schema, &fields, false))?;
        let filter = BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            None,
            Some(true),
        ]);
        groups.update_batch(
            &[values(vec![
                Some(2),
                Some(0),
                Some(0),
                Some(99),
                Some(99),
                None,
            ])?],
            &[0, 0, 0, 0, 0, 1],
            Some(&filter),
            3,
        )?;
        assert_eq!(
            groups
                .evaluate(EmitTo::First(1))?
                .as_primitive::<Decimal128Type>()
                .value(0),
            6667
        );
        let rest = groups.evaluate(EmitTo::All)?;
        assert_eq!(rest.len(), 2);
        assert_eq!(rest.null_count(), 2);
        assert!(groups.supports_convert_to_state());
        let raw = [values(vec![Some(-2), Some(0), Some(0), Some(99)])?];
        let filter = BooleanArray::from(vec![true, true, true, false]);
        let state = groups.convert_to_state(&raw, Some(&filter))?;
        groups.merge_batch(&state, &[0, 0, 0, 0], None, 1)?;
        let state = groups.state(EmitTo::All)?;
        let mut final_groups = udf.create_groups_accumulator(args(&schema, &fields, false))?;
        final_groups.merge_batch(&state, &[0], None, 1)?;
        assert_eq!(
            final_groups
                .evaluate(EmitTo::All)?
                .as_primitive::<Decimal128Type>()
                .value(0),
            -6667
        );
        Ok(())
    }
}
