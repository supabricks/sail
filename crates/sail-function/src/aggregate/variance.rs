//! Spark's second central moment update/merge arithmetic.
//!
//! Preserve operation order from Spark 4.2 CentralMomentAgg, including merges
//! into a zero buffer. Algebraically equivalent Welford expressions have different
//! floating-point rounding. Input and partition order can still affect results.
use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array};
use arrow::datatypes::{DataType, Field, FieldRef, Float64Type};
use datafusion_common::cast::as_float64_array;
use datafusion_common::{Result, ScalarValue, internal_err, not_impl_err};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, EmitTo, GroupsAccumulator, Signature, Volatility,
};
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::accumulate::accumulate;
use datafusion_functions_aggregate_common::utils::GenericDistinctBuffer;

macro_rules! function {
    ($factory:ident, $name:literal, $sample:literal, $sqrt:literal) => {
        pub fn $factory() -> Arc<AggregateUDF> {
            static FUNCTION: LazyLock<Arc<AggregateUDF>> = LazyLock::new(|| {
                Arc::new(AggregateUDF::from(SparkVariance {
                    name: $name,
                    kind: Kind {
                        sample: $sample,
                        sqrt: $sqrt,
                    },
                    signature: Signature::exact(vec![DataType::Float64], Volatility::Immutable),
                }))
            });
            Arc::clone(&FUNCTION)
        }
    };
}
function!(var_samp_udaf, "var_samp", true, false);
function!(var_pop_udaf, "var_pop", false, false);
function!(stddev_udaf, "stddev_samp", true, true);
function!(stddev_pop_udaf, "stddev_pop", false, true);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Kind {
    sample: bool,
    sqrt: bool,
}
impl Kind {
    fn evaluate(self, m: Moments) -> Option<f64> {
        if m.n == 0.0 || (self.sample && m.n == 1.0) {
            return None;
        }
        let variance = m.m2 / (m.n - if self.sample { 1.0 } else { 0.0 });
        Some(if self.sqrt { variance.sqrt() } else { variance })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparkVariance {
    name: &'static str,
    kind: Kind,
    signature: Signature,
}
impl AggregateUDFImpl for SparkVariance {
    fn name(&self) -> &str {
        self.name
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }
    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        if args.is_distinct {
            return Ok(vec![Arc::new(Field::new(
                format!("{}[distinct]", args.name),
                DataType::List(Arc::new(Field::new_list_field(DataType::Float64, true))),
                true,
            ))]);
        }
        Ok(["n", "mean", "m2"]
            .into_iter()
            .map(|name| {
                Arc::new(Field::new(
                    format!("{}[{name}]", args.name),
                    DataType::Float64,
                    false,
                ))
            })
            .collect())
    }
    fn accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(VarianceAccumulator {
            moments: Moments::default(),
            kind: self.kind,
            distinct: args
                .is_distinct
                .then(|| GenericDistinctBuffer::new(DataType::Float64)),
        }))
    }
    fn create_sliding_accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        if args.is_distinct {
            return not_impl_err!("DISTINCT statistical windows are not supported");
        }
        Ok(Box::new(SlidingVariance {
            values: VecDeque::new(),
            moments: Moments::default(),
            dirty: false,
            kind: self.kind,
        }))
    }
    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        !args.is_distinct
    }
    fn create_groups_accumulator(&self, _: AccumulatorArgs) -> Result<Box<dyn GroupsAccumulator>> {
        Ok(Box::new(VarianceGroups {
            groups: Vec::new(),
            kind: self.kind,
        }))
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Moments {
    n: f64,
    mean: f64,
    m2: f64,
}
impl Moments {
    fn update(&mut self, value: f64) {
        self.n += 1.0;
        let delta = value - self.mean;
        let delta_n = delta / self.n;
        self.mean += delta_n;
        self.m2 += delta * (delta - delta_n);
    }
    fn merge(&mut self, other: Self) {
        let n = self.n + other.n;
        let delta = other.mean - self.mean;
        let delta_n = if n == 0.0 { 0.0 } else { delta / n };
        self.mean += delta_n * other.n;
        self.m2 = self.m2 + other.m2 + delta * delta_n * self.n * other.n;
        self.n = n;
    }
}

#[derive(Debug)]
struct VarianceAccumulator {
    moments: Moments,
    kind: Kind,
    distinct: Option<GenericDistinctBuffer<Float64Type>>,
}
impl Accumulator for VarianceAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if let Some(distinct) = &mut self.distinct {
            return distinct.update_batch(values);
        }
        for v in as_float64_array(&values[0])?.iter().flatten() {
            self.moments.update(v);
        }
        Ok(())
    }
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if let Some(distinct) = &mut self.distinct {
            return distinct.merge_batch(states);
        }
        let n = as_float64_array(&states[0])?;
        let mean = as_float64_array(&states[1])?;
        let m2 = as_float64_array(&states[2])?;
        for i in 0..n.len() {
            self.moments.merge(Moments {
                n: n.value(i),
                mean: mean.value(i),
                m2: m2.value(i),
            });
        }
        Ok(())
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        if let Some(distinct) = &self.distinct {
            return distinct.state();
        }
        Ok([self.moments.n, self.moments.mean, self.moments.m2]
            .into_iter()
            .map(|v| ScalarValue::Float64(Some(v)))
            .collect())
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut m = self.moments;
        if let Some(distinct) = &self.distinct {
            m = Moments::default();
            for v in &distinct.values {
                m.update(v.0);
            }
        }
        Ok(ScalarValue::Float64(self.kind.evaluate(m)))
    }
    fn size(&self) -> usize {
        size_of_val(self) + self.distinct.as_ref().map_or(0, |d| d.size())
    }
}

// Sliding frames need their retained values: Spark recalculates each frame,
// whereas inverse moment updates accumulate different rounding and cannot
// recover when a NaN or infinity leaves the frame. This buffer is confined to
// bounded windows; ordinary scalar/grouped aggregation keeps constant state.
#[derive(Debug)]
struct SlidingVariance {
    values: VecDeque<f64>,
    moments: Moments,
    dirty: bool,
    kind: Kind,
}
impl SlidingVariance {
    fn refresh(&mut self) {
        if self.dirty {
            self.moments = Moments::default();
            for value in &self.values {
                self.moments.update(*value);
            }
            self.dirty = false;
        }
    }
}
impl Accumulator for SlidingVariance {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        for value in as_float64_array(&values[0])?.iter().flatten() {
            self.values.push_back(value);
            if !self.dirty {
                self.moments.update(value);
            }
        }
        Ok(())
    }
    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        for value in as_float64_array(&values[0])?.iter().flatten() {
            if self.values.pop_front().map(f64::to_bits) != Some(value.to_bits()) {
                return internal_err!("Sliding variance retraction differs from retained frame");
            }
            self.dirty = true;
        }
        Ok(())
    }
    fn supports_retract_batch(&self) -> bool {
        true
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        self.refresh();
        Ok(ScalarValue::Float64(self.kind.evaluate(self.moments)))
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        not_impl_err!("Sliding variance is not a partial aggregate")
    }
    fn merge_batch(&mut self, _: &[ArrayRef]) -> Result<()> {
        not_impl_err!("Sliding variance is not a partial aggregate")
    }
    fn size(&self) -> usize {
        size_of_val(self) + self.values.capacity() * size_of::<f64>()
    }
}

#[derive(Debug)]
struct VarianceGroups {
    groups: Vec<Moments>,
    kind: Kind,
}
impl GroupsAccumulator for VarianceGroups {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        indices: &[usize],
        filter: Option<&BooleanArray>,
        total: usize,
    ) -> Result<()> {
        self.groups.resize(total, Moments::default());
        accumulate(indices, as_float64_array(&values[0])?, filter, |i, v| {
            self.groups[i].update(v)
        });
        Ok(())
    }
    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        indices: &[usize],
        filter: Option<&BooleanArray>,
        total: usize,
    ) -> Result<()> {
        self.groups.resize(total, Moments::default());
        let n = as_float64_array(&values[0])?;
        let mean = as_float64_array(&values[1])?;
        let m2 = as_float64_array(&values[2])?;
        for (row, &group) in indices.iter().enumerate() {
            if filter.is_some_and(|f| f.is_null(row) || !f.value(row)) {
                continue;
            }
            self.groups[group].merge(Moments {
                n: n.value(row),
                mean: mean.value(row),
                m2: m2.value(row),
            });
        }
        Ok(())
    }
    fn state(&mut self, emit: EmitTo) -> Result<Vec<ArrayRef>> {
        let groups = emit.take_needed(&mut self.groups);
        Ok(vec![
            Arc::new(Float64Array::from_iter_values(groups.iter().map(|m| m.n))),
            Arc::new(Float64Array::from_iter_values(
                groups.iter().map(|m| m.mean),
            )),
            Arc::new(Float64Array::from_iter_values(groups.iter().map(|m| m.m2))),
        ])
    }
    fn evaluate(&mut self, emit: EmitTo) -> Result<ArrayRef> {
        Ok(Arc::new(Float64Array::from_iter(
            emit.take_needed(&mut self.groups)
                .into_iter()
                .map(|m| self.kind.evaluate(m)),
        )))
    }
    fn size(&self) -> usize {
        size_of_val(self) + self.groups.capacity() * size_of::<Moments>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn kind() -> Kind {
        Kind {
            sample: true,
            sqrt: false,
        }
    }
    fn scalar() -> VarianceAccumulator {
        VarianceAccumulator {
            moments: Moments::default(),
            kind: kind(),
            distinct: None,
        }
    }
    fn array(values: Vec<Option<f64>>) -> ArrayRef {
        Arc::new(Float64Array::from(values))
    }

    #[test]
    fn spark_update_order_on_reduced_inventory_groups() {
        for (values, expected) in [
            ([10., 404., 13., 814.], 147020.25000000003_f64),
            ([113., 838., 88., 105.], 135532.6666666667_f64),
        ] {
            let mut m = Moments::default();
            for value in values {
                m.update(value);
            }
            assert_eq!(kind().evaluate(m), Some(expected));
        }
    }
    #[test]
    fn scalar_partial_states_merge_in_spark_expression_order() -> Result<()> {
        let mut left = scalar();
        left.update_batch(&[array(vec![Some(10.), Some(404.)])])?;
        let mut right = scalar();
        right.update_batch(&[array(vec![Some(13.), Some(814.)])])?;
        let mut result = scalar();
        for partial in [&mut left, &mut right] {
            let states = partial
                .state()?
                .into_iter()
                .map(|s| s.to_array())
                .collect::<Result<Vec<_>>>()?;
            result.merge_batch(&states)?;
        }
        // Independent Spark local[2] uses two partial buffers for these VALUES.
        assert_eq!(result.evaluate()?, ScalarValue::Float64(Some(147020.25)));
        Ok(())
    }
    #[test]
    fn empty_singleton_nonfinite_and_population_semantics() {
        assert_eq!(kind().evaluate(Moments::default()), None);
        let pop = Kind {
            sample: false,
            sqrt: false,
        };
        let mut m = Moments::default();
        m.update(7.);
        assert_eq!(kind().evaluate(m), None);
        assert_eq!(pop.evaluate(m), Some(0.));
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let mut m = Moments::default();
            m.update(value);
            assert_eq!(kind().evaluate(m), None);
            assert!(pop.evaluate(m).is_some_and(f64::is_nan));
        }
    }
    #[test]
    fn grouped_filter_nulls_and_partial_emission() -> Result<()> {
        let mut groups = VarianceGroups {
            groups: Vec::new(),
            kind: kind(),
        };
        groups.update_batch(
            &[array(vec![Some(1.), Some(2.), Some(3.), Some(8.), None])],
            &[0, 0, 0, 1, 2],
            Some(&BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(true),
                Some(true),
                None,
            ])),
            3,
        )?;
        let result = groups.evaluate(EmitTo::First(1))?;
        assert_eq!(
            as_float64_array(&result)?.iter().collect::<Vec<_>>(),
            vec![Some(2.)]
        );
        groups.update_batch(&[array(vec![Some(9.)])], &[0], None, 2)?;
        let states = groups.state(EmitTo::All)?;
        let mut merged = VarianceGroups {
            groups: Vec::new(),
            kind: kind(),
        };
        merged.merge_batch(&states, &[0, 1], None, 2)?;
        let result = merged.evaluate(EmitTo::All)?;
        assert_eq!(
            as_float64_array(&result)?.iter().collect::<Vec<_>>(),
            vec![Some(0.5), None]
        );
        assert!(groups.groups.is_empty());
        assert!(merged.groups.is_empty());
        Ok(())
    }
    #[test]
    fn scalar_and_grouped_paths_share_moment_arithmetic() -> Result<()> {
        let values = [array(vec![
            Some(10.),
            Some(404.),
            None,
            Some(13.),
            Some(814.),
        ])];
        let mut scalar = scalar();
        scalar.update_batch(&values)?;
        let mut groups = VarianceGroups {
            groups: Vec::new(),
            kind: kind(),
        };
        groups.update_batch(&values, &[0, 0, 0, 0, 0], None, 1)?;
        let result = groups.evaluate(EmitTo::All)?;
        assert_eq!(
            ScalarValue::Float64(Some(as_float64_array(&result)?.value(0))),
            scalar.evaluate()?
        );
        Ok(())
    }
    #[test]
    fn distinct_partial_state_deduplicates_across_batches() -> Result<()> {
        let new = || VarianceAccumulator {
            moments: Moments::default(),
            kind: kind(),
            distinct: Some(GenericDistinctBuffer::new(DataType::Float64)),
        };
        let mut left = new();
        left.update_batch(&[array(vec![Some(1.), Some(2.), None])])?;
        let mut right = new();
        right.update_batch(&[array(vec![Some(2.), Some(3.), None])])?;
        let mut result = new();
        for partial in [&mut left, &mut right] {
            let states = partial
                .state()?
                .into_iter()
                .map(|s| s.to_array())
                .collect::<Result<Vec<_>>>()?;
            result.merge_batch(&states)?;
        }
        assert_eq!(result.evaluate()?, ScalarValue::Float64(Some(1.)));
        Ok(())
    }
    #[test]
    fn sliding_frame_recomputes_after_retraction() -> Result<()> {
        let mut a = SlidingVariance {
            values: VecDeque::new(),
            moments: Moments::default(),
            dirty: false,
            kind: kind(),
        };
        a.update_batch(&[array(vec![
            Some(10.),
            Some(404.),
            Some(13.),
            Some(814.),
            None,
        ])])?;
        a.retract_batch(&[array(vec![Some(10.)])])?;
        assert_eq!(a.evaluate()?, ScalarValue::Float64(Some(160430.3333333333)));
        a.retract_batch(&[array(vec![Some(404.), Some(13.), Some(814.), None])])?;
        assert_eq!(a.evaluate()?, ScalarValue::Float64(None));
        Ok(())
    }
    #[test]
    fn sliding_frame_recovers_when_nonfinite_values_leave() -> Result<()> {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut a = SlidingVariance {
                values: VecDeque::new(),
                moments: Moments::default(),
                dirty: false,
                kind: kind(),
            };
            a.update_batch(&[array(vec![Some(value), None, Some(1.), Some(2.), Some(3.)])])?;
            assert!(matches!(a.evaluate()?,ScalarValue::Float64(Some(x)) if x.is_nan()));
            a.retract_batch(&[array(vec![Some(value), None])])?;
            assert_eq!(a.evaluate()?, ScalarValue::Float64(Some(1.)));
        }
        Ok(())
    }
}
