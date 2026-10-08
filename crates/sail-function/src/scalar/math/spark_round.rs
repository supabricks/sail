//! Decimal ROUND has a narrower result precision in Spark than in DataFusion.
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::functions::math::round::RoundFunc;
use datafusion_common::{Result, ScalarValue, plan_err};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
};

#[derive(Debug, PartialEq, Eq, Hash, Default)]
pub struct SparkRound {
    inner: RoundFunc,
}

impl ScalarUDFImpl for SparkRound {
    fn name(&self) -> &str {
        "spark_round"
    }
    fn signature(&self) -> &Signature {
        self.inner.signature()
    }
    fn return_type(&self, types: &[DataType]) -> Result<DataType> {
        self.inner.return_type(types)
    }
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        if let DataType::Decimal128(p, s) = args.arg_fields[0].data_type() {
            let places = match args.scalar_arguments.get(1) {
                None => 0,
                Some(Some(ScalarValue::Int32(Some(n)))) => i64::from(*n),
                Some(Some(ScalarValue::Int64(Some(n)))) => *n,
                Some(Some(v)) if v.is_null() => 0,
                _ => return plan_err!("ROUND decimal scale must be a constant integer"),
            };
            let integral = i64::from(*p) - i64::from(*s) + 1;
            let scale = i64::from(*s).min(places.max(0));
            let precision = if places < 0 {
                integral.max(places.saturating_neg().saturating_add(1))
            } else {
                integral + scale
            };
            Ok(Arc::new(Field::new(
                self.name(),
                DataType::Decimal128(precision.min(38) as u8, scale as i8),
                true,
            )))
        } else {
            self.inner.return_field_from_args(args)
        }
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        self.inner.invoke_with_args(args)
    }
}
