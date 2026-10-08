//! Spark decimal arithmetic keeps operand scales until the final HALF_UP rounding.
//! DataFusion's binary decimal operators use different division and scale-reduction
//! rules, so casting their result cannot recover the missing fractional digits.
use std::sync::Arc;

use datafusion::arrow::array::{AsArray, Decimal128Array};
use datafusion::arrow::datatypes::{DataType, Decimal128Type, Field, FieldRef};
use datafusion_common::{Result, ScalarValue, exec_err, plan_err};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use num::{BigInt, Integer, Signed, ToPrimitive};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecimalOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkDecimal {
    pub op: DecimalOp,
    pub ansi_mode: bool,
    pub allow_precision_loss: bool,
    signature: Signature,
}

impl SparkDecimal {
    pub fn new(op: DecimalOp, ansi_mode: bool, allow_precision_loss: bool) -> Self {
        Self {
            op,
            ansi_mode,
            allow_precision_loss,
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }

    pub fn result_type(&self, types: &[DataType]) -> Result<DataType> {
        let [DataType::Decimal128(p1, s1), DataType::Decimal128(p2, s2)] = types else {
            return plan_err!("decimal arithmetic requires two Decimal128 operands: {types:?}");
        };
        let (p1, s1, p2, s2) = (
            i16::from(*p1),
            i16::from(*s1),
            i16::from(*p2),
            i16::from(*s2),
        );
        let (precision, scale) = match self.op {
            DecimalOp::Add | DecimalOp::Subtract => {
                let scale = s1.max(s2);
                ((p1 - s1).max(p2 - s2) + scale + 1, scale)
            }
            DecimalOp::Multiply => (p1 + p2 + 1, s1 + s2),
            DecimalOp::Divide => {
                let scale = 6.max(s1 + p2 + 1);
                if self.allow_precision_loss {
                    (p1 - s1 + s2 + scale, scale)
                } else {
                    let integral = 38.min(p1 - s1 + s2);
                    let mut fractional = 38.min(scale);
                    let excess = integral + fractional - 38;
                    if excess > 0 {
                        fractional -= excess / 2 + 1;
                    }
                    ((integral + fractional).min(38), fractional)
                }
            }
        };
        // Spark DecimalType.adjustPrecisionScale (allowPrecisionLoss=true).
        let scale = if self.allow_precision_loss && precision > 38 && scale >= 0 {
            (38 - (precision - scale)).max(scale.min(6))
        } else {
            scale.min(38)
        };
        Ok(DataType::Decimal128(precision.min(38) as u8, scale as i8))
    }
}

fn ten(exponent: u32) -> BigInt {
    BigInt::from(10).pow(exponent)
}

fn half_up(numerator: BigInt, denominator: BigInt) -> BigInt {
    let (quotient, remainder) = numerator.div_rem(&denominator);
    if remainder.abs() * 2 >= denominator.abs() {
        quotient + numerator.signum() * denominator.signum()
    } else {
        quotient
    }
}

fn rescale(value: BigInt, from: i16, to: i16) -> BigInt {
    if to >= from {
        value * ten((to - from) as u32)
    } else {
        half_up(value, ten((from - to) as u32))
    }
}

// Most warehouse values fit in i128 even when declared as DECIMAL(38,s).
// Use checked integer arithmetic in that common case; arbitrary precision is
// only needed when an intermediate exceeds i128, never a reason to overflow
// a result that would fit in the declared decimal type.
fn half_up_i128(n: i128, d: i128) -> Option<i128> {
    let q = n.checked_div(d)?;
    let r = n.checked_rem(d)?;
    if r.unsigned_abs() * 2 >= d.unsigned_abs() {
        q.checked_add(n.signum() * d.signum())
    } else {
        Some(q)
    }
}

fn rescale_i128(value: i128, from: i16, to: i16) -> Option<i128> {
    if to >= from {
        value.checked_mul(10_i128.checked_pow((to - from) as u32)?)
    } else {
        half_up_i128(value, 10_i128.checked_pow((from - to) as u32)?)
    }
}

fn calculate_i128(op: DecimalOp, l: i128, r: i128, s1: i16, s2: i16, out: i16) -> Option<i128> {
    match op {
        DecimalOp::Add | DecimalOp::Subtract => {
            let common = s1.max(s2);
            let l = rescale_i128(l, s1, common)?;
            let r = rescale_i128(r, s2, common)?;
            rescale_i128(
                if op == DecimalOp::Add {
                    l.checked_add(r)?
                } else {
                    l.checked_sub(r)?
                },
                common,
                out,
            )
        }
        DecimalOp::Multiply => rescale_i128(l.checked_mul(r)?, s1 + s2, out),
        DecimalOp::Divide => {
            let shift = out + s2 - s1;
            if shift >= 0 {
                half_up_i128(l.checked_mul(10_i128.checked_pow(shift as u32)?)?, r)
            } else {
                half_up_i128(l, r.checked_mul(10_i128.checked_pow((-shift) as u32)?)?)
            }
        }
    }
}

impl ScalarUDFImpl for SparkDecimal {
    fn name(&self) -> &str {
        match self.op {
            DecimalOp::Add => "spark_decimal_add",
            DecimalOp::Subtract => "spark_decimal_subtract",
            DecimalOp::Multiply => "spark_decimal_multiply",
            DecimalOp::Divide => "spark_decimal_divide",
        }
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, types: &[DataType]) -> Result<DataType> {
        self.result_type(types)
    }
    fn coerce_types(&self, types: &[DataType]) -> Result<Vec<DataType>> {
        self.result_type(types)?;
        Ok(types.to_vec())
    }
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let types = args
            .arg_fields
            .iter()
            .map(|f| f.data_type().clone())
            .collect::<Vec<_>>();
        Ok(Arc::new(Field::new(
            self.name(),
            self.result_type(&types)?,
            !self.ansi_mode || args.arg_fields.iter().any(|f| f.is_nullable()),
        )))
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let types = args.args.iter().map(|a| a.data_type()).collect::<Vec<_>>();
        let DataType::Decimal128(precision, scale) = self.result_type(&types)? else {
            unreachable!()
        };
        let [DataType::Decimal128(_, s1), DataType::Decimal128(_, s2)] = types.as_slice() else {
            unreachable!()
        };
        let (s1, s2, out_scale) = (i16::from(*s1), i16::from(*s2), i16::from(scale));
        let scalar = args
            .args
            .iter()
            .all(|a| matches!(a, ColumnarValue::Scalar(_)));
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let left = arrays[0].as_primitive::<Decimal128Type>();
        let right = arrays[1].as_primitive::<Decimal128Type>();
        let limit_i128 = 10_i128.pow(u32::from(precision));
        let limit = BigInt::from(limit_i128);
        let mut result = Vec::with_capacity(left.len());
        for (l, r) in left.iter().zip(right.iter()) {
            let (Some(l), Some(r)) = (l, r) else {
                result.push(None);
                continue;
            };
            if self.op == DecimalOp::Divide && r == 0 {
                if self.ansi_mode {
                    return exec_err!("Division by zero");
                }
                result.push(None);
                continue;
            }
            if let Some(value) = calculate_i128(self.op, l, r, s1, s2, out_scale) {
                if value.unsigned_abs() >= limit_i128 as u128 {
                    if self.ansi_mode {
                        return exec_err!("Decimal overflow for decimal({precision},{scale})");
                    }
                    result.push(None);
                } else {
                    result.push(Some(value));
                }
                continue;
            }
            let (l, r) = (BigInt::from(l), BigInt::from(r));
            let value = match self.op {
                DecimalOp::Add | DecimalOp::Subtract => {
                    let common = s1.max(s2);
                    let l = rescale(l, s1, common);
                    let r = rescale(r, s2, common);
                    rescale(
                        if self.op == DecimalOp::Add {
                            l + r
                        } else {
                            l - r
                        },
                        common,
                        out_scale,
                    )
                }
                DecimalOp::Multiply => rescale(l * r, s1 + s2, out_scale),
                DecimalOp::Divide => {
                    let shift = out_scale + s2 - s1;
                    if shift >= 0 {
                        half_up(l * ten(shift as u32), r)
                    } else {
                        half_up(l, r * ten((-shift) as u32))
                    }
                }
            };
            if value.abs() >= limit {
                if self.ansi_mode {
                    return exec_err!("Decimal overflow for decimal({precision},{scale})");
                }
                result.push(None);
            } else {
                result.push(value.to_i128());
            }
        }
        let array = Decimal128Array::from(result).with_precision_and_scale(precision, scale)?;
        if scalar {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &array, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(Arc::new(array)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spark_result_types() -> Result<()> {
        for (op, a, b, expected) in [
            (DecimalOp::Divide, (15, 4), (15, 4), (35, 20)),
            (DecimalOp::Divide, (17, 2), (17, 2), (37, 20)),
            (DecimalOp::Divide, (21, 2), (27, 2), (38, 17)),
            (DecimalOp::Divide, (21, 4), (17, 2), (38, 19)),
            (DecimalOp::Divide, (20, 0), (2, 1), (27, 6)),
            (DecimalOp::Multiply, (38, 18), (1, 0), (38, 16)),
            (DecimalOp::Add, (38, 18), (38, 18), (38, 17)),
        ] {
            assert_eq!(
                SparkDecimal::new(op, true, true).result_type(&[
                    DataType::Decimal128(a.0, a.1),
                    DataType::Decimal128(b.0, b.1)
                ])?,
                DataType::Decimal128(expected.0, expected.1)
            );
        }
        assert_eq!(
            SparkDecimal::new(DecimalOp::Divide, true, false)
                .result_type(&[DataType::Decimal128(38, 18), DataType::Decimal128(38, 18)])?,
            DataType::Decimal128(38, 18)
        );
        Ok(())
    }

    #[test]
    fn signed_half_up_and_wide_intermediates() {
        for (n, d, out) in [
            (1, 2, 1),
            (-1, 2, -1),
            (1, -2, -1),
            (-1, -2, 1),
            (1, 3, 0),
            (-2, 3, -1),
        ] {
            assert_eq!(half_up(n.into(), d.into()), BigInt::from(out));
            assert_eq!(half_up_i128(n, d), Some(out));
        }
        let wide = ten(100) + BigInt::from(5) * ten(61);
        assert_eq!(rescale(wide, 100, 38), ten(38) + BigInt::from(1));
    }
}
