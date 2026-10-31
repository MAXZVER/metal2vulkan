//! Compute builtin and constant-value binding helpers.

use crate::reflect::KernelStageInput;

/// One `[[stage_in]]` scalar load: struct member (location), lane within the
/// attribute, byte offset inside the stride, and the storage scalar kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::passes) struct StageInScalar {
    pub location: u32,
    pub lane: u32,
    pub offset: u32,
    pub half: bool,
    pub signed: bool,
    /// `false` and not `signed`/`half` = a plain `uint` format.
    pub float: bool,
}

/// The validated plan for one `[[stage_in]]` parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::passes) struct StageInPlan {
    pub buffer_index: u32,
    pub stride: u32,
    /// 0 = `ThreadPositionInGridX`, 1 = `...GridY`.
    pub axis: u32,
    /// Scalars sorted by (location, lane); the block's flat member order.
    pub scalars: Vec<StageInScalar>,
    /// Lanes per attribute, in location order.
    pub lanes_of: Vec<(u32, u32)>,
}

fn format_components(format: u32) -> Result<(u32, bool, bool, bool), String> {
    // MTLAttributeFormat ordinals; only the 4-byte scalar families lower this
    // revision, the 8/16-bit integer and normalized families refuse by name.
    let (lanes, half, signed, float) = match format {
        25 => (2, true, false, false),                       // Half2
        26 => (3, true, false, false),                       // Half3
        27 => (4, true, false, false),                       // Half4
        28 => (1, false, false, true),                       // Float
        29 => (2, false, false, true),                       // Float2
        30 => (3, false, false, true),                       // Float3
        31 => (4, false, false, true),                       // Float4
        32..=35 => (1 + (format - 32), false, true, false),  // Int..Int4
        36..=39 => (1 + (format - 36), false, false, false), // UInt..UInt4
        1..=24 => {
            return Err(format!(
                "MTLAttributeFormat ordinal {format} (an 8/16-bit or normalized \
                 vector format) needs 8/16-bit integer buffer loads, which this \
                 translator does not emit"
            ))
        }
        other => {
            return Err(format!(
                "stage-in attribute format {other} is not an MTLAttributeFormat \
                 this translator knows"
            ))
        }
    };
    Ok((lanes, half, signed, float))
}

/// Validate a descriptor and lay out its scalar loads. Refusals are typed.
pub(in crate::passes) fn plan_kernel_stage_input(
    input: &KernelStageInput,
) -> Result<StageInPlan, String> {
    let attributes: Vec<crate::reflect::KernelStageInAttribute> = input.attributes().collect();
    let buffer_index = attributes
        .first()
        .map(|attribute| attribute.buffer_index)
        .ok_or("stage-input descriptor has no attributes")?;
    if attributes
        .iter()
        .any(|attribute| attribute.buffer_index != buffer_index)
    {
        return Err(format!(
            "stage-input descriptor spans more than one buffer table slot \
             (first {buffer_index}); per-buffer fetches are not lowered"
        ));
    }
    let layout = input
        .layouts()
        .find(|layout| layout.buffer_index == buffer_index)
        .ok_or_else(|| format!("stage-input descriptor has no layout for buffer {buffer_index}"))?;
    let axis = match layout.step_function {
        5 => 0,
        6 => 1,
        7 | 8 => {
            return Err(format!(
                "stage-input step function {} (ThreadPositionInGrid{{X,Y}}Indexed) \
                 needs the index buffer, which this translator does not lower",
                layout.step_function
            ))
        }
        other => {
            return Err(format!(
                "stage-input step function {other} is not a compute stage-in \
                 step this translator lowers"
            ))
        }
    };
    let stride = layout.stride;
    if stride == 0 {
        return Err("stage-input layout stride is zero".to_string());
    }
    let mut locations: Vec<u32> = attributes.iter().map(|a| a.location).collect();
    locations.sort_unstable();
    locations.dedup();
    if locations.first() != Some(&0) || locations.len() != attributes.len() {
        return Err(format!(
            "stage-input attribute locations {locations:?} do not name struct \
             members 0..n in order"
        ));
    }
    let mut scalars = Vec::new();
    let mut lanes_of = Vec::new();
    for attribute in attributes.iter() {
        let (lanes, half, signed, float) = format_components(attribute.format)?;
        let scalar_size = if half { 2 } else { 4 };
        let end = attribute
            .offset
            .checked_add(scalar_size * lanes)
            .ok_or("stage-input attribute offset overflows")?;
        if end > stride {
            return Err(format!(
                "stage-input stride {stride} is smaller than attribute location {} \
                 needs (offset {} + {} bytes)",
                attribute.location,
                attribute.offset,
                end - attribute.offset
            ));
        }
        for lane in 0..lanes {
            scalars.push(StageInScalar {
                location: attribute.location,
                lane,
                offset: attribute.offset + lane * scalar_size,
                half,
                signed,
                float,
            });
        }
        lanes_of.push((attribute.location, lanes));
    }
    scalars.sort_by_key(|s| (s.location, s.lane));
    let mut offsets: Vec<u32> = scalars.iter().map(|s| s.offset).collect();
    offsets.sort_unstable();
    offsets.dedup();
    if offsets.len() != scalars.len() {
        return Err("stage-in attribute offsets overlap".to_string());
    }
    Ok(StageInPlan {
        buffer_index,
        stride,
        axis,
        scalars,
        lanes_of,
    })
}

/// One attribute's load shape, as the collapse pass consumes it.
#[derive(Clone, Copy, Debug)]
pub(in crate::passes) struct StageInAttr {
    /// First flat struct member of this attribute.
    pub first_member: u32,
    pub lanes: u32,
    /// Scalar type of each loaded member (f16/f32/i32/u32).
    pub scalar_ty: Word,
    /// Vector type for `lanes > 1`, already built.
    pub vec_ty: Option<Word>,
    /// Half storage: a float member of the AIR struct converts per lane.
    pub half: bool,
}

/// The declared block variable plus the per-attribute load shapes.
pub(in crate::passes) struct StageInBlock {
    pub var: Word,
    pub attrs: Vec<StageInAttr>,
}

/// Declare the StorageBuffer that mirrors the descriptor: one scalar member
/// per loaded component at its explicit descriptor offset (std430-legal for
/// scalar members at any stride), `ArrayStride` = the layout stride.
pub(in crate::passes) fn declare_stage_in_block(
    ctx: &mut crate::passes::Ctx,
    plan: &StageInPlan,
) -> Result<StageInBlock, String> {
    let mut member_tys: Vec<Word> = Vec::with_capacity(plan.scalars.len());
    let mut member_offsets: Vec<u32> = Vec::with_capacity(plan.scalars.len());
    if plan.scalars.iter().any(|scalar| scalar.half) {
        ctx.add_capability(spirv::Capability::Float16);
        ctx.add_capability(spirv::Capability::StorageBuffer16BitAccess);
    }
    for scalar in &plan.scalars {
        let ty = if scalar.half {
            ctx.ty_half()
        } else if scalar.signed {
            let id = ctx.module.fresh_id();
            ctx.new_globals.push(type_inst(
                Op::TypeInt,
                id,
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(1)],
            ));
            id
        } else if scalar.float {
            ctx.ty_float()
        } else {
            ctx.ty_uint()
        };
        member_tys.push(ty);
        member_offsets.push(scalar.offset);
    }
    let block_ty = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeStruct,
        block_ty,
        member_tys.iter().map(|ty| Operand::IdRef(*ty)).collect(),
    ));
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(block_ty),
            Operand::Decoration(Decoration::Block),
        ],
    ));
    for (member, offset) in member_offsets.iter().enumerate() {
        let member = member as u32;
        ctx.module.annotations.push(Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(block_ty),
                Operand::LiteralBit32(member),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(*offset),
            ],
        ));
    }
    let rta = ctx.ty_runtime_array(block_ty);
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(rta),
            Operand::Decoration(Decoration::ArrayStride),
            Operand::LiteralBit32(plan.stride),
        ],
    ));
    let pptr = ctx.ty_ptr(StorageClass::StorageBuffer, rta);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(pptr),
        Some(var),
        vec![Operand::StorageClass(StorageClass::StorageBuffer)],
    ));
    let mut attrs = Vec::with_capacity(plan.lanes_of.len());
    let mut next_member = 0u32;
    for &(location, lanes) in &plan.lanes_of {
        let scalar = plan
            .scalars
            .iter()
            .find(|s| s.location == location)
            .ok_or("stage-in plan lost an attribute")?;
        let scalar_ty = member_tys[next_member as usize];
        let vec_ty = match (scalar.half, scalar.signed, scalar.float) {
            (true, _, _) => Some(ctx.ty_vech(lanes)),
            (false, true, _) => Some(ctx.ty_vec_sint(lanes)),
            (false, false, true) => Some(ctx.ty_vecf(lanes)),
            (false, false, false) => Some(ctx.ty_vec_uint(lanes)),
        };
        let vec_ty = (lanes > 1).then_some(vec_ty).flatten();
        attrs.push(StageInAttr {
            first_member: next_member,
            lanes,
            scalar_ty,
            vec_ty,
            half: scalar.half,
        });
        next_member += lanes;
    }
    Ok(StageInBlock { var, attrs })
}

/// Lanes of an entry parameter's type: 1 for scalars, N for `OpTypeVector`.
pub(in crate::passes) fn vector_lanes_of(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
) -> Option<u32> {
    match defs.get(&ty)?.operands.as_slice() {
        [Operand::IdRef(_), Operand::LiteralBit32(lanes)] => Some(*lanes),
        _ => None,
    }
}

#[cfg(test)]
mod task71_tests {
    use super::*;
    use crate::reflect::{KernelStageInAttribute, KernelStageInLayout, KernelStageInput};

    fn descriptor(format: u32, offset: u32, stride: u32, step: u32) -> KernelStageInput {
        let mut input = KernelStageInput {
            attributes: [None; crate::reflect::STAGE_IN_MAX],
            attributes_len: 1,
            layouts: [None; crate::reflect::STAGE_IN_MAX],
            layouts_len: 1,
        };
        input.attributes[0] = Some(KernelStageInAttribute {
            location: 0,
            buffer_index: 1,
            format,
            offset,
        });
        input.layouts[0] = Some(KernelStageInLayout {
            buffer_index: 1,
            step_function: step,
            stride,
        });
        input
    }

    /// pipe 125: Float3, offset 0, stride 12, step X.
    #[test]
    fn float3_stride12_x() {
        let plan = plan_kernel_stage_input(&descriptor(30, 0, 12, 5)).unwrap();
        assert_eq!(plan.axis, 0);
        assert_eq!(plan.stride, 12);
        assert_eq!(plan.scalars.len(), 3);
        assert_eq!(plan.scalars[2].offset, 8);
        assert!(!plan.scalars[0].half);
        assert!(plan.scalars[0].signed == false);
    }

    /// pipe 169: Half3, stride 8 — scalars at 0/2/4 bytes.
    #[test]
    fn half3_stride8() {
        let plan = plan_kernel_stage_input(&descriptor(26, 0, 8, 5)).unwrap();
        assert_eq!(plan.scalars.len(), 3);
        assert_eq!(plan.scalars[1].offset, 2);
        assert!(plan.scalars[0].half);
    }

    #[test]
    fn y_step_selects_axis1() {
        assert_eq!(
            plan_kernel_stage_input(&descriptor(30, 0, 12, 6))
                .unwrap()
                .axis,
            1
        );
    }

    #[test]
    fn indexed_step_refuses_by_name() {
        let err = plan_kernel_stage_input(&descriptor(30, 0, 12, 7)).unwrap_err();
        assert!(err.contains("ThreadPositionInGrid"), "{err}");
    }

    #[test]
    fn unknown_format_refuses_by_ordinal() {
        let err = plan_kernel_stage_input(&descriptor(40, 0, 16, 5)).unwrap_err();
        assert!(err.contains("40"), "{err}");
    }

    #[test]
    fn normalized_refuses_by_ordinal() {
        let err = plan_kernel_stage_input(&descriptor(9, 0, 4, 5)).unwrap_err();
        assert!(err.contains('9') && err.contains("8/16-bit"), "{err}");
    }

    #[test]
    fn small_stride_refuses() {
        assert!(plan_kernel_stage_input(&descriptor(30, 0, 8, 5)).is_err());
    }

    #[test]
    fn two_buffers_refuse() {
        let mut input = descriptor(28, 0, 8, 5);
        input.attributes[1] = Some(KernelStageInAttribute {
            location: 1,
            buffer_index: 2,
            format: 28,
            offset: 4,
        });
        input.attributes_len = 2;
        let err = plan_kernel_stage_input(&input).unwrap_err();
        assert!(err.contains("more than one buffer"), "{err}");
    }

    #[test]
    fn int_formats_are_signed_uints_are_not() {
        let plan = plan_kernel_stage_input(&descriptor(34, 0, 16, 5)).unwrap();
        assert!(plan.scalars[0].signed);
        let plan = plan_kernel_stage_input(&descriptor(38, 0, 16, 5)).unwrap();
        assert!(!plan.scalars[0].signed);
    }
}

use super::*;

pub(in crate::passes) fn bind_kernel_uvec3_builtin(
    ctx: &mut Ctx,
    defs: &HashMap<Word, Instruction>,
    bindings: &mut Vec<(Word, ParamBinding)>,
    pid: Word,
    pty: Word,
    builtin: BuiltIn,
) {
    // AIR compute builtins can be scalar `uint` or `uint3`. Vulkan exposes the corresponding compute
    // builtins as v3uint; scalar kernels receive .x, vector kernels receive the full vector.
    let var = bind_kernel_v3uint_builtin(ctx, builtin);
    bind_kernel_uvec3_builtin_var(ctx, defs, bindings, pid, pty, var);
}

pub(in crate::passes) fn bind_kernel_uvec3_builtin_var(
    ctx: &mut Ctx,
    defs: &HashMap<Word, Instruction>,
    bindings: &mut Vec<(Word, ParamBinding)>,
    pid: Word,
    pty: Word,
    var: Word,
) {
    let uint_ty = ctx.ty_uint();
    let v2u = ctx.ty_vec_uint(2);
    let v3u = ctx.ty_vec_uint(3);
    let vector_lanes = scalar_or_vector_component(defs, pty).and_then(|(_component, lanes)| lanes);
    if pty == v3u {
        bindings.push((pid, ParamBinding::LoadVar { var, ty: v3u }));
    } else if pty == v2u || vector_lanes == Some(2) {
        bindings.push((
            pid,
            ParamBinding::LoadVarVectorPrefix {
                var,
                vec_ty: v3u,
                scalar_ty: uint_ty,
                prefix_ty: v2u,
                out_ty: if pty == v2u { v2u } else { pty },
                lanes: 2,
            },
        ));
    } else if vector_lanes == Some(3) {
        bindings.push((
            pid,
            ParamBinding::LoadVarVectorPrefix {
                var,
                vec_ty: v3u,
                scalar_ty: uint_ty,
                prefix_ty: v3u,
                out_ty: pty,
                lanes: 3,
            },
        ));
    } else {
        let scalar_ty = if pty == uint_ty { pty } else { uint_ty };
        bindings.push((
            pid,
            ParamBinding::LoadVarComponent {
                var,
                vec_ty: v3u,
                scalar_ty,
                out_ty: pty,
                comp: 0,
            },
        ));
    }
}

pub(in crate::passes) fn const_kernel_local_size(
    ctx: &mut Ctx,
    defs: &HashMap<Word, Instruction>,
    ty: Word,
    values: [u32; 3],
) -> Option<Word> {
    let def = defs.get(&ty)?;
    match def.class.opcode {
        Op::TypeInt => Some(ctx.const_int_of(ty, values[0] as i64)),
        Op::TypeVector => {
            let component = match def.operands.first()? {
                Operand::IdRef(component) => *component,
                _ => return None,
            };
            if !matches!(
                defs.get(&component).map(|inst| inst.class.opcode),
                Some(Op::TypeInt)
            ) {
                return None;
            }
            let lanes = match def.operands.get(1)? {
                Operand::LiteralBit32(lanes @ 2..=3) => *lanes,
                _ => return None,
            };
            let ops = values[..lanes as usize]
                .iter()
                .copied()
                .map(|value| Operand::IdRef(ctx.const_int_of(component, value as i64)))
                .collect();
            let id = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::ConstantComposite,
                Some(ty),
                Some(id),
                ops,
            ));
            Some(id)
        }
        _ => None,
    }
}

pub(in crate::passes) fn const_ivec(ctx: &mut Ctx, ty: Word, values: &[i32]) -> Word {
    let int_ty = ctx.ty_sint();
    let ops = values
        .iter()
        .copied()
        .map(|value| Operand::IdRef(ctx.const_int_of(int_ty, value as i64)))
        .collect();
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::ConstantComposite,
        Some(ty),
        Some(id),
        ops,
    ));
    id
}

pub(in crate::passes) fn is_raw_uint_buffer_block(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
) -> bool {
    let Some(block) = defs.get(&ty) else {
        return false;
    };
    if block.class.opcode != Op::TypeStruct || block.operands.len() != 1 {
        return false;
    }
    let Some(Operand::IdRef(runtime)) = block.operands.first() else {
        return false;
    };
    let Some(runtime_def) = defs.get(runtime) else {
        return false;
    };
    if runtime_def.class.opcode != Op::TypeRuntimeArray {
        return false;
    }
    let Some(Operand::IdRef(elem)) = runtime_def.operands.first() else {
        return false;
    };
    let Some(elem_def) = defs.get(elem) else {
        return false;
    };
    elem_def.class.opcode == Op::TypeInt
        && elem_def.operands.first() == Some(&Operand::LiteralBit32(32))
        && elem_def.operands.get(1) == Some(&Operand::LiteralBit32(0))
}
