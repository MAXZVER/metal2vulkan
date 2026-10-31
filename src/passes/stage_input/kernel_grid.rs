//! Exact-thread dispatch payload shared by AIR grid builtins.

use super::*;

/// Declare the eighteen-scalar push-constant block once. Separate `u32` members keep the public ABI
/// tightly packed without inheriting a `uvec3` aggregate alignment. Members 12-14 are the
/// stage-in region origin, 15-17 its size (Metal's `MTLRegion` is 3D; `[[stage_in_grid_origin]]`
/// and `[[stage_in_grid_size]]` may be `uint`, `uint2` or `uint3`).
///
/// Metal's `MTLComputeCommandEncoder.setStageInRegion:` states the region whose ORIGIN offsets
/// the per-thread stage-in fetch: for `MTLStepFunctionThreadPositionInGridX` the attribute index
/// is `thread_position_in_grid.x + region.origin.x` (and .y/.origin.y for the Y form) — the
/// origin indexes INTO the stage-in buffer, which is exactly what the `[[stage_in]]` lowering
/// and the `[[stage_in_grid_origin]]` parameter both read here.
pub(in crate::passes) fn bind_kernel_grid_push_constant_once(
    ctx: &mut Ctx,
    slot: &mut Option<Word>,
    offset: u32,
) -> Word {
    if let Some(var) = *slot {
        return var;
    }
    let uint_ty = ctx.ty_uint();
    let block_ty = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeStruct,
        block_ty,
        (0..crate::reflect::KERNEL_DISPATCH_PUSH_CONSTANT_SIZE / 4)
            .map(|_| Operand::IdRef(uint_ty))
            .collect(),
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
    for member in 0..crate::reflect::KERNEL_DISPATCH_PUSH_CONSTANT_SIZE / 4 {
        ctx.module.annotations.push(Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(block_ty),
                Operand::LiteralBit32(member),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(offset + member * 4),
            ],
        ));
    }
    let pointer_ty = ctx.ty_ptr(StorageClass::PushConstant, block_ty);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(pointer_ty),
        Some(var),
        vec![Operand::StorageClass(StorageClass::PushConstant)],
    ));
    ctx.module.debug_names.push(Instruction::new(
        Op::Name,
        None,
        None,
        vec![
            Operand::IdRef(var),
            Operand::LiteralString("metal2vulkan_dispatch_region".into()),
        ],
    ));
    ctx.interface.push(var);
    *slot = Some(var);
    var
}

pub(in crate::passes) fn load_kernel_dispatch_component(
    ctx: &mut Ctx,
    instructions: &mut Vec<Instruction>,
    var: Word,
    member: u32,
) -> Word {
    let uint_ty = ctx.ty_uint();
    let pointer_ty = ctx.ty_ptr(StorageClass::PushConstant, uint_ty);
    let member_index = ctx.const_uint(member);
    let pointer = ctx.module.fresh_id();
    instructions.push(Instruction::new(
        Op::AccessChain,
        Some(pointer_ty),
        Some(pointer),
        vec![Operand::IdRef(var), Operand::IdRef(member_index)],
    ));
    let value = ctx.module.fresh_id();
    instructions.push(Instruction::new(
        Op::Load,
        Some(uint_ty),
        Some(value),
        vec![Operand::IdRef(pointer)],
    ));
    value
}

/// Materialize a selected three-component field in the AIR parameter's integer shape.
pub(in crate::passes) fn materialize_kernel_dispatch_field(
    ctx: &mut Ctx,
    instructions: &mut Vec<Instruction>,
    var: Word,
    first_member: u32,
    out_ty: Word,
    lanes: u32,
) -> Result<Word, String> {
    if !(1..=3).contains(&lanes)
        || first_member + lanes > crate::reflect::KERNEL_DISPATCH_PUSH_CONSTANT_SIZE / 4
    {
        return Err(format!(
            "kernel dispatch field has unsupported member {first_member} and lane count {lanes}"
        ));
    }
    let uint_ty = ctx.ty_uint();
    let components = (0..lanes)
        .map(|lane| load_kernel_dispatch_component(ctx, instructions, var, first_member + lane))
        .collect::<Vec<_>>();
    let value = if lanes == 1 {
        components[0]
    } else {
        let vector_ty = ctx.ty_vec_uint(lanes);
        let vector = ctx.module.fresh_id();
        instructions.push(Instruction::new(
            Op::CompositeConstruct,
            Some(vector_ty),
            Some(vector),
            components.into_iter().map(Operand::IdRef).collect(),
        ));
        vector
    };
    let value_ty = if lanes == 1 {
        uint_ty
    } else {
        ctx.ty_vec_uint(lanes)
    };
    if value_ty == out_ty {
        return Ok(value);
    }
    let converted = ctx.module.fresh_id();
    instructions.push(Instruction::new(
        Op::UConvert,
        Some(out_ty),
        Some(converted),
        vec![Operand::IdRef(value)],
    ));
    Ok(converted)
}
