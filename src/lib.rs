use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::Type;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature};
use cranelift_codegen::settings;
use cranelift_codegen::settings::Configurable;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;
use pgrx::PgMemoryContexts::{CurrentMemoryContext, TopMemoryContext};
use pgrx::{pg_guard, pg_sys};
use std::mem::{offset_of, size_of, transmute};
use std::os::raw::c_int;
use std::ptr;
use std::time::Instant;

mod sys;

pgrx::pg_module_magic!();

const TRUSTED: MemFlags = MemFlags::trusted(); // TODO: Is this assumption of alignment correct?
const DATUM_SIZE: i32 = size_of::<pg_sys::Datum>() as i32;
const BOOL_SIZE: i32 = size_of::<bool>() as i32;
const PTR_SIZE: i32 = size_of::<*const bool>() as i32;

#[repr(C)]
struct JitContext {
    base: sys::JitContext,
    builder_ctx: FunctionBuilderContext,
    module: JITModule,
    ctx: cranelift_codegen::Context,
    make_ro_fn: cranelift_module::FuncId,
}

struct CompiledExprState {
    func: pg_sys::ExprStateEvalFunc,
}

impl JitContext {
    fn new(jit_flags: c_int) -> Self {
        // TODO: Check flags
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap(); // TODO: Control via GUC?
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        let mut module = JITModule::new(builder);

        let datum_type = Type::int_with_byte_size(DATUM_SIZE as u16).unwrap();

        let mut make_ro_sig = Signature::new(module.isa().default_call_conv());
        make_ro_sig.returns.push(AbiParam::new(datum_type));
        make_ro_sig.params.push(AbiParam::new(datum_type));

        let make_ro_fn = module
            .declare_function(
                "MakeExpandedObjectReadOnlyInternal",
                cranelift_module::Linkage::Local,
                &make_ro_sig,
            )
            .unwrap();

        Self {
            base: sys::JitContext {
                flags: jit_flags,
                resowner: unsafe { pg_sys::CurrentResourceOwner },
                instr: sys::JitInstrumentation {
                    created_functions: 0,
                    generation_counter: pg_sys::instr_time { ticks: 0 },
                    inlining_counter: pg_sys::instr_time { ticks: 0 },
                    optimization_counter: pg_sys::instr_time { ticks: 0 },
                    emission_counter: pg_sys::instr_time { ticks: 0 },
                },
            },
            builder_ctx: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
            make_ro_fn,
        }
    }

    unsafe fn compile_expr(&mut self, state: &mut pg_sys::ExprState) -> bool {
        let start = Instant::now();

        let datum_type = Type::int_with_byte_size(DATUM_SIZE as u16).unwrap();
        let bool_type = Type::int_with_byte_size(BOOL_SIZE as u16).unwrap();
        let ptr_type = Type::int_with_byte_size(PTR_SIZE as u16).unwrap();

        self.ctx
            .func
            .signature
            .returns
            .push(AbiParam::new(datum_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // state
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // econtext
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // isnullp

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

        let mut v0sig = Signature::new(self.module.isa().default_call_conv());
        v0sig.returns.push(AbiParam::new(ptr_type));
        v0sig.params.push(AbiParam::new(ptr_type));
        let v0sig = builder.import_signature(v0sig);

        let make_ro_fn = self
            .module
            .declare_func_in_func(self.make_ro_fn, builder.func);

        let init_block = builder.create_block();
        let blocks = (0..state.steps_len)
            .map(|_| builder.create_block())
            .collect::<Vec<_>>();

        builder.append_block_params_for_function_params(init_block);
        builder.switch_to_block(init_block);

        let param_state = builder.block_params(init_block)[0];
        let _param_econtext = builder.block_params(init_block)[1];
        let param_isnull = builder.block_params(init_block)[2];

        let bool_false = builder.ins().iconst(bool_type, 0);
        let bool_true = builder.ins().iconst(bool_type, 1);
        let datum_false = builder.ins().iconst(datum_type, 0);
        let datum_true = builder.ins().iconst(datum_type, 1);

        let tmpvalue_off = builder
            .ins()
            .iconst(ptr_type, offset_of!(pg_sys::ExprState, resvalue) as i64);
        let tmpnull_off = builder
            .ins()
            .iconst(ptr_type, offset_of!(pg_sys::ExprState, resnull) as i64);
        let p_tmpvalue = builder.ins().iadd(param_state, tmpvalue_off);
        let p_tmpnull = builder.ins().iadd(param_state, tmpnull_off);
        let p_resultslot = builder.ins().load(
            ptr_type,
            TRUSTED,
            param_state,
            offset_of!(pg_sys::ExprState, resultslot) as i32,
        );

        builder.ins().jump(blocks[0], &[]);

        builder.seal_block(init_block);

        let mut all = true;

        for i in 0..(state.steps_len as usize) {
            let step = state.steps.add(i);
            let opcode = (*step).opcode as u32;

            builder.switch_to_block(blocks[i]);

            match opcode {
                pg_sys::ExprEvalOp_EEOP_DONE => {
                    let tmpvalue = builder.ins().load(datum_type, TRUSTED, p_tmpvalue, 0);
                    let tmpnull = builder.ins().load(bool_type, TRUSTED, p_tmpnull, 0);

                    builder.ins().store(TRUSTED, tmpnull, param_isnull, 0);

                    builder.ins().return_(&[tmpvalue]);
                }
                //pg_sys::ExprEvalOp_EEOP_INNER_FETCHSOME => (),
                //pg_sys::ExprEvalOp_EEOP_OUTER_FETCHSOME => (),
                //pg_sys::ExprEvalOp_EEOP_SCAN_FETCHSOME => (),
                //pg_sys::ExprEvalOp_EEOP_INNER_VAR => (),
                //pg_sys::ExprEvalOp_EEOP_OUTER_VAR => (),
                //pg_sys::ExprEvalOp_EEOP_SCAN_VAR => (),
                //pg_sys::ExprEvalOp_EEOP_INNER_SYSVAR => (),
                //pg_sys::ExprEvalOp_EEOP_OUTER_SYSVAR => (),
                //pg_sys::ExprEvalOp_EEOP_SCAN_SYSVAR => (),
                //pg_sys::ExprEvalOp_EEOP_WHOLEROW => (),
                //pg_sys::ExprEvalOp_EEOP_ASSIGN_INNER_VAR => (),
                //pg_sys::ExprEvalOp_EEOP_ASSIGN_OUTER_VAR => (),
                //pg_sys::ExprEvalOp_EEOP_ASSIGN_SCAN_VAR => (),
                pg_sys::ExprEvalOp_EEOP_ASSIGN_TMP | pg_sys::ExprEvalOp_EEOP_ASSIGN_TMP_MAKE_RO => {
                    let resultnum = (*step).d.assign_tmp.resultnum;

                    let tmpvalue = builder.ins().load(datum_type, TRUSTED, p_tmpvalue, 0);
                    let tmpnull = builder.ins().load(bool_type, TRUSTED, p_tmpnull, 0);

                    let p_tts_values = builder.ins().load(
                        ptr_type,
                        TRUSTED,
                        p_resultslot,
                        offset_of!(pg_sys::TupleTableSlot, tts_values) as i32,
                    );
                    let p_tts_isnull = builder.ins().load(
                        ptr_type,
                        TRUSTED,
                        p_resultslot,
                        offset_of!(pg_sys::TupleTableSlot, tts_isnull) as i32,
                    );

                    builder
                        .ins()
                        .store(TRUSTED, tmpnull, p_tts_isnull, resultnum);

                    if opcode == pg_sys::ExprEvalOp_EEOP_ASSIGN_TMP_MAKE_RO {
                        let call_block = builder.create_block();

                        // TODO: Check explicitly for 1?
                        builder
                            .ins()
                            .brif(tmpnull, blocks[i + 1], &[], call_block, &[]);

                        builder.switch_to_block(call_block);

                        let call = builder.ins().call(make_ro_fn, &[tmpvalue]);
                        let retvalue = builder.inst_results(call)[0];

                        builder.ins().store(
                            TRUSTED,
                            retvalue,
                            p_tts_values,
                            resultnum * DATUM_SIZE,
                        );

                        builder.ins().jump(blocks[i + 1], &[]);

                        builder.seal_block(call_block);
                    } else {
                        builder.ins().store(
                            TRUSTED,
                            tmpvalue,
                            p_tts_values,
                            resultnum * DATUM_SIZE,
                        );

                        builder.ins().jump(blocks[i + 1], &[]);
                    }
                }
                pg_sys::ExprEvalOp_EEOP_CONST => {
                    let value = builder
                        .ins()
                        .iconst(datum_type, (*step).d.constval.value.value() as i64);
                    let isnull = builder
                        .ins()
                        .iconst(bool_type, (*step).d.constval.isnull as i64);

                    let p_resvalue = builder.ins().iconst(ptr_type, (*step).resvalue as i64);
                    let p_resnull = builder.ins().iconst(ptr_type, (*step).resnull as i64);

                    builder.ins().store(TRUSTED, value, p_resvalue, 0);
                    builder.ins().store(TRUSTED, isnull, p_resnull, 0);

                    builder.ins().jump(blocks[i + 1], &[]);
                }
                pg_sys::ExprEvalOp_EEOP_FUNCEXPR => {
                    let fcinfo = (*step).d.func.fcinfo_data;

                    let p_resvalue = builder.ins().iconst(ptr_type, (*step).resvalue as i64);
                    let p_resnull = builder.ins().iconst(ptr_type, (*step).resnull as i64);

                    let p_fcinfo = builder.ins().iconst(ptr_type, fcinfo as i64);
                    let fn_addr = builder
                        .ins()
                        .iconst(ptr_type, transmute::<_, i64>((*step).d.func.fn_addr));

                    builder.ins().store(
                        TRUSTED,
                        bool_false,
                        p_fcinfo,
                        offset_of!(pg_sys::FunctionCallInfoBaseData, isnull) as i32,
                    );

                    let call = builder.ins().call_indirect(v0sig, fn_addr, &[p_fcinfo]);
                    let retvalue = builder.inst_results(call)[0];

                    let retisnull = builder.ins().load(
                        bool_type,
                        TRUSTED,
                        p_fcinfo,
                        offset_of!(pg_sys::FunctionCallInfoBaseData, isnull) as i32,
                    );

                    builder.ins().store(TRUSTED, retvalue, p_resvalue, 0);
                    builder.ins().store(TRUSTED, retisnull, p_resnull, 0);

                    builder.ins().jump(blocks[i + 1], &[]);
                }
                pg_sys::ExprEvalOp_EEOP_FUNCEXPR_STRICT => {
                    let fcinfo = (*step).d.func.fcinfo_data;
                    let nargs = (*step).d.func.nargs as usize;

                    debug_assert!(nargs != 0, "argumentless strict functions are pointless");

                    let p_resvalue = builder.ins().iconst(ptr_type, (*step).resvalue as i64);
                    let p_resnull = builder.ins().iconst(ptr_type, (*step).resnull as i64);

                    builder.ins().store(TRUSTED, bool_true, p_resnull, 0);

                    let arg_blocks = (0..nargs + 1)
                        .map(|_| builder.create_block())
                        .collect::<Vec<_>>();

                    builder.ins().jump(arg_blocks[0], &[]);

                    for argno in 0..nargs {
                        builder.switch_to_block(arg_blocks[argno]);

                        let p_argisnull = builder.ins().iconst(
                            ptr_type,
                            (ptr::addr_of!((*fcinfo).args) as usize
                                + size_of::<pg_sys::NullableDatum>() * argno
                                + offset_of!(pg_sys::NullableDatum, isnull))
                                as i64,
                        );

                        // TODO: Check explicitly for 1?
                        let argisnull = builder.ins().load(bool_type, TRUSTED, p_argisnull, 0);

                        builder.ins().brif(
                            argisnull,
                            blocks[i + 1],
                            &[],
                            arg_blocks[argno + 1],
                            &[],
                        );

                        builder.seal_block(arg_blocks[argno]);
                    }

                    builder.switch_to_block(arg_blocks[nargs]);

                    let p_fcinfo = builder.ins().iconst(ptr_type, fcinfo as i64);
                    let fn_addr = builder
                        .ins()
                        .iconst(ptr_type, transmute::<_, i64>((*step).d.func.fn_addr));

                    builder.ins().store(
                        TRUSTED,
                        bool_false,
                        p_fcinfo,
                        offset_of!(pg_sys::FunctionCallInfoBaseData, isnull) as i32,
                    );

                    let call = builder.ins().call_indirect(v0sig, fn_addr, &[p_fcinfo]);
                    let retvalue = builder.inst_results(call)[0];

                    let retisnull = builder.ins().load(
                        bool_type,
                        TRUSTED,
                        p_fcinfo,
                        offset_of!(pg_sys::FunctionCallInfoBaseData, isnull) as i32,
                    );

                    builder.ins().store(TRUSTED, retvalue, p_resvalue, 0);
                    builder.ins().store(TRUSTED, retisnull, p_resnull, 0);

                    builder.ins().jump(blocks[i + 1], &[]);

                    builder.seal_block(arg_blocks[nargs]);
                }
                //pg_sys::ExprEvalOp_EEOP_FUNCEXPR_FUSAGE => (),
                //pg_sys::ExprEvalOp_EEOP_FUNCEXPR_STRICT_FUSAGE => (),
                //pg_sys::ExprEvalOp_EEOP_BOOL_AND_STEP_FIRST => (),
                //pg_sys::ExprEvalOp_EEOP_BOOL_AND_STEP => (),
                //pg_sys::ExprEvalOp_EEOP_BOOL_AND_STEP_LAST => (),
                //pg_sys::ExprEvalOp_EEOP_BOOL_OR_STEP_FIRST => (),
                //pg_sys::ExprEvalOp_EEOP_BOOL_OR_STEP => (),
                //pg_sys::ExprEvalOp_EEOP_BOOL_OR_STEP_LAST => (),
                //pg_sys::ExprEvalOp_EEOP_BOOL_NOT_STEP => (),
                pg_sys::ExprEvalOp_EEOP_QUAL => {
                    let jumpdone = (*step).d.qualexpr.jumpdone as usize;

                    let p_resvalue = builder.ins().iconst(ptr_type, (*step).resvalue as i64);
                    let p_resnull = builder.ins().iconst(ptr_type, (*step).resnull as i64);

                    let resvalue = builder.ins().load(datum_type, TRUSTED, p_resvalue, 0);
                    let resnull = builder.ins().load(bool_type, TRUSTED, p_resnull, 0);

                    // TODO: Check explicitly for 1?
                    let is_false = builder.ins().icmp(IntCC::Equal, resvalue, datum_false);
                    let null_or_false = builder.ins().bor(resnull, is_false);

                    let then_block = builder.create_block();

                    builder
                        .ins()
                        .brif(null_or_false, then_block, &[], blocks[i + 1], &[]);

                    builder.switch_to_block(then_block);

                    builder.ins().store(TRUSTED, datum_false, p_resvalue, 0);
                    builder.ins().store(TRUSTED, bool_false, p_resnull, 0);

                    builder.ins().jump(blocks[jumpdone], &[]);

                    builder.seal_block(then_block);
                }
                pg_sys::ExprEvalOp_EEOP_JUMP => {
                    let jumpdone = (*step).d.jump.jumpdone as usize;

                    builder.ins().jump(blocks[jumpdone], &[]);
                }
                //pg_sys::ExprEvalOp_EEOP_JUMP_IF_NULL => (),
                //pg_sys::ExprEvalOp_EEOP_JUMP_IF_NOT_NULL => (),
                //pg_sys::ExprEvalOp_EEOP_JUMP_IF_NOT_TRUE => (),
                pg_sys::ExprEvalOp_EEOP_NULLTEST_ISNULL => {
                    let p_resvalue = builder.ins().iconst(ptr_type, (*step).resvalue as i64);
                    let p_resnull = builder.ins().iconst(ptr_type, (*step).resnull as i64);

                    let resnull = builder.ins().load(bool_type, TRUSTED, p_resnull, 0);
                    let isnull = builder.ins().select(resnull, datum_true, datum_false);

                    builder.ins().store(TRUSTED, isnull, p_resvalue, 0);
                    builder.ins().store(TRUSTED, bool_false, p_resnull, 0);

                    builder.ins().jump(blocks[i + 1], &[]);
                }
                pg_sys::ExprEvalOp_EEOP_NULLTEST_ISNOTNULL => {
                    let p_resvalue = builder.ins().iconst(ptr_type, (*step).resvalue as i64);
                    let p_resnull = builder.ins().iconst(ptr_type, (*step).resnull as i64);

                    let resnull = builder.ins().load(bool_type, TRUSTED, p_resnull, 0);
                    let isnotnull = builder.ins().select(resnull, datum_false, datum_true);

                    builder.ins().store(TRUSTED, isnotnull, p_resvalue, 0);
                    builder.ins().store(TRUSTED, bool_false, p_resnull, 0);

                    builder.ins().jump(blocks[i + 1], &[]);
                }
                //pg_sys::ExprEvalOp_EEOP_NULLTEST_ROWISNULL => (),
                //pg_sys::ExprEvalOp_EEOP_NULLTEST_ROWISNOTNULL => (),
                pg_sys::ExprEvalOp_EEOP_BOOLTEST_IS_TRUE => {
                    let p_resvalue = builder.ins().iconst(ptr_type, (*step).resvalue as i64);
                    let p_resnull = builder.ins().iconst(ptr_type, (*step).resnull as i64);

                    let resnull = builder.ins().load(bool_type, TRUSTED, p_resnull, 0);

                    let then_block = builder.create_block();

                    // TODO: Check explicitly for 1?
                    builder
                        .ins()
                        .brif(resnull, then_block, &[], blocks[i + 1], &[]);

                    builder.switch_to_block(then_block);

                    builder.ins().store(TRUSTED, datum_false, p_resvalue, 0);
                    builder.ins().store(TRUSTED, bool_false, p_resnull, 0);

                    builder.ins().jump(blocks[i + 1], &[]);

                    builder.seal_block(then_block);
                }
                //pg_sys::ExprEvalOp_EEOP_BOOLTEST_IS_NOT_TRUE => (),
                //pg_sys::ExprEvalOp_EEOP_BOOLTEST_IS_FALSE => (),
                //pg_sys::ExprEvalOp_EEOP_BOOLTEST_IS_NOT_FALSE => (),
                //pg_sys::ExprEvalOp_EEOP_PARAM_EXEC => (),
                //pg_sys::ExprEvalOp_EEOP_PARAM_EXTERN => (),
                //pg_sys::ExprEvalOp_EEOP_PARAM_CALLBACK => (),
                //pg_sys::ExprEvalOp_EEOP_CASE_TESTVAL => (),
                //pg_sys::ExprEvalOp_EEOP_MAKE_READONLY => (),
                //pg_sys::ExprEvalOp_EEOP_IOCOERCE => (),
                //pg_sys::ExprEvalOp_EEOP_DISTINCT => (),
                //pg_sys::ExprEvalOp_EEOP_NOT_DISTINCT => (),
                //pg_sys::ExprEvalOp_EEOP_NULLIF => (),
                //pg_sys::ExprEvalOp_EEOP_SQLVALUEFUNCTION => (),
                //pg_sys::ExprEvalOp_EEOP_CURRENTOFEXPR => (),
                //pg_sys::ExprEvalOp_EEOP_NEXTVALUEEXPR => (),
                //pg_sys::ExprEvalOp_EEOP_ARRAYEXPR => (),
                //pg_sys::ExprEvalOp_EEOP_ARRAYCOERCE => (),
                //pg_sys::ExprEvalOp_EEOP_ROW => (),
                //pg_sys::ExprEvalOp_EEOP_ROWCOMPARE_STEP => (),
                //pg_sys::ExprEvalOp_EEOP_ROWCOMPARE_FINAL => (),
                //pg_sys::ExprEvalOp_EEOP_MINMAX => (),
                //pg_sys::ExprEvalOp_EEOP_FIELDSELECT => (),
                //pg_sys::ExprEvalOp_EEOP_FIELDSTORE_DEFORM => (),
                //pg_sys::ExprEvalOp_EEOP_FIELDSTORE_FORM => (),
                //pg_sys::ExprEvalOp_EEOP_SBSREF_SUBSCRIPTS => (),
                //pg_sys::ExprEvalOp_EEOP_SBSREF_OLD => (),
                //pg_sys::ExprEvalOp_EEOP_SBSREF_ASSIGN => (),
                //pg_sys::ExprEvalOp_EEOP_SBSREF_FETCH => (),
                //pg_sys::ExprEvalOp_EEOP_DOMAIN_TESTVAL => (),
                //pg_sys::ExprEvalOp_EEOP_DOMAIN_NOTNULL => (),
                //pg_sys::ExprEvalOp_EEOP_DOMAIN_CHECK => (),
                //pg_sys::ExprEvalOp_EEOP_CONVERT_ROWTYPE => (),
                //pg_sys::ExprEvalOp_EEOP_SCALARARRAYOP => (),
                //pg_sys::ExprEvalOp_EEOP_HASHED_SCALARARRAYOP => (),
                //pg_sys::ExprEvalOp_EEOP_XMLEXPR => (),
                //pg_sys::ExprEvalOp_EEOP_JSON_CONSTRUCTOR => (),
                //pg_sys::ExprEvalOp_EEOP_IS_JSON => (),
                //pg_sys::ExprEvalOp_EEOP_AGGREF => (),
                //pg_sys::ExprEvalOp_EEOP_GROUPING_FUNC => (),
                //pg_sys::ExprEvalOp_EEOP_WINDOW_FUNC => (),
                //pg_sys::ExprEvalOp_EEOP_SUBPLAN => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_STRICT_DESERIALIZE => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_DESERIALIZE => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_STRICT_INPUT_CHECK_ARGS => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_STRICT_INPUT_CHECK_NULLS => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_PLAIN_PERGROUP_NULLCHECK => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYVAL => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_PLAIN_TRANS_STRICT_BYVAL => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_PLAIN_TRANS_BYVAL => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_PLAIN_TRANS_INIT_STRICT_BYREF => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_PLAIN_TRANS_STRICT_BYREF => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_PLAIN_TRANS_BYREF => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_PRESORTED_DISTINCT_SINGLE => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_PRESORTED_DISTINCT_MULTI => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_ORDERED_TRANS_DATUM => (),
                //pg_sys::ExprEvalOp_EEOP_AGG_ORDERED_TRANS_TUPLE => (),
                pg_sys::ExprEvalOp_EEOP_LAST => debug_assert!(false, "unexpected EEOP_LAST"),
                opcode => {
                    all = false;
                    pgrx::warning!("Unsupported opcode {}", opcode);
                }
            }

            builder.seal_block(blocks[i]);
        }

        builder.finalize();

        if all {
            self.base.instr.generation_counter.ticks += (Instant::now() - start).as_nanos() as i64;
            self.base.instr.created_functions += 1;

            pgrx::notice!("{}", self.ctx.func.display());

            let start = Instant::now();

            let id = self
                .module
                .declare_anonymous_function(&self.ctx.func.signature)
                .unwrap();
            self.module
                .define_function(id, &mut self.ctx)
                .map_err(|e| e.to_string())
                .unwrap();
            self.module.finalize_definitions().unwrap();

            let func = self.module.get_finalized_function(id);
            state.evalfunc = Some(wrapper);

            let jit_ctx = CurrentMemoryContext.palloc_struct();
            ptr::write(
                jit_ctx,
                CompiledExprState {
                    func: transmute(func),
                },
            );
            state.evalfunc_private = jit_ctx as *mut core::ffi::c_void;

            self.base.instr.emission_counter.ticks += (Instant::now() - start).as_nanos() as i64;

            pgrx::notice!("{}", self.ctx.func.display());
        }

        self.module.clear_context(&mut self.ctx);

        all
    }
}

// TODO: This wrapper is only here to make things easier to debug.
unsafe extern "C" fn wrapper(
    state: *mut pg_sys::ExprState,
    econtext: *mut pg_sys::ExprContext,
    isnull: *mut bool,
) -> pg_sys::Datum {
    let func = (*((*state).evalfunc_private as *const CompiledExprState))
        .func
        .unwrap();

    pgrx::notice!("calling compiled expression {:#x}", func as usize);
    let ret = func(state, econtext, isnull);
    pgrx::notice!("ret {:#x}", ret.value() as u64);
    ret
}

#[pg_guard]
unsafe extern "C" fn compile_expr(state: *mut pg_sys::ExprState) -> bool {
    let parent = (*state).parent;

    debug_assert!(
        !parent.is_null(),
        "we only support expressions with a parent"
    );

    let context;

    if (*(*parent).state).es_jit.is_null() {
        context = TopMemoryContext.palloc_struct();
        ptr::write(context, JitContext::new((*(*parent).state).es_jit_flags));

        sys::ResourceOwnerEnlargeJIT((*context).base.resowner);
        sys::ResourceOwnerRememberJIT((*context).base.resowner, context.into());

        (*(*parent).state).es_jit = context as *mut pg_sys::JitContext;
    } else {
        context = (*(*parent).state).es_jit as *mut JitContext;
    }

    (*context).compile_expr(&mut *state)
}

#[pg_guard]
unsafe extern "C" fn release_context(context: *mut pg_sys::JitContext) {
    let context = context as *mut JitContext;
    ptr::drop_in_place(context);
}

unsafe extern "C" fn reset_after_error() {}

#[allow(non_snake_case)]
#[no_mangle]
pub unsafe extern "C" fn _PG_jit_provider_init(cb: *mut sys::JitProviderCallbacks) {
    (*cb).reset_after_error = Some(reset_after_error);
    (*cb).release_context = Some(release_context);
    (*cb).compile_expr = Some(compile_expr);
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_pg_craneliftjit() {
        Spi::run("SELECT 69, 420;").unwrap();
        Spi::run("SELECT pg_jit_available();").unwrap();
        Spi::run("SELECT 1 + x FROM generate_series(1, 100) x").unwrap();
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec!["jit_provider=pg_craneliftjit", "jit_above_cost=0"]
    }
}
