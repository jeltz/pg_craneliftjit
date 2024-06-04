use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::Type;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings;
use cranelift_codegen::settings::Configurable;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;
use pgrx::prelude::*;
use std::cell::OnceCell;
use std::cell::RefCell;
use std::mem::offset_of;

pgrx::pg_module_magic!();

const TRUSTED: MemFlags = MemFlags::trusted(); // TODO: Is this assumption of alignment correct?
const DATUM_SIZE: i32 = std::mem::size_of::<pg_sys::Datum>() as i32;
const BOOL_SIZE: i32 = std::mem::size_of::<bool>() as i32;
const PTR_SIZE: i32 = std::mem::size_of::<*const bool>() as i32;

struct Compiler {
    builder_ctx: FunctionBuilderContext,
    module: JITModule,
    ctx: cranelift_codegen::Context,
}

// TODO: What should the context contain?
struct JitContext {
    func: pg_sys::ExprStateEvalFunc,
}

impl Compiler {
    fn new() -> Self {
        // TODO: Check flags
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        let module = JITModule::new(builder);

        Self {
            builder_ctx: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
        }
    }

    unsafe fn compile_expr(&mut self, state: &mut pg_sys::ExprState) -> bool {
        let datum_type = Type::int_with_byte_size(DATUM_SIZE as u16).unwrap();
        let bool_type = Type::int_with_byte_size(BOOL_SIZE as u16).unwrap();
        let ptr_type = Type::int_with_byte_size(PTR_SIZE as u16).unwrap();

        self.module.clear_context(&mut self.ctx); // TODO: Move to after?

        self.ctx
            .func
            .signature
            .returns
            .push(AbiParam::new(datum_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // state
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // econtext
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // isnullp

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

        let init_block = builder.create_block();
        let blocks = (0..state.steps_len)
            .map(|_| builder.create_block())
            .collect::<Vec<_>>();

        builder.append_block_params_for_function_params(init_block);
        builder.switch_to_block(init_block);

        let param_state = builder.block_params(init_block)[0];
        let _param_econtext = builder.block_params(init_block)[1];
        let param_isnull = builder.block_params(init_block)[2];
        let tmpvalue_off = builder.ins().iconst(ptr_type, offset_of!(pg_sys::ExprState, resvalue) as i64);
        let tmpnull_off = builder.ins().iconst(ptr_type, offset_of!(pg_sys::ExprState, resnull) as i64);
        let p_tmpvalue = builder
            .ins()
            .iadd(param_state, tmpvalue_off);
        let p_tmpnull = builder
            .ins()
            .iadd(param_state, tmpnull_off);
        let p_resultslot = builder
            .ins()
            .load(ptr_type, TRUSTED, param_state, offset_of!(pg_sys::ExprState, resultslot) as i32);

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
                pg_sys::ExprEvalOp_EEOP_SCAN_FETCHSOME => {
                    all = false;
                }
                //pg_sys::ExprEvalOp_EEOP_INNER_VAR => (),
                //pg_sys::ExprEvalOp_EEOP_OUTER_VAR => (),
                pg_sys::ExprEvalOp_EEOP_SCAN_VAR => {
                    all = false;
                }
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

                    let p_tts_values = builder
                        .ins()
                        .load(ptr_type, TRUSTED, p_resultslot, offset_of!(pg_sys::TupleTableSlot, tts_values) as i32);
                    let p_tts_isnull = builder
                        .ins()
                        .load(ptr_type, TRUSTED, p_resultslot, offset_of!(pg_sys::TupleTableSlot, tts_isnull) as i32);

                    if opcode == pg_sys::ExprEvalOp_EEOP_ASSIGN_TMP_MAKE_RO {
                        // TODO
                        all = false;

                        builder.ins().store(
                            TRUSTED,
                            tmpvalue,
                            p_tts_values,
                            resultnum * DATUM_SIZE,
                        );
                    } else {
                        builder.ins().store(
                            TRUSTED,
                            tmpvalue,
                            p_tts_values,
                            resultnum * DATUM_SIZE,
                        );
                    }
                    builder
                        .ins()
                        .store(TRUSTED, tmpnull, p_tts_isnull, resultnum);

                    builder.ins().jump(blocks[i + 1], &[]);
                }
                pg_sys::ExprEvalOp_EEOP_CONST => {
                    let value = builder
                        .ins()
                        .iconst(datum_type, (*step).d.constval.value.value() as i64);
                    let isnull = builder
                        .ins()
                        .iconst(bool_type, (*step).d.constval.isnull as i64);

                    let p_resvalue = builder
                        .ins()
                        .iconst(ptr_type, (*step).resvalue as i64);
                    let p_resnull = builder
                        .ins()
                        .iconst(ptr_type, (*step).resnull as i64);

                    builder.ins().store(TRUSTED, value, p_resvalue, 0);
                    builder.ins().store(TRUSTED, isnull, p_resnull, 0);

                    builder.ins().jump(blocks[i + 1], &[]);
                }
                pg_sys::ExprEvalOp_EEOP_FUNCEXPR => {
                    //info!("fn_oid: {}", (*(*step).d.func.finfo).fn_oid.as_u32());
                    all = false;
                }
                pg_sys::ExprEvalOp_EEOP_FUNCEXPR_STRICT => {
                    //info!("fn_oid: {}", (*(*step).d.func.finfo).fn_oid.as_u32());
                    all = false;
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

                    let p_resvalue = builder
                        .ins()
                        .iconst(ptr_type, (*step).resvalue as i64);
                    let p_resnull = builder
                        .ins()
                        .iconst(ptr_type, (*step).resnull as i64);

                    let resvalue = builder.ins().load(datum_type, TRUSTED, p_resvalue, 0);
                    let resnull = builder.ins().load(bool_type, TRUSTED, p_resnull, 0);

                    let datum_false = builder.ins().iconst(datum_type, 0);
                    let is_false = builder.ins().icmp(IntCC::Equal, resvalue, datum_false);
                    let null_or_false = builder.ins().bor(resnull, is_false);

                    let then_block = builder.create_block();

                    builder
                        .ins()
                        .brif(null_or_false, then_block, &[], blocks[i + 1], &[]);

                    builder.switch_to_block(then_block);

                    let bool_false = builder.ins().iconst(bool_type, 0);
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
                //pg_sys::ExprEvalOp_EEOP_NULLTEST_ISNULL => (),
                //pg_sys::ExprEvalOp_EEOP_NULLTEST_ISNOTNULL => (),
                //pg_sys::ExprEvalOp_EEOP_NULLTEST_ROWISNULL => (),
                //pg_sys::ExprEvalOp_EEOP_NULLTEST_ROWISNOTNULL => (),
                //pg_sys::ExprEvalOp_EEOP_BOOLTEST_IS_TRUE => (),
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
                opcode => panic!("Unsupported opcode {}", opcode),
            }

            builder.seal_block(blocks[i]);
        }

        builder.finalize();

        if all {
            println!("{}", self.ctx.func.display());

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
            let jit_ctx = Box::new(JitContext {
                func: std::mem::transmute(func),
            });
            state.evalfunc_private = Box::into_raw(jit_ctx) as *mut core::ffi::c_void;
        }

        all
    }

    fn release_context(&mut self, _context: &JitContext) {
        // TODO
    }

    fn reset_after_error(&mut self) {
        println!("Reset!");
        self.builder_ctx = FunctionBuilderContext::new();
        // TODO: Deallocate functions
        // TODO: Clear self.ctx?
        // TODO: evalfunc_private?
    }
}

// TODO: This wrapper is only here to make things easier to debug.
unsafe extern "C" fn wrapper(state: *mut pg_sys::ExprState, econtext: *mut pg_sys::ExprContext, isnull: *mut bool) -> pg_sys::Datum {
    let func = (*((*state).evalfunc_private as *const JitContext)).func.unwrap();

    notice!("calling compiled expression {:#x}", func as usize);
    let ret = func(state, econtext, isnull);
    notice!("ret {:#x}", ret.value() as u64);
    ret
}

static COMPILER: CompilerWrapper = CompilerWrapper::new();

// Trust that PostgreSQL is single-threaded
struct CompilerWrapper(OnceCell<RefCell<Compiler>>);

impl CompilerWrapper {
    const fn new() -> Self {
        Self(OnceCell::new())
    }

    unsafe fn get_or_init(&self) -> &RefCell<Compiler> {
        self.0.get_or_init(|| RefCell::new(Compiler::new()))
    }
}

unsafe impl Send for CompilerWrapper {}
unsafe impl Sync for CompilerWrapper {}

#[pg_guard]
unsafe extern "C" fn compile_expr(state: *mut pg_sys::ExprState) -> bool {
    let mut compiler = COMPILER.get_or_init().borrow_mut();
    compiler.compile_expr(&mut *state)
}

#[pg_guard]
unsafe extern "C" fn release_context(context: *mut pg_sys::JitContext) {
    let mut compiler = COMPILER.get_or_init().borrow_mut();
    let context = Box::from_raw(context as *mut JitContext);
    compiler.release_context(&context);
}

#[pg_guard]
unsafe extern "C" fn reset_after_error() {
    let mut compiler = COMPILER.get_or_init().borrow_mut();
    compiler.reset_after_error();
}

#[repr(C)]
pub struct JitProviderCallbacks {
    reset_after_error: Option<unsafe extern "C" fn()>,
    release_context: Option<unsafe extern "C" fn(context: *mut pg_sys::JitContext)>,
    compile_expr: Option<unsafe extern "C" fn(state: *mut pg_sys::ExprState) -> bool>,
}

#[allow(non_snake_case)]
#[no_mangle]
pub unsafe extern "C" fn _PG_jit_provider_init(cb: *mut JitProviderCallbacks) {
    // TODO: Remove debug logging?
    let mut builder = env_logger::Builder::new();
    builder.target(env_logger::Target::Stdout);
    //builder.filter_level(log::LevelFilter::Debug);
    builder.init();

    (*cb).reset_after_error = Some(reset_after_error);
    (*cb).release_context = Some(release_context);
    (*cb).compile_expr = Some(compile_expr);
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_hello_pg_jit_cranelift() {
        Spi::run("SELECT 69, 420;").unwrap();
        Spi::run("SELECT pg_jit_available();").unwrap();
        Spi::run("SELECT 1 + x FROM generate_series(1, 100) x").unwrap();
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec!["jit_provider=pg_jit_cranelift", "jit_above_cost=0"]
    }
}
