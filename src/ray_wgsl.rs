//! Turn ray-query statements into ordinary calls so Naga's WGSL backend can
//! print them.
//!
//! The backend has no spelling for [`Statement::RayQuery`] and panics on one.
//! A ray query is a builtin call in WGSL (`rayQueryInitialize`, and the rest),
//! which the backend *can* print, so this rewrites the module into those calls,
//! lets the backend run, and then deletes the empty functions it had to invent
//! for the calls to name. What is left is the builtin call, and an
//! `enable wgpu_ray_query` so a WGSL frontend reads it back as a ray query.

use naga::{
    Arena, Block, Expression, Function, FunctionArgument, FunctionResult, Handle, Span, Statement,
    Type, TypeInner,
};

struct Ops {
    initialize: Handle<Function>,
    proceed: Handle<Function>,
    get_committed: Handle<Function>,
    get_candidate: Handle<Function>,
    confirm: Handle<Function>,
    generate: Handle<Function>,
    terminate: Handle<Function>,
}

impl Ops {
    fn names() -> &'static [&'static str] {
        &[
            "rayQueryInitialize",
            "rayQueryProceed",
            "rayQueryGetCommittedIntersection",
            "rayQueryGetCandidateIntersection",
            "rayQueryConfirmIntersection",
            "rayQueryGenerateIntersection",
            "rayQueryTerminate",
        ]
    }
}

/// Rewrite `module` so [`crate::to_wgsl`] can print its ray queries.
///
/// The module must already be valid. This adds one empty function per builtin
/// and points the ray-query statements at them. The stand-ins go at the front
/// of the function arena: Naga only lets a function call one declared earlier,
/// and the ray query may live in a plain function rather than an entry point.
pub(crate) fn rewrite(module: &mut naga::Module) -> Result<(), String> {
    let dummies = dummy_functions(module)?;
    let mut old = std::mem::take(&mut module.functions);
    let mut functions = Arena::new();
    let mut dummy_handles = Vec::with_capacity(dummies.len());
    for dummy in dummies {
        dummy_handles.push(functions.append(dummy, Span::UNDEFINED));
    }
    let mut relocated = Vec::new();
    for (_, function, span) in old.drain() {
        relocated.push(functions.append(function, span));
    }
    for (_, function) in functions.iter_mut() {
        remap_calls(function, &relocated);
    }
    module.functions = functions;
    let ops = Ops {
        initialize: dummy_handles[0],
        proceed: dummy_handles[1],
        get_committed: dummy_handles[2],
        get_candidate: dummy_handles[3],
        confirm: dummy_handles[4],
        generate: dummy_handles[5],
        terminate: dummy_handles[6],
    };
    for (_, function) in module.functions.iter_mut() {
        rewrite_function(function, &ops);
    }
    for entry in &mut module.entry_points {
        remap_calls(&mut entry.function, &relocated);
        rewrite_function(&mut entry.function, &ops);
    }
    Ok(())
}

/// Existing calls name functions by their old handles. `relocated[i]` is where
/// the function that used to be at index `i` lives now.
fn remap_calls(function: &mut Function, relocated: &[Handle<Function>]) {
    for (_, expr) in function.expressions.iter_mut() {
        if let Expression::CallResult(callee) = expr {
            *callee = relocated[callee.index()];
        }
    }
    remap_block(&mut function.body, relocated);
}

fn remap_block(block: &mut Block, relocated: &[Handle<Function>]) {
    for stmt in block.iter_mut() {
        match stmt {
            Statement::Call { function, .. } => *function = relocated[function.index()],
            Statement::If { accept, reject, .. } => {
                remap_block(accept, relocated);
                remap_block(reject, relocated);
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                remap_block(body, relocated);
                remap_block(continuing, relocated);
            }
            Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    remap_block(&mut case.body, relocated);
                }
            }
            Statement::Block(inner) => remap_block(inner, relocated),
            _ => {}
        }
    }
}

fn rewrite_function(function: &mut Function, ops: &Ops) {
    let mut body = std::mem::take(&mut function.body);
    rewrite_block(&mut body, function, ops);
    function.body = body;
}

/// Delete the invented functions from backend output and request the extension
/// a frontend needs in order to read the calls as ray queries.
pub(crate) fn finish_text(wgsl: String) -> String {
    let stripped = strip_functions(&wgsl, Ops::names());
    // `RayDesc` and `RayIntersection` are predeclared once the extension is
    // enabled. Emitting our own copies makes the builtin's return type and the
    // variable's type two different structs that happen to share a name.
    let stripped = strip_structs(&stripped, &["RayDesc", "RayIntersection"]);
    if stripped.contains("enable wgpu_ray_query;") {
        stripped
    } else {
        format!("enable wgpu_ray_query;\n{stripped}")
    }
}

fn dummy_functions(module: &mut naga::Module) -> Result<Vec<Function>, String> {
    let ray_query = find_type(module, |inner| matches!(inner, TypeInner::RayQuery { .. }))
        .ok_or("ray query shader has no `ray_query` type")?;
    let acceleration = find_type(module, |inner| {
        matches!(inner, TypeInner::AccelerationStructure { .. })
    })
    .ok_or("ray query shader has no `acceleration_structure` type")?;
    let query_ptr = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Pointer {
                base: ray_query,
                space: naga::AddressSpace::Function,
            },
        },
        Span::UNDEFINED,
    );
    let desc = module.generate_ray_desc_type();
    let intersection = module.generate_ray_intersection_type();
    let bool_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(naga::Scalar::BOOL),
        },
        Span::UNDEFINED,
    );
    let f32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(naga::Scalar::F32),
        },
        Span::UNDEFINED,
    );

    let query_arg = |name: &str| FunctionArgument {
        name: Some(name.into()),
        ty: query_ptr,
        binding: None,
    };
    Ok(vec![
        stub(
            "rayQueryInitialize",
            vec![
                query_arg("query"),
                FunctionArgument {
                    name: Some("acceleration_structure".into()),
                    ty: acceleration,
                    binding: None,
                },
                FunctionArgument {
                    name: Some("desc".into()),
                    ty: desc,
                    binding: None,
                },
            ],
            None,
        ),
        stub("rayQueryProceed", vec![query_arg("query")], Some(bool_ty)),
        stub(
            "rayQueryGetCommittedIntersection",
            vec![query_arg("query")],
            Some(intersection),
        ),
        stub(
            "rayQueryGetCandidateIntersection",
            vec![query_arg("query")],
            Some(intersection),
        ),
        stub(
            "rayQueryConfirmIntersection",
            vec![query_arg("query")],
            None,
        ),
        stub(
            "rayQueryGenerateIntersection",
            vec![
                query_arg("query"),
                FunctionArgument {
                    name: Some("hit_t".into()),
                    ty: f32_ty,
                    binding: None,
                },
            ],
            None,
        ),
        stub("rayQueryTerminate", vec![query_arg("query")], None),
    ])
}

fn stub(name: &str, arguments: Vec<FunctionArgument>, result: Option<Handle<Type>>) -> Function {
    let mut function = Function {
        name: Some(name.into()),
        arguments,
        result: result.map(|ty| FunctionResult { ty, binding: None }),
        local_variables: Arena::new(),
        expressions: Arena::new(),
        named_expressions: Default::default(),
        body: Block::new(),
        diagnostic_filter_leaf: None,
    };
    if let Some(ty) = result {
        // `ZeroValue` is in scope without an `Emit`; emitting it is rejected.
        let value = function
            .expressions
            .append(Expression::ZeroValue(ty), Span::UNDEFINED);
        let mut body = Block::new();
        body.push(Statement::Return { value: Some(value) }, Span::UNDEFINED);
        function.body = body;
    }
    function
}

fn find_type(module: &naga::Module, pred: impl Fn(&TypeInner) -> bool) -> Option<Handle<Type>> {
    module
        .types
        .iter()
        .find_map(|(handle, ty)| pred(&ty.inner).then_some(handle))
}

fn rewrite_block(block: &mut Block, function: &mut Function, ops: &Ops) {
    let old = std::mem::take(block);
    let mut rebuilt = Block::new();
    for (stmt, span) in old.span_into_iter() {
        for stmt in rewrite_stmt(stmt, function, ops) {
            rebuilt.push(stmt, span);
        }
    }
    *block = rebuilt;
}

fn rewrite_stmt(stmt: Statement, function: &mut Function, ops: &Ops) -> Vec<Statement> {
    match stmt {
        Statement::Emit(range) => rewrite_emit(range, function, ops),
        Statement::RayQuery { query, fun } => vec![rewrite_ray(query, fun, function, ops)],
        Statement::If {
            condition,
            mut accept,
            mut reject,
        } => {
            rewrite_block(&mut accept, function, ops);
            rewrite_block(&mut reject, function, ops);
            vec![Statement::If {
                condition,
                accept,
                reject,
            }]
        }
        Statement::Loop {
            mut body,
            mut continuing,
            break_if,
        } => {
            rewrite_block(&mut body, function, ops);
            rewrite_block(&mut continuing, function, ops);
            vec![Statement::Loop {
                body,
                continuing,
                break_if,
            }]
        }
        Statement::Switch {
            selector,
            mut cases,
        } => {
            for case in &mut cases {
                rewrite_block(&mut case.body, function, ops);
            }
            vec![Statement::Switch { selector, cases }]
        }
        Statement::Block(mut inner) => {
            rewrite_block(&mut inner, function, ops);
            vec![Statement::Block(inner)]
        }
        other => vec![other],
    }
}

/// An intersection read is an expression the backend cannot print. Name it
/// with a call instead, and keep emitting everything else in the range.
fn rewrite_emit(
    range: naga::Range<Expression>,
    function: &mut Function,
    ops: &Ops,
) -> Vec<Statement> {
    let end = range.index_range().end;
    let mut out = Vec::new();
    let mut chunk: Option<u32> = None;
    for handle in range {
        let index = handle.index() as u32;
        let intersection = match function.expressions[handle] {
            Expression::RayQueryGetIntersection { query, committed } => Some((query, committed)),
            _ => None,
        };
        let Some((query, committed)) = intersection else {
            if chunk.is_none() {
                chunk = Some(index);
            }
            continue;
        };
        if let Some(start) = chunk.take() {
            out.push(Statement::Emit(naga::Range::from_index_range(
                start..index,
                &function.expressions,
            )));
        }
        let fun = if committed {
            ops.get_committed
        } else {
            ops.get_candidate
        };
        function.expressions[handle] = Expression::CallResult(fun);
        out.push(Statement::Call {
            function: fun,
            arguments: vec![query],
            result: Some(handle),
        });
    }
    if let Some(start) = chunk {
        out.push(Statement::Emit(naga::Range::from_index_range(
            start..end,
            &function.expressions,
        )));
    }
    out
}

fn rewrite_ray(
    query: Handle<Expression>,
    fun: naga::RayQueryFunction,
    function: &mut Function,
    ops: &Ops,
) -> Statement {
    let (fun_handle, arguments, result) = match fun {
        naga::RayQueryFunction::Initialize {
            acceleration_structure,
            descriptor,
        } => (
            ops.initialize,
            vec![query, acceleration_structure, descriptor],
            None,
        ),
        naga::RayQueryFunction::Proceed { result } => {
            function.expressions[result] = Expression::CallResult(ops.proceed);
            (ops.proceed, vec![query], Some(result))
        }
        naga::RayQueryFunction::GenerateIntersection { hit_t } => {
            (ops.generate, vec![query, hit_t], None)
        }
        naga::RayQueryFunction::ConfirmIntersection => (ops.confirm, vec![query], None),
        naga::RayQueryFunction::Terminate => (ops.terminate, vec![query], None),
    };
    Statement::Call {
        function: fun_handle,
        arguments,
        result,
    }
}

/// Remove each `struct <name> { ... }` the backend emitted for a predeclared type.
fn strip_structs(src: &str, names: &[&str]) -> String {
    strip_items(src, "struct ", names)
}

/// Remove each `fn <name> ... { ... }` the backend emitted for a stand-in.
fn strip_functions(src: &str, names: &[&str]) -> String {
    strip_items(src, "fn ", names)
}

fn strip_items(src: &str, keyword: &str, names: &[&str]) -> String {
    let mut out = String::new();
    let mut rest = src;
    while let Some(at) = find_function(rest, keyword, names) {
        out.push_str(&rest[..at]);
        let after = skip_function(&rest[at..]);
        rest = &rest[at + after..];
    }
    out.push_str(rest);
    out
}

fn find_function(src: &str, keyword: &str, names: &[&str]) -> Option<usize> {
    let mut search_from = 0;
    while let Some(rel) = src[search_from..].find(keyword) {
        let at = search_from + rel;
        let name_at = at + keyword.len();
        let name_end = src[name_at..]
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .map(|i| name_at + i)
            .unwrap_or(src.len());
        let name = &src[name_at..name_end];
        if names.contains(&name) {
            // The signature starts a function item, not a call: a call is
            // `name(` and a definition here is `fn name`.
            return Some(at);
        }
        search_from = name_end;
    }
    None
}

/// Byte length of one function item, from `fn` through the closing brace.
fn skip_function(src: &str) -> usize {
    let Some(open) = src.find('{') else {
        return src.len();
    };
    let mut depth = 0;
    for (i, ch) in src[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return open + i + '}'.len_utf8();
                }
            }
            _ => {}
        }
    }
    src.len()
}
