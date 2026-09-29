//! `&&` and `||` evaluate their right side only when the left does not
//! decide, as Rust's lazy boolean operators do.
//!
//! A right side that acts (a call, an atomic) or that can go wrong (an index
//! that may be out of bounds) is lowered under an `if`. One that only reads
//! and computes is left as a plain `&&`, since evaluating it early cannot be
//! observed, and a guarded one costs a local.

use synaga::naga::{self, Expression, Statement};

/// Where a statement sits: each `if` it is under, and which branch.
#[derive(Clone, Debug, PartialEq)]
enum Step {
    Accept,
    Reject,
}

fn lower(src: &str) -> naga::Module {
    let module = synaga::parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    synaga::validate(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
    module
}

fn function<'a>(module: &'a naga::Module, name: &str) -> &'a naga::Function {
    module
        .functions
        .iter()
        .map(|(_, f)| f)
        .find(|f| f.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no function `{name}`"))
}

/// The branches leading to each statement `found` picks out, in order.
fn paths(block: &naga::Block, found: &dyn Fn(&Statement) -> bool) -> Vec<Vec<Step>> {
    fn walk(
        block: &naga::Block,
        found: &dyn Fn(&Statement) -> bool,
        path: &mut Vec<Step>,
        out: &mut Vec<Vec<Step>>,
    ) {
        for statement in block.iter() {
            if found(statement) {
                out.push(path.clone());
            }
            match statement {
                Statement::If { accept, reject, .. } => {
                    path.push(Step::Accept);
                    walk(accept, found, path, out);
                    path.pop();
                    path.push(Step::Reject);
                    walk(reject, found, path, out);
                    path.pop();
                }
                Statement::Block(inner) => walk(inner, found, path, out),
                Statement::Loop {
                    body, continuing, ..
                } => {
                    walk(body, found, path, out);
                    walk(continuing, found, path, out);
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(block, found, &mut Vec::new(), &mut out);
    out
}

/// The branches leading to the `Emit` that evaluates `wanted` expressions.
fn emitted_under(func: &naga::Function, wanted: &dyn Fn(&Expression) -> bool) -> Vec<Vec<Step>> {
    let found = |statement: &Statement| match statement {
        Statement::Emit(range) => range.clone().any(|h| wanted(&func.expressions[h])),
        _ => false,
    };
    paths(&func.body, &found)
}

const COUNTER: &str = "static counter: StorageMut<AtomicU32> = group(0).binding(0);";

#[test]
fn and_acts_only_when_the_left_holds() {
    let module = lower(&format!(
        "{COUNTER}\nfn first(enabled: bool) -> bool {{ enabled && counter.fetch_add(1) == 0 }}"
    ));
    let f = function(&module, "first");
    let atomics = paths(&f.body, &|s| matches!(s, Statement::Atomic { .. }));
    assert_eq!(atomics, [vec![Step::Accept]], "{:#?}", f.body);
}

#[test]
fn or_acts_only_when_the_left_fails() {
    let module = lower(&format!(
        "{COUNTER}\nfn first(done: bool) -> bool {{ done || counter.fetch_add(1) == 0 }}"
    ));
    let f = function(&module, "first");
    let atomics = paths(&f.body, &|s| matches!(s, Statement::Atomic { .. }));
    assert_eq!(atomics, [vec![Step::Reject]], "{:#?}", f.body);
}

#[test]
fn an_index_the_left_side_guards_is_not_read_early() {
    let module = lower(
        r#"
        static items: Storage<[u32]> = group(0).binding(0);
        fn present(i: u32) -> bool { i < items.len() && items[i as usize] != 0 }
        "#,
    );
    let f = function(&module, "present");
    let reads = emitted_under(f, &|e| matches!(e, Expression::Access { .. }));
    assert_eq!(reads, [vec![Step::Accept]], "{:#?}", f.body);
}

#[test]
fn nested_operators_nest_their_guards() {
    let module = lower(&format!(
        "{COUNTER}\nfn f(a: bool, b: bool) -> bool {{ a && (b || counter.fetch_add(1) == 0) }}"
    ));
    let f = function(&module, "f");
    let atomics = paths(&f.body, &|s| matches!(s, Statement::Atomic { .. }));
    assert_eq!(atomics, [vec![Step::Accept, Step::Reject]], "{:#?}", f.body);
}

#[test]
fn each_side_is_evaluated_once_and_in_order() {
    let module = lower(
        r#"
        static log: StorageMut<[u32]> = group(0).binding(0);
        fn mark(i: u32) -> bool {
            log.get_mut()[i as usize] = 1;
            true
        }
        fn both() -> bool { mark(0) && mark(1) }
        fn either() -> bool { mark(0) || mark(1) }
        "#,
    );
    for (name, guard) in [("both", Step::Accept), ("either", Step::Reject)] {
        let f = function(&module, name);
        let calls = paths(&f.body, &|s| matches!(s, Statement::Call { .. }));
        assert_eq!(calls, [vec![], vec![guard]], "{name}: {:#?}", f.body);
    }
}

#[test]
fn a_right_side_that_only_reads_stays_a_plain_operator() {
    let module = lower(
        r#"
        #[derive(Clone, Copy)]
        struct Params { lo: f32, hi: f32 }
        static params: Uniform<Params> = group(0).binding(0);
        fn inside(x: f32, y: f32) -> bool {
            x > params.lo && y < params.hi || x == y
        }
        "#,
    );
    let f = function(&module, "inside");
    assert!(f.local_variables.is_empty(), "{:#?}", f.local_variables);
    let branches = paths(&f.body, &|s| matches!(s, Statement::If { .. }));
    assert!(branches.is_empty(), "{:#?}", f.body);
}

#[test]
fn a_guarded_condition_steers_control_flow() {
    // The value a lazy `&&` produces is what `if` and `while` test.
    let module = lower(&format!(
        r#"{COUNTER}
        static items: Storage<[u32]> = group(0).binding(1);
        fn count(n: u32) -> u32 {{
            let mut i = 0u32;
            while i < n && items[i as usize] != 0 {{
                i += 1;
            }}
            if i < n && counter.fetch_add(1) == 0 {{
                i += 1;
            }}
            i
        }}
        "#
    ));
    let f = function(&module, "count");
    let atomics = paths(&f.body, &|s| matches!(s, Statement::Atomic { .. }));
    assert_eq!(atomics, [vec![Step::Accept]], "{:#?}", f.body);
}
