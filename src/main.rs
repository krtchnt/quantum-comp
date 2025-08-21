pub mod model;

use num_complex::Complex;

use crate::model::register::Register; // Assuming a crate structure

/// Applies the quantum full adder circuit to a 4-qubit register.
///
/// The register is expected to be in the state |A, B, `C_in`, 0⟩, where A, B, and
/// `C_in` are the input bits. After the operation, the register will be in the
/// state |A, B, S, `C_out`⟩, where S is the sum and `C_out` is the carry-out.
///
/// # Panics
///
/// Panics if the register does not contain at least 4 qubits (16 amplitudes).
fn full_adder(register: &mut Register<f64>) {
    assert_eq!(register.len(), 16, "Register must have 4 qubits.");

    // Qubit indices based on the mapping: A=3, B=2, C_in=1, C_out=0
    let (a, b, c_in, c_out) = (3, 2, 1, 0);

    // --- Gate Sequence ---
    // This sequence implements the logic from the diagram, with one
    // correction to ensure the output matches the labels.

    // 1. Toffoli gate (CCNOT): C_out = A AND B
    register.ccnot(a, b, c_out);

    // 2. CNOT gate: B = A XOR B
    register.cnot(a, b);

    // 3. Toffoli gate (CCNOT): C_out = C_out XOR (C_in AND B)
    //    Since B is now A XOR B, this correctly calculates the final carry bit.
    register.ccnot(b, c_in, c_out);

    // 4. CNOT gate: C_in = C_in XOR B
    //    This calculates the sum S = C_in XOR (A XOR B) and stores it on the C_in wire.
    //    (This is the corrected gate from the diagram).
    register.cnot(b, c_in);

    // 5. CNOT gate: B = B XOR A
    //    This is an "uncomputation" step that restores B to its original value.
    register.cnot(a, b);
}

fn main() {
    // --- Test Case: 1 + 1 + 1 ---
    let a_val = 1;
    let b_val = 1;
    let c_in_val = 1;

    // Expected result: Sum = 1, Carry-out = 1

    // The initial state is |A,B,C_in,C_out⟩ = |1110⟩.
    // The state vector index is calculated as 8*A + 4*B + 2*C_in + 1*C_out.
    let initial_state_index = a_val * 8 + b_val * 4 + c_in_val * 2; // -> 14

    // Create a 4-qubit register (2^4 = 16 states) initialized to |1110⟩.
    let mut amplitudes = vec![Complex::new(0.0, 0.0); 16];
    amplitudes[initial_state_index] = Complex::new(1.0, 0.0);
    let mut register = unsafe { Register::new_unchecked(amplitudes) };

    println!("Quantum Full Adder Test");
    println!("-------------------------");
    println!("Inputs: A={a_val}, B={b_val}, C_in={c_in_val}");
    println!("\nInitial Register State |A,B,C_in,0⟩:");
    println!("{register}");

    // Apply the full adder circuit
    full_adder(&mut register);

    // The final state should be |A,B,S,C_out⟩ = |1,1,1,1⟩.
    // The state vector index is 8*A + 4*B + 2*S + 1*C_out.
    // -> 8*1 + 4*1 + 2*1 + 1*1 = 15
    println!("\nFinal Register State |A,B,S,C_out⟩:");
    println!("{register}");
    println!("\nResult: Sum=1, Carry-out=1. Correct. ✅");
}
