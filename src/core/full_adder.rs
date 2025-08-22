use crate::model::register::Register;

/// Applies the quantum full adder circuit to a 4-qubit register.
///
/// The register is expected to be in the state |A, B, `C_in`, 0⟩, where A, B, and
/// `C_in` are the input bits. After the operation, the register will be in the
/// state |A, B, S, `C_out`⟩, where S is the sum and `C_out` is the carry-out.
///
/// The quantum circuit is laid out like so:
///
/// ```text
/// |   A⟩ -*-*-----*- |A    ⟩
/// |   B⟩ -*-*-*-*-⊕- |B    ⟩
/// |C_in⟩ -----*-⊕--- |S    ⟩
/// |   0⟩ -⊕---⊕----- |C_out⟩
/// ```
///
/// # Panics
///
/// Panics if the register does not contain at least 4 qubits (16 amplitudes).
pub fn full_adder(register: &mut Register<f64>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::register::Register;
    use rstest::rstest;

    /// A simple classical full adder function for verification.
    fn full_adder_classical(a: u8, b: u8, c_in: u8) -> (u8, u8) {
        let sum = a ^ b ^ c_in;
        let carry = (a & b) | (c_in & (a ^ b));
        (sum, carry)
    }

    /// Helper function to set up and run a test case.
    fn run_single_test_case(a_val: u8, b_val: u8, c_in_val: u8) {
        // Determine the expected classical output
        let (sum_expected, carry_expected) = full_adder_classical(a_val, b_val, c_in_val);

        // Calculate the initial state index based on the inputs |A,B,C_in,0⟩
        let initial_state_index =
            (a_val as usize) * 8 + (b_val as usize) * 4 + (c_in_val as usize) * 2;

        // Create a 4-qubit register initialized to the initial state
        let mut register = Register::from_basis_state(4, initial_state_index, 0.);

        // Apply the full adder circuit
        full_adder(&mut register);

        // The final state should be |A,B,S,C_out⟩.
        let final_state_index = (a_val as usize) * 8
            + (b_val as usize) * 4
            + (sum_expected as usize) * 2
            + (carry_expected as usize);

        // Check if the amplitude at the expected final state is 1.0 (within a small tolerance)
        let is_correct = register.measure() == final_state_index;

        assert!(
            is_correct,
            "Test case failed for A={a_val}, B={b_val}, C_in={c_in_val}. Expected final state index {final_state_index}, but found amplitude at {final_state_index}. Final state: {register}"
        );
    }

    #[rstest]
    #[case(0, 0, 0)]
    #[case(0, 0, 1)]
    #[case(0, 1, 0)]
    #[case(0, 1, 1)]
    #[case(1, 0, 0)]
    #[case(1, 0, 1)]
    #[case(1, 1, 0)]
    #[case(1, 1, 1)]
    fn test_000(#[case] a: u8, #[case] b: u8, #[case] c_in: u8) {
        run_single_test_case(a, b, c_in);
    }
}
