use std::{
    fmt::{Debug, Display, Formatter},
    ops::AddAssign,
    slice::Iter,
    vec::IntoIter,
};

use num_complex::Complex;
use num_traits::{ConstOne, ConstZero, Float, Num, Signed, Zero};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use crate::model::qubit::Qubit;

/// Represents the quantum state of a system.
///
/// A `Register` holds a vector of complex amplitudes, where each amplitude
/// corresponds to a basis state of the quantum system. The vector of amplitudes
/// is normalised, meaning the sum of the squared magnitudes of the amplitudes
/// must equal one. This normalization condition ensures that the total
/// probability of finding the system in any of its basis states is 1.
///
/// The type parameter `T` represents the underlying floating-point type for the
/// complex numbers, such as `f32` or `f64`.
///
/// # Invariants
///
/// The sum of the squared magnitudes of all amplitudes must be approximately 1.
/// Specifically, `(sum(x.norm_sqr()) - 1.0).abs() < T::epsilon()`.
pub struct Register<T>(Box<[Complex<T>]>);

impl<T> Register<T> {
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn iter(&'_ self) -> Iter<'_, Complex<T>> {
        self.0.iter()
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T> IntoIterator for Register<T> {
    type Item = Complex<T>;

    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Register<T> {
    type Item = &'a Complex<T>;
    type IntoIter = std::slice::Iter<'a, Complex<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Clone + Signed + Float + AddAssign + Debug> Register<T> {
    /// Creates a new `Register` from an iterator of complex amplitudes,
    /// returning `None` if the amplitudes are not normalised.
    ///
    /// This function takes an iterator of `Complex<T>` values representing the
    /// amplitudes of the basis states. It checks if the sum of the squared
    /// magnitudes of these amplitudes is approximately equal to 1. If the
    /// normalization condition holds, it returns `Some(Self)` containing the
    /// new register; otherwise, it returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex;
    /// use quantum_comp::model::register::Register;
    ///
    /// // A valid quantum state
    /// let amplitudes = vec![
    ///     Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ///     Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
    /// ];
    /// let register = Register::new(amplitudes).unwrap();
    ///
    /// // An invalid quantum state
    /// let amplitudes = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
    /// assert!(Register::new(amplitudes).is_none());
    /// ```
    pub fn new(amplitudes: impl IntoIterator<Item = Complex<T>>) -> Option<Self> {
        // 2ϵ is more appropriately forgiving than ϵ
        let epsilon = T::epsilon() + T::epsilon();
        let mut total_prob = T::zero();
        let res = amplitudes
            .into_iter()
            .inspect(|x| total_prob += x.norm_sqr())
            .collect();
        ((total_prob - T::one()).abs() < epsilon).then_some(Self(res))
    }
}

impl<T: Float> Register<T> {
    /// Constructs a `Register` representing a single basis state.
    ///
    /// The function creates a quantum state with an amplitude of magnitude 1
    /// at a specified `index`, and an amplitude of 0 for all other indices.
    /// The phase of the complex amplitude at the given index can be set using
    /// the `angle` parameter.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds for the specified number of
    /// qubits.
    pub fn from_basis_state(num_qubits: u8, index: usize, angle: T) -> Self {
        let size = 1_usize << num_qubits;
        assert!(index < size, "Basis state index is out of bounds");

        let mut amplitudes = vec![Complex::zero(); size];
        amplitudes[index] = Complex::from_polar(T::one(), angle);

        // SAFETY: The `from_basis_state` function ensures that only a single amplitude
        // is non-zero and has a magnitude of 1. All other amplitudes are zero.
        // The sum of the squared magnitudes is therefore `1.0.norm_sqr()` which is `1.0`.
        // This satisfies the `Register` normalization invariant.
        unsafe { Self::new_unchecked(amplitudes) }
    }
}

impl<T> Register<T> {
    /// Creates a new `Register` from an iterator of complex amplitudes without
    /// checking the normalization condition.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it assumes that the sum of the squared
    /// magnitudes of the provided `amplitudes` is approximately 1. If this
    /// condition is not met, it can lead to an invalid quantum state, violating
    /// the fundamental laws of quantum mechanics and potentially causing
    /// unpredictable behavior in subsequent operations.
    ///
    /// This function should only be used when you are absolutely certain that
    /// the input amplitudes are already normalised.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::FRAC_1_SQRT_2;
    ///
    /// use num_complex::Complex;
    /// use quantum_comp::model::register::Register;
    ///
    /// // The amplitudes are known to be normalised.
    /// let amplitudes = vec![
    ///     Complex::new(FRAC_1_SQRT_2, 0.0),
    ///     Complex::new(FRAC_1_SQRT_2, 0.0),
    /// ];
    /// let register = unsafe { Register::new_unchecked(amplitudes) };
    /// ```
    pub unsafe fn new_unchecked(amplitudes: impl IntoIterator<Item = Complex<T>>) -> Self {
        Self(amplitudes.into_iter().collect())
    }
}

impl<T: Clone + Num + ConstZero> Register<T> {
    /// Computes the **tensor product** of this register with a single qubit.
    ///
    /// This method combines the state of the current register (which may represent
    /// `n` qubits) with a single additional qubit, producing a new register that
    /// represents an `(n + 1)`-qubit system.
    ///
    /// Given:
    /// - Current register with amplitudes: `[r₀, r₁, ..., rₖ₋₁]`
    /// - Qubit with amplitudes: `(γ, δ)` (often called `alpha` and `beta`)
    ///
    /// The resulting register will have `2 * len` amplitudes arranged as:
    ///
    /// ```text
    /// [r₀ * γ, r₁ * γ, ..., rₖ₋₁ * γ, r₀ * δ, r₁ * δ, ..., rₖ₋₁ * δ]
    /// ```
    ///
    /// Conceptually:
    /// ```text
    /// |ψ⟩ (size k) ⊗ |q⟩ = [r₀, r₁, ..., rₖ₋₁] ⊗ [γ, δ]
    ///                   = [r₀γ, r₁γ, ..., rₖ₋₁γ, r₀δ, r₁δ, ..., rₖ₋₁δ]
    /// ```
    ///
    /// ```text
    /// ||ψ ⊗ φ||² = ||ψ||² * ||φ||² = 1 * 1 = 1
    /// ```
    #[must_use]
    pub fn tensor_qubit(self, other: Qubit<T>) -> Self {
        let len = self.len();
        let mut amplitudes = vec![Complex::ZERO; 2 * len];
        let (gamma, delta) = other.into_alpha_beta();

        for (i, amp) in self.into_iter().enumerate() {
            amplitudes[i] = amp.clone() * gamma.clone();
            amplitudes[i + len] = amp * delta.clone();
        }

        // SAFETY: The tensor product of two normalised states is always normalised.
        unsafe { Self::new_unchecked(amplitudes) }
    }
}

impl<T: Clone + Num> Register<T> {
    /// Computes the **tensor product** of this register with another register.
    ///
    /// This method produces a new `Register` whose state is the tensor product of
    /// the two input registers. If the first register represents an `n`-qubit system
    /// and the second represents an `m`-qubit system, the result will represent
    /// an `(n + m)`-qubit system.
    ///
    /// Given:
    /// - `self` with amplitudes: `[a₀, a₁, ..., aₖ₋₁]`
    /// - `other` with amplitudes: `[b₀, b₁, ..., bₗ₋₁]`
    ///
    /// The resulting amplitudes are computed as the **outer product**:
    ///
    /// ```text
    /// [a₀b₀, a₀b₁, ..., a₀bₗ₋₁,
    ///  a₁b₀, a₁b₁, ..., a₁bₗ₋₁,
    ///  ...
    ///  aₖ₋₁b₀, ..., aₖ₋₁bₗ₋₁]
    /// ```
    ///
    /// Conceptually:
    /// ```text
    /// |ψ⟩ ⊗ |φ⟩ = Σ_i Σ_j aᵢ bⱼ |i⟩|j⟩
    /// ```
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn tensor(self, other: Self) -> Self {
        let mut amplitudes = Vec::with_capacity(self.len() * other.len());

        for amp_self in self {
            for amp_other in &other {
                amplitudes.push(amp_self.clone() * amp_other.clone());
            }
        }

        // SAFETY: The tensor product of two normalised states is always normalised.
        unsafe { Self::new_unchecked(amplitudes) }
    }
}

impl<T: Clone + Num> Register<T> {
    /// Applies a **CNOT gate** (controlled-NOT) to the register.
    ///
    /// - `control`: The index of the control qubit (0 = least significant bit).
    /// - `target`: The index of the target qubit (must be different from control).
    ///
    /// # Panics
    ///
    /// Panics if `control == target` or if either index is out of range.
    pub fn cnot(&mut self, control: usize, target: usize) {
        assert!(control != target, "Control and target must be different");
        let n = self.len().ilog2() as usize;
        assert!(control < n && target < n, "Qubit index out of range");

        let mask_control = 1 << control;
        let mask_target = 1 << target;

        for i in 0..self.len() {
            if i & mask_control != 0 {
                let j = i ^ mask_target; // Flip target bit
                if j > i {
                    self.0.swap(i, j);
                }
            }
        }
    }
}

impl<T: Signed + ConstOne + ConstZero + Clone> Register<T> {
    /// Applies a **CY gate** (controlled-Y) to the register.
    ///
    /// The CY gate applies a Pauli-Y gate to the target qubit if the control
    /// qubit is in the |1⟩ state.
    ///
    /// This means for any pair of basis states where the control bit is 1,
    /// and they only differ by the target bit, their amplitudes `a` (for target=0)
    /// and `b` (for target=1) are updated as follows:
    ///
    /// `a` → `-i * b`
    /// `b` → `i * a`
    ///
    /// - `control`: The index of the control qubit (0 = least significant bit).
    /// - `target`: The index of the target qubit (must be different from control).
    ///
    /// # Panics
    ///
    /// Panics if `control == target` or if either index is out of range.
    pub fn cy(&mut self, control: usize, target: usize) {
        assert!(control != target, "Control and target must be different");
        let n = self.len().ilog2() as usize;
        assert!(control < n && target < n, "Qubit index out of range");

        let control_mask = 1 << control;
        let target_mask = 1 << target;

        for i in 0..self.len() {
            // Identify pairs of states where the control bit is 1. We process
            // each pair only once by selecting the state where the target bit is 0.
            let control_is_set = (i & control_mask) != 0;
            let target_is_set = (i & target_mask) != 0;

            if control_is_set && !target_is_set {
                // `i` corresponds to basis state |...control=1...target=0...⟩
                let j = i | target_mask;
                // `j` corresponds to basis state |...control=1...target=1...⟩

                let amp_i = self.0[i].clone(); // Original amplitude for |...1...0...⟩
                let amp_j = self.0[j].clone(); // Original amplitude for |...1...1...⟩

                // Apply the Y-gate transformation based on the CY logic.
                self.0[i] = Complex::<T>::I * -amp_j;
                self.0[j] = Complex::<T>::I * amp_i;
            }
        }
    }
}

impl<T: Clone + Signed> Register<T> {
    /// Applies a **CZ gate** (controlled-Z) to the register.
    ///
    /// The CZ gate applies a Z gate (phase flip) to the target qubit if the
    /// control qubit is in the |1⟩ state. This is equivalent to flipping the sign
    /// of the amplitude for any basis state where both the control and target
    /// qubits are 1.
    ///
    /// - `control`: The index of the control qubit (0 = least significant bit).
    /// - `target`: The index of the target qubit (must be different from control).
    ///
    /// # Panics
    ///
    /// Panics if `control == target` or if either index is out of range.
    pub fn cz(&mut self, control: usize, target: usize) {
        assert!(control != target, "Control and target must be different");
        let n = self.len().ilog2() as usize;
        assert!(control < n && target < n, "Qubit index out of range");

        let combined_mask = (1 << control) | (1 << target);

        for i in 0..self.len() {
            // Check if both the control and target bits are set in the state index 'i'.
            if (i & combined_mask) == combined_mask {
                // Apply the phase flip: amplitude -> -amplitude.
                self.0[i] = -self.0[i].clone();
            }
        }
    }
}

impl<T: Clone + Num> Register<T> {
    /// Applies a **CCNOT gate** (Toffoli gate) to the register.
    ///
    /// The CCNOT gate applies a NOT gate to the target qubit if and only if
    /// both control qubits are in the |1⟩ state.
    ///
    /// - `control1`: The index of the first control qubit (0 = LSB).
    /// - `control2`: The index of the second control qubit.
    /// - `target`: The index of the target qubit.
    ///
    /// # Panics
    ///
    /// Panics if any of the qubit indices are the same, or if any index is
    /// out of range.
    pub fn ccnot(&mut self, control1: usize, control2: usize, target: usize) {
        assert!(
            control1 != control2 && control1 != target && control2 != target,
            "Control and target qubits must be unique"
        );
        let n = self.len().ilog2() as usize;
        assert!(
            control1 < n && control2 < n && target < n,
            "Qubit index out of range"
        );

        let control_mask = (1 << control1) | (1 << control2);
        let target_mask = 1 << target;

        for i in 0..self.len() {
            // Check if both control bits are set for the current basis state 'i'.
            if (i & control_mask) == control_mask {
                // If so, this state is part of a pair whose amplitudes should be swapped.
                // The other state in the pair is the one with the target bit flipped.
                let j = i ^ target_mask;

                // To avoid swapping each pair twice, we only perform the swap
                // for the smaller index of the pair.
                if j > i {
                    self.0.swap(i, j);
                }
            }
        }
    }
}

impl<T> Register<T> {
    /// Applies a **SWAP gate** to the register.
    ///
    /// This gate swaps the quantum states of two specified qubits.
    ///
    /// - `qubit1`: The index of the first qubit to swap (0 = least significant bit).
    /// - `qubit2`: The index of the second qubit to swap.
    ///
    /// # Panics
    ///
    /// Panics if `qubit1 == qubit2` or if either index is out of range.
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) {
        assert!(qubit1 != qubit2, "The two qubits to swap must be different");
        let n = self.len().ilog2() as usize;
        assert!(qubit1 < n && qubit2 < n, "Qubit index out of range");

        let mask1 = 1 << qubit1;
        let mask2 = 1 << qubit2;

        for i in 0..self.len() {
            // Check if the bits at the qubit1 and qubit2 positions are different.
            // The `^` operator on booleans acts as an XOR.
            let bit1_is_set = (i & mask1) != 0;
            let bit2_is_set = (i & mask2) != 0;

            if bit1_is_set ^ bit2_is_set {
                // If the bits differ, this state is part of a pair to be swapped.
                // The other state `j` is found by flipping both bits.
                let j = i ^ mask1 ^ mask2;

                // To avoid swapping each pair twice, we only perform the swap
                // when processing the state with the smaller index.
                if j > i {
                    self.0.swap(i, j);
                }
            }
        }
    }
}

impl<T: Clone + Num + PartialOrd + Zero + AddAssign> Register<T>
where
    StandardUniform: Distribution<T>,
{
    /// Measures the entire quantum register, collapsing the state.
    ///
    /// This function performs a probabilistic measurement on the register.
    /// It returns the classical state (as a `usize` index) that was measured.
    /// After measurement, the quantum state of the register is updated to reflect
    /// the collapse, with the measured state having an amplitude of 1 and all
    /// others having an amplitude of 0.
    ///
    /// # Panics
    ///
    /// This function will panic if a random number generator cannot be created.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex;
    /// use quantum_comp::model::register::Register;
    ///
    /// let amplitudes = vec![
    ///     Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ///     Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
    /// ];
    /// let mut register = Register::new(amplitudes).unwrap();
    ///
    /// // The measurement result will be either 0 or 1, each with ~50% probability.
    /// let measured_state = register.measure();
    /// println!("Measured state: {}", measured_state);
    ///
    /// // After measurement, the register's state has collapsed.
    /// // The state |measured_state⟩ has amplitude (1.0, 0.0), others are (0.0, 0.0).
    /// ```
    pub fn measure(&mut self) -> usize {
        // Step 1: Get the squared magnitude (probability) of each amplitude.
        let probabilities: Vec<T> = self.0.iter().map(Complex::norm_sqr).collect();

        // Step 2: Create a cumulative distribution.
        let mut cumulative_probs = Vec::with_capacity(probabilities.len());
        let mut sum = T::zero();
        for prob in probabilities {
            sum += prob;
            cumulative_probs.push(sum.clone());
        }

        // Step 3: Generate a random number between 0 and 1.
        let mut rng = rand::rng();
        let random_number = rng.random::<T>();

        // Step 4: Find which state corresponds to the random number.
        let mut measured_index = 0;
        for (i, prob) in cumulative_probs.iter().enumerate() {
            if random_number <= *prob {
                measured_index = i;
                break;
            }
        }

        // Step 5: Collapse the state to the measured index.
        let mut collapsed_state = Vec::with_capacity(self.len());
        let zero = Complex::zero();
        let one = Complex::new(T::one(), T::zero());
        for i in 0..self.len() {
            if i == measured_index {
                collapsed_state.push(one.clone());
            } else {
                collapsed_state.push(zero.clone());
            }
        }
        self.0 = collapsed_state.into_boxed_slice();

        measured_index
    }
}

impl<T: Clone + Signed + Float + Display> Display for Register<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let num_qubits = self.len().ilog2() as usize;

        let formatted_states: Vec<String> = self
            .0
            .iter()
            .enumerate()
            .map(|(i, amplitude)| {
                // Get the bit representation for the state index 'i'
                let bit_string = format!("{i:0num_qubits$b}");
                format!("({amplitude:.2})|{bit_string}⟩")
            })
            .collect();

        write!(f, "{}", formatted_states.join(" + "))
    }
}
