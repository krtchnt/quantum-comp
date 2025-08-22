use std::{
    fmt::{Display, Formatter},
    ops::Neg,
};

use num_complex::Complex;
use num_traits::{ConstOne, ConstZero, Float, FloatConst, Num, Signed};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use crate::model::register::Register;

/// Represents a single qubit.
///
/// A qubit is the quantum analogue of a classical bit. While a classical bit can
/// only exist in the state 0 or 1, a qubit can exist in a superposition of both.
/// This superposition is represented by a linear combination of the basis states
/// |0> and |1>, where the coefficients are complex numbers.
///
/// The state of a qubit is typically written as `|ψ⟩ = α|0⟩ + β|1⟩`, where `α`
/// and `β` are complex numbers called **amplitudes**. The squares of the
/// absolute values of these amplitudes, `|α|²` and `|β|²`, represent the
/// probabilities of measuring the qubit in the |0> or |1> state, respectively.
/// According to the **normalization condition**, these probabilities must sum to 1:
/// `|α|² + |β|² = 1`.
pub struct Qubit<T>(Complex<T>, Complex<T>);

impl<T> Qubit<T> {
    pub fn into_alpha_beta(self) -> (Complex<T>, Complex<T>) {
        (self.0, self.1)
    }
}

impl<T: ConstOne + ConstZero> Qubit<T> {
    /// The canonical basis state `|0>`, representing a classical 0.
    pub const KET_0: Self = Self(Complex::ONE, Complex::ZERO);
    /// The canonical basis state `|1>`, representing a classical 1.
    pub const KET_1: Self = Self(Complex::ZERO, Complex::ONE);
}

impl<T: Clone + Signed + Float> Qubit<T> {
    /// Constructs a new `Qubit` from two complex amplitudes.
    ///
    /// This method checks the **normalization condition** `|α|² + |β|² = 1` to
    /// ensure the state is valid. If the condition is not met, it returns `None`.
    pub fn new(alpha: Complex<T>, beta: Complex<T>) -> Option<Self> {
        let total_prob = alpha.norm_sqr() + beta.norm_sqr();
        ((total_prob - T::one()).abs() < T::epsilon()).then_some(Self(alpha, beta))
    }
}

impl<T> Qubit<T> {
    /// Constructs a new `Qubit` from two complex amplitudes without checking the
    /// normalization condition.
    ///
    /// # Safety
    /// This function is unsafe because it does not validate that `|α|² + |β|²`
    /// equals 1. Using an unnormalised qubit state can lead to incorrect
    /// probabilities and invalid quantum operations.
    pub const unsafe fn new_unchecked(alpha: Complex<T>, beta: Complex<T>) -> Self {
        Self(alpha, beta)
    }

    /// The identity gate.
    ///
    /// This gate leaves the qubit's state unchanged. It corresponds to a 2x2
    /// identity matrix:
    ///
    /// ```text
    /// [ 1  0 ]
    /// [ 0  1 ]
    /// ```
    #[inline]
    #[must_use]
    pub const fn identity(self) -> Self {
        self
    }

    /// The Pauli-X gate.
    ///
    /// This gate, also known as the bit-flip gate, is the quantum equivalent of
    /// the classical NOT gate. It swaps the amplitudes of the `|0>` and `|1>`
    /// states.
    ///
    /// It corresponds to the Pauli-X matrix:
    ///
    /// ```text
    /// [ 0  1 ]
    /// [ 1  0 ]
    /// ```
    #[must_use]
    pub fn pauli_x(self) -> Self {
        // SAFETY: as addition is commutative,
        // alpha^2 + beta^2 == beta^2 + alpha^2, so they will still
        // abide by the normalisation condition, thus this is safe.
        unsafe { Self::new_unchecked(self.1, self.0) }
    }
}

impl<T: Num + Clone + ConstOne + ConstZero + Neg<Output = T>> Qubit<T> {
    /// The Pauli-Y gate.
    ///
    /// This gate performs a bit-flip and a phase-flip. It transforms the `|0>`
    /// state to `i|1>` and the `|1>` state to `-i|0>`.
    ///
    /// It corresponds to the Pauli-Y matrix:
    ///
    /// ```text
    /// [ 0  -i ]
    /// [ i   0 ]
    /// ```
    #[must_use]
    pub fn pauli_y(self) -> Self {
        // SAFETY: for every complex a + bi: i(a + bi) = ai - b,
        // meaning the amplitudes a and b themselves are not changed,
        // so their norm must also not changed, so the normalisation
        // condition still holds true as only their norms are used,
        // thus this is safe.
        unsafe { Self::new_unchecked(self.1 * -Complex::I, self.0 * Complex::I) }
    }
}

impl<T: Clone + Num + Neg<Output = T>> Qubit<T> {
    /// The Pauli-Z gate.
    ///
    /// This gate is a phase-flip gate that leaves the `|0>` state unchanged and
    /// flips the sign of the `|1>` state.
    ///
    /// It corresponds to the Pauli-Z matrix:
    ///
    /// ```text
    /// [ 1  0 ]
    /// [ 0 -1 ]
    /// ```
    #[must_use]
    pub fn pauli_z(self) -> Self {
        // SAFETY: as beta is only negated, its norm will stay
        // unchanged, so the normalisation condition still holds,
        // thus this is safe.
        unsafe { Self::new_unchecked(self.0, -self.1) }
    }
}

impl<T: Clone + Num + FloatConst> Qubit<T> {
    /// The Hadamard gate.
    ///
    /// This gate is a key operation in quantum computing that creates a
    /// superposition state from a basis state. It transforms the `|0>` state to
    /// `(|0> + |1>) / √2` and the `|1>` state to `(|0> - |1>) / √2`.
    ///
    /// It corresponds to the Hadamard matrix:
    ///
    /// ```text
    /// [ 1/√2  1/√2 ]
    /// [ 1/√2 -1/√2 ]
    /// ```
    #[must_use]
    pub fn hadamard(self) -> Self {
        let alpha = (self.0.clone() + self.1.clone()) * T::FRAC_1_SQRT_2();
        let beta = (self.0 - self.1) * T::FRAC_1_SQRT_2();

        // SAFETY: the highest unnormalised value of alpha is √2, so
        // by normalising both alpha and beta by √2, the old beta
        // values cancels out after adding them up to 1.0 again,
        // abiding the normalisation condition, so this is safe.
        unsafe { Self::new_unchecked(alpha, beta) }
    }
}

impl<T: ConstOne + ConstZero + Float> Qubit<T> {
    /// The general phase shift gate `R_φ`.
    ///
    /// This gate applies a phase shift `e^(iφ)` to the `|1>` state while leaving
    /// the `|0>` state unchanged.
    ///
    /// It corresponds to the matrix:
    ///
    /// ```text
    /// [ 1     0   ]
    /// [ 0  e^(iφ) ]
    /// ```
    ///
    /// where `φ` is the given phase angle in radians.
    #[must_use]
    pub fn phase_shift(self, phi: T) -> Self {
        // SAFETY: The `|0>` amplitude `self.0` is unchanged, and the `|1>`
        // amplitude `self.1` is multiplied by a complex number with a magnitude
        // of 1 (`e^(iφ)`). This means that `|self.1|^2` remains unchanged, and
        // therefore, the normalization condition `|self.0|^2 + |self.1|^2 = 1`
        // is preserved.
        unsafe { Self::new_unchecked(self.0, self.1 * (Complex::I * phi).exp()) }
    }
}

impl<T: FloatConst> Qubit<T> {
    /// The T gate, or π/4-gate.
    ///
    /// This is a specific phase shift gate where the phase angle φ = π/4.
    /// It corresponds to the matrix:
    ///
    /// ```text
    /// [ 1      0    ]
    /// [ 0  e^(iπ/4) ]
    /// ```
    ///
    /// where e^(iπ/4) = 1/√2 + i/√2. This gate is a common building block
    /// in quantum circuits.
    #[must_use]
    pub fn t_gate(self) -> Self {
        // SAFETY: The T gate is a special case of the `phase_shift` gate with a
        // fixed phase angle, and as such, it also preserves the normalization of
        // the qubit state. The magnitude of `self.1 * e^(iπ/4)` is `|self.1|`,
        // as `|e^(iπ/4)| = 1`.
        unsafe { Self::new_unchecked(self.0, Complex::new(T::FRAC_1_SQRT_2(), T::FRAC_1_SQRT_2())) }
    }
}

impl<T: Clone + Num + ConstOne + ConstZero> Qubit<T> {
    /// The S gate, or π/2-gate.
    ///
    /// This is a specific phase shift gate where the phase angle φ = π/2.
    /// It corresponds to the matrix:
    ///
    /// ```text
    /// [ 1  0 ]
    /// [ 0  i ]
    /// ```
    ///
    /// The S gate is a rotation by 90 degrees around the Z-axis of the Bloch sphere.
    #[must_use]
    pub fn s_gate(self) -> Self {
        // SAFETY: The S gate is a special case of the `phase_shift` gate,
        // applying a multiplication by `i` to the `|1>` amplitude. Since the
        // magnitude of `i` is 1, the normalization of the qubit state is
        // preserved. `|self.1 * i|^2 = |self.1|^2 * |i|^2 = |self.1|^2 * 1`.
        unsafe { Self::new_unchecked(self.0, self.1 * Complex::I) }
    }
}

impl<T: Clone + Num + PartialOrd + ConstOne + ConstZero> Qubit<T>
where
    StandardUniform: Distribution<T>,
{
    /// Simulates the measurement of a qubit in the computational
    /// basis.
    ///
    /// The state of the qubit, `|psi> = a|0> + b|1>`, collapses to
    /// either `|0>` or `|1>`. The outcome is probabilistic, with the
    /// probability of collapsing to `|0>` being `|a|^2` and the
    /// probability of collapsing to `|1>` being `|b|^2`. The return
    /// value is the collapsed state.
    #[must_use]
    pub fn measure(&mut self) -> bool {
        let prob_0 = self.0.norm_sqr();
        let mut rng = rand::rng();
        let res = rng.random::<T>() < prob_0;
        *self = if res { Self::KET_0 } else { Self::KET_1 };
        res
    }
}

impl<T: Clone + Num> Qubit<T> {
    /// Computes the tensor product of two qubits, returning a 2-qubit register.
    ///
    /// The tensor product combines the two single-qubit states into a single
    /// quantum register representing a two-qubit system.
    ///
    /// For two qubits, `|ψ⟩ = α|0⟩ + β|1⟩` and `|φ⟩ = γ|0⟩ + δ|1⟩`, the
    /// resulting state `|ψ⟩ ⊗ |φ⟩` is:
    /// `(αγ)|00⟩ + (αδ)|01⟩ + (βγ)|10⟩ + (βδ)|11⟩`
    ///
    /// The returned `Register` will have four complex amplitudes corresponding to
    /// the basis states `|00⟩`, `|01⟩`, `|10⟩`, and `|11⟩`, respectively.
    ///
    /// # Arguments
    ///
    /// * `other`: The other qubit to be tensored with.
    ///
    /// # Returns
    ///
    /// A new `Register` representing the combined two-qubit state. The
    /// function is marked as `unsafe` because it bypasses the normalization
    /// check in `Register::new_unchecked`, relying on the fact that the
    /// tensor product of two normalised states is always normalised.
    #[must_use]
    pub fn tensor(self, other: Self) -> Register<T> {
        let (alpha, beta) = (self.0, self.1);
        let (gamma, delta) = (other.0, other.1);

        let amplitudes = vec![
            alpha.clone() * gamma.clone(), // |00>
            alpha * delta.clone(),         // |01>
            beta.clone() * gamma,          // |10>
            beta * delta,                  // |11>
        ];

        // SAFETY: The tensor product of two normalised vectors is always
        // normalised. If |alpha|^2 + |beta|^2 = 1 and |gamma|^2 + |delta|^2 = 1,
        // then |alpha*gamma|^2 + |alpha*delta|^2 + |beta*gamma|^2 + |beta*delta|^2
        // = |alpha|^2|gamma|^2 + |alpha|^2|delta|^2 + |beta|^2|gamma|^2 + |beta|^2|delta|^2
        // = |alpha|^2(|gamma|^2 + |delta|^2) + |beta|^2(|gamma|^2 + |delta|^2)
        // = (|alpha|^2 + |beta|^2)(|gamma|^2 + |delta|^2)
        // = 1 * 1 = 1.
        unsafe { Register::new_unchecked(amplitudes) }
    }
}

impl<T: Clone + Num + ConstZero> Qubit<T> {
    /// Computes the **tensor product** of this qubit with a quantum register.
    ///
    /// # Description
    /// This method combines the state of a single qubit (two amplitudes) with a
    /// quantum register representing `n` qubits, resulting in an `(n + 1)`-qubit
    /// register.
    ///
    /// If the qubit has amplitudes `(α, β)` and the register has amplitudes
    /// `[r₀, r₁, ..., rₖ₋₁]`, then the resulting register will have `2 * k`
    /// amplitudes arranged as:
    ///
    /// ```text
    /// [α * r₀, α * r₁, ..., α * rₖ₋₁,
    ///  β * r₀, β * r₁, ..., β * rₖ₋₁]
    /// ```
    ///
    /// Conceptually:
    /// ```text
    /// |q⟩ ⊗ |ψ⟩ = [α, β] ⊗ [r₀, r₁, ..., rₖ₋₁]
    ///            = [αr₀, αr₁, ..., αrₖ₋₁, βr₀, βr₁, ..., βrₖ₋₁]
    /// ```
    ///
    /// This is similar to `Register::tensor_qubit` but with the qubit as the **left**
    /// operand.
    #[must_use]
    pub fn tensor_register(self, other: Register<T>) -> Register<T> {
        let other_len = other.len();
        let mut amplitudes = vec![Complex::ZERO; 2 * other_len];
        let (alpha, beta) = (self.0, self.1);

        for (i, amp) in other.into_iter().enumerate() {
            amplitudes[i] = alpha.clone() * amp.clone();
            amplitudes[i + other_len] = beta.clone() * amp;
        }

        // SAFETY: The tensor product of two normalised states is always normalised.
        unsafe { Register::new_unchecked(amplitudes) }
    }
}

impl<T: Display + Num + Clone + PartialOrd> Display for Qubit<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:.2})|0⟩ + ({:.2})|1⟩", self.0, self.1)
    }
}
