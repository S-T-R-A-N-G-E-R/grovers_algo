import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.exceptions import QiskitError
import matplotlib.pyplot as plt
import io

###############################################################################
# Helper functions
###############################################################################

def get_possible_states(n_qubits):
    """
    Generate all possible binary states for n qubits.
    E.g., n_qubits=2 -> ['00', '01', '10', '11']
    """
    return [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]

def apply_oracle(qc, n_qubits, target_state):
    """
    Marks (flips the phase of) the given target_state using a multi-controlled-Z.
    """
    # Convert target_state string (e.g., '101') into bits
    target_bits = [int(bit) for bit in target_state[::-1]]

    for i, bit in enumerate(target_bits):
        if bit == 0:
            qc.x(i)
    # Multi-controlled Z = H on last qubit -> MCX -> H on last qubit
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    # Undo the X-gates
    for i, bit in enumerate(target_bits):
        if bit == 0:
            qc.x(i)

def apply_diffusion(qc, n_qubits):
    """
    Performs the "inversion about the mean" (diffusion) step:
      D = H^⊗n * X^⊗n * (multi-controlled Z) * X^⊗n * H^⊗n
    """
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    # Multi-controlled Z
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

def run_grovers_algorithm(n_qubits, target_state, iterations=1):
    """
    Creates a Grover's circuit for n_qubits, marking target_state,
    and repeats (Oracle + Diffusion) for 'iterations' times.
    Returns the final circuit and measurement counts.
    """
    # 1. Create a quantum circuit
    qc = QuantumCircuit(n_qubits)

    # 2. Initialize all qubits in the uniform superposition
    qc.h(range(n_qubits))

    # 3. Repeat each Grover iteration: Oracle -> Diffusion
    for _ in range(iterations):
        apply_oracle(qc, n_qubits, target_state)
        apply_diffusion(qc, n_qubits)

    # 4. Measure all qubits
    qc.measure_all()

    # 5. Simulate the circuit
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=1024).result()
    counts = result.get_counts()

    return qc, counts

###############################################################################
# Streamlit App
###############################################################################

st.title("Interactive Grover’s Algorithm Demo")

# Sidebar for user input
st.sidebar.header("Simulation Parameters")
n_qubits = st.sidebar.selectbox("Number of Qubits", options=[2, 3], index=0)
target_options = get_possible_states(n_qubits)
target_state = st.sidebar.selectbox("Target State", options=target_options, 
                                    index=len(target_options)-1)  # e.g. '11' or '111'
iterations = st.sidebar.slider("Number of Iterations", min_value=1, max_value=4, value=1)

# Run button
if st.button("Run Simulation"):
    st.write("Running Grover’s Algorithm...")

    try:
        # Build and simulate circuit
        circuit, counts = run_grovers_algorithm(n_qubits, target_state, iterations)

        # Show the circuit
        st.write("Quantum Circuit:")
        try:
            fig = circuit.draw(output='mpl', scale=1.5)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            st.image(buf, caption="Grover’s Circuit", use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.write("Circuit visualization failed (using text instead):", str(e))
            st.code(circuit.draw(output='text'))

        # Show the measurement outcomes
        st.write("Measurement Outcomes:")
        st.write(counts)
        fig = plot_histogram(counts, title=f"Grover’s Algorithm Result (Target {target_state})")
        st.pyplot(fig)

    except QiskitError as e:
        st.error(f"A Qiskit error occurred: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Explanation
st.write("""
### How It Works
1. **Superposition**: Hadamard gates create all possible states with equal amplitudes.
2. **Oracle**: Marks (phase-flips) the target state.
3. **Diffusion**: Reflects amplitudes about the average, amplifying the marked state’s amplitude.
4. Grover’s Algorithm converges on the target state in \(O(\sqrt{N})\) steps.

**For 2 qubits**, the probability of measuring the target state oscillates:
- 1 iteration \(\rightarrow\) ~100%
- 2 iterations \(\rightarrow\) ~25%
- 3 iterations \(\rightarrow\) ~100%
- 4 iterations \(\rightarrow\) ~25%
""")

st.write("Developed by Swapnil Roy for Quantum Computing Exhibition")
