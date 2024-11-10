from pyqubo import Binary
# from dwave.system import LeapHybridSampler
import neal

# Define the variables
x1, x2, x3, y = Binary('x1'), Binary('x2'), Binary('x3'), Binary('y')

# Define the Hamiltonian
H = 2 * (3 * y + x1 * x2 - 2 * x1 * y - 2 * x2 * y) + y * x3

# Compile the model into a BQM
model = H.compile()
bqm = model.to_bqm()

# Solve using Simulated Annealing
num_reads = 200  # Number of samples
sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=num_reads)

# Check for the condition where x1 = x2 = x3 = 1
all_ones_count = sum(1 for sample in sampleset.samples() if sample['x1'] == 1 and sample['x2'] == 1 and sample['x3'] == 1)

print(f"Number of occurrences where x1 = x2 = x3 = 1: {all_ones_count}")
