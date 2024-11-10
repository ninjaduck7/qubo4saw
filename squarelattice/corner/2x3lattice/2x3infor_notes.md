# 2x3 Lattice Simulation Notes

## Parameters Overview

This note describes the key parameters recorded during the simulation of a 2x3 lattice using a Hamiltonian approach. The parameters and metrics are helpful for understanding the computational resources used, as well as assessing the performance of the simulation.

### 1. **Compilation Time**
- **Definition**: The time required to compile the Hamiltonian into a form suitable for solving, such as a QUBO (Quadratic Unconstrained Binary Optimization) or BQM (Binary Quadratic Model).
- **Value Example**: `Compilation time: 0.0001 seconds`
- **Explanation**: Compilation involves converting the symbolic form of the Hamiltonian into a concrete optimization model. This process is typically very fast, especially for small problem sizes, because it involves algebraic transformations rather than computationally intensive calculations.

### 2. **Sampling Time**
- **Definition**: The time taken to solve the compiled Hamiltonian using a sampler, in this case, a simulated annealing sampler.
- **Value Example**: `Sampling time: 0.0035 seconds`
- **Explanation**: This step involves the solver iterating through various configurations to find the optimal solution (minimum energy state). The time is usually influenced by the complexity of the problem and the number of reads.

### 3. **Plotting Time**
- **Definition**: The time taken to generate and save plots that visualize the results of the sampling process.
- **Value Example**: `Plotting time: 0.8173 seconds`
- **Explanation**: Plotting involves creating visual representations of the sampled lattice configurations and saving them as images. This can be relatively time-consuming due to file I/O operations and rendering multiple figures.

### 4. **Total Execution Time**
- **Definition**: The total time taken for the entire simulation, including compilation, sampling, and plotting.
- **Value Example**: `Total execution time: 0.8209 seconds`
- **Explanation**: This value is the sum of compilation, sampling, and plotting times. It gives an overall view of the runtime efficiency of the entire program.

### 5. **Number of Reads**
- **Definition**: The number of times the sampler reads solutions.
- **Value Example**: `Number of reads: 10`
- **Explanation**: In each read, the sampler attempts to find a good solution for the Hamiltonian. More reads can improve the likelihood of finding an optimal solution but increase the sampling time.

### 6. **Memory Usage After Compilation**
- **Definition**: The memory used by the program after compiling the Hamiltonian.
- **Value Example**: `Memory usage after compilation: 20.50 MB`
- **Explanation**: Compilation of the Hamiltonian involves creating data structures that hold the model, which contributes to memory usage.

### 7. **Memory Usage After Sampling**
- **Definition**: The memory used by the program after completing the sampling phase.
- **Value Example**: `Memory usage after sampling: 25.30 MB`
- **Explanation**: During sampling, the sampler may create additional data structures to store results, which can increase memory usage.

### 8. **Memory Usage After Plotting**
- **Definition**: The memory used by the program after generating and saving all the plots.
- **Value Example**: `Memory usage after plotting: 30.75 MB`
- **Explanation**: Plotting requires additional memory for figure objects, which may significantly increase memory usage depending on the number and complexity of plots.

### 9. **Hamiltonian Parameters (A_mono and A_bond)**
- **A_mono**: The coefficient controlling the contribution of the node term in the Hamiltonian (`H_node`).
  - **Value Example**: `A_mono: 1`
  - **Explanation**: This coefficient adjusts the importance of the node-related term in the Hamiltonian, which represents the number of active nodes.
- **A_bond**: The coefficient controlling the bond-related terms in the Hamiltonian (`H_bond_all - (-H_bond)`).
  - **Value Example**: `A_bond: 1`
  - **Explanation**: This coefficient adjusts the importance of the bond-related term in the Hamiltonian, representing interactions between different nodes and layers in the lattice.

## Considerations for Future Simulations
- **Scalability**: As the lattice size grows, both the sampling time and memory usage will likely increase, making it important to consider efficient memory management and optimized sampling strategies.
- **Parameter Tuning**: Adjusting `A_mono` and `A_bond` can significantly affect the solution quality. Experimentation with these parameters may be necessary to achieve the desired solution characteristics.
- **Resource Monitoring**: Using `psutil` to monitor memory usage provides useful insights into the program's resource consumption. This information can help identify bottlenecks or areas for optimization, particularly for larger simulations.

