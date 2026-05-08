# Reservoir Topology & Lorenz 63

Reproduction of results from Rathor et al. (2026) using Julia.

## Papers

- **Rathor et al. (2026)** — *Prediction performance of random reservoirs with different topology for nonlinear dynamical systems with different number of degrees of freedom*  
  Chaos 36, 033117 — https://doi.org/10.1063/5.0314081

- **Lukoševičius (2012)** — *A Practical Guide to Applying Echo State Networks*  
  Lecture Notes in Computer Science, 7700 — https://doi.org/10.1007/978-3-642-35289-8_36

## Running

```
julia --project=. lorenz_esn.jl
```

Outputs figures to `figures/` and scores to `outputs/results.json`.
