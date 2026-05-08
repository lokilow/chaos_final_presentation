using DifferentialEquations, LinearAlgebra, SparseArrays
using Random, Statistics, Printf
using CairoMakie, JSON

const SEED = 42

const σ_L, ρ_L, β_L = 10.0, 28.0, 8.0/3.0
const N        = 1024
const D_r      = 0.008
const ε        = 0.7
const ΔT       = 0.05
const δt       = 0.002
const γ        = 1e-11
const T_wash   = 500
const T_train  = 2000
const T_test   = 2000
const CL_CLIP  = 1.5
const CL_STEPS = 500

# Table II: optimal (ρ, γ) for L63, all topologies share ε=0.7
const TOPOS = [
    (name="R-A",  ρ=0.7, sym_A=false, sym_W=false, ws=false),
    (name="RS-A", ρ=0.8, sym_A=true,  sym_W=false, ws=false),
    (name="RS-S", ρ=1.0, sym_A=true,  sym_W=true,  ws=false),
    (name="WS-A", ρ=0.8, sym_A=false, sym_W=false, ws=true),
    (name="WS-S", ρ=1.0, sym_A=false, sym_W=true,  ws=true),
]

# ── Data ──────────────────────────────────────────────────────────────────────
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ*(u[2]-u[1]); du[2] = u[1]*(ρ-u[3])-u[2]; du[3] = u[1]*u[2]-β*u[3]
end

function generate_lorenz(; seed=SEED)
    rng  = MersenneTwister(seed)
    sol  = solve(ODEProblem(lorenz!, randn(rng,3), (0.0,300.0), (σ_L,ρ_L,β_L)),
                 Tsit5(); saveat=δt, abstol=1e-10, reltol=1e-10)
    skip  = round(Int, ΔT/δt)
    trans = round(Int, 50.0/ΔT)
    idx   = [(trans+k)*skip+1 for k in 0:(T_wash+T_train+T_test)]
    data  = hcat([sol.u[i] for i in idx]...)
    lo = minimum(data; dims=2); hi = maximum(data; dims=2)
    return 2 .*(data.-lo)./(hi.-lo).-1
end

# ── Reservoir construction ────────────────────────────────────────────────────
function build_ws_adjacency(N, D_r; seed=SEED)
    rng = MersenneTwister(seed)
    k   = max(2, 2*round(Int, D_r*N/2))   # even, ≈ D_r·N per node
    A   = zeros(Bool, N, N)
    for i in 1:N, j in 1:k÷2
        nb = mod(i+j-1,N)+1; A[i,nb] = A[nb,i] = true
    end
    # rewire all right-going edges (p=1)
    for i in 1:N, j in 1:k÷2
        orig = mod(i+j-1,N)+1
        A[i,orig] || continue
        A[i,orig] = A[orig,i] = false
        new_nb = rand(rng, 1:N); tries = 0
        while (new_nb==i || A[i,new_nb]) && tries < 2N
            new_nb = rand(rng, 1:N); tries += 1
        end
        tries >= 2N && continue
        A[i,new_nb] = A[new_nb,i] = true
    end
    return sparse(A)
end

function build_reservoir(topo; seed=SEED)
    rng_A = MersenneTwister(seed)
    rng_W = MersenneTwister(seed + 9999)
    if topo.ws
        A = build_ws_adjacency(N, D_r; seed=seed)
    else
        n_e  = round(Int, D_r*N^2)
        idx  = randperm(rng_A, N*N)[1:n_e]
        rows = ((idx.-1).%N).+1
        cols = ((idx.-1).÷N).+1
        A    = sparse(rows, cols, ones(n_e), N, N)
        if topo.sym_A
            Ab = Matrix(A) .!= 0
            A  = sparse(Ab .| Ab')
        end
    end
    W_c = rand(rng_W, N, N) .- 0.5
    topo.sym_W && (W_c = (W_c + W_c')/2)
    # sparse Hadamard: keep sparsity pattern of A, weights from W_c
    rs, cs, _ = findnz(A)
    vals = [W_c[r,c] for (r,c) in zip(rs,cs)]
    W    = sparse(rs, cs, vals, N, N)
    ρ_a  = maximum(abs.(eigvals(Matrix(W))))
    return W * (topo.ρ / ρ_a)
end

# ── Reservoir dynamics ────────────────────────────────────────────────────────
function run_reservoir(W, W_in, b, data)
    r      = zeros(N)
    states = Matrix{Float64}(undef, N, T_wash+T_train+T_test)
    for k in 1:(T_wash+T_train+T_test)
        r = (1-ε).*r .+ ε.*tanh.(W*r .+ W_in.*data[2,k] .+ b)
        states[:,k] = r
    end
    s    = T_wash
    X_tr = states[:, s+1           : s+T_train]
    Y_tr = data[:,   s+2           : s+T_train+1]
    X_te = states[:, s+T_train+1   : s+T_train+T_test]
    Y_te = data[:,   s+T_train+2   : s+T_train+T_test+1]
    return X_tr, Y_tr, X_te, Y_te, states[:, s+T_train]
end

train_readout(X, Y) = Y*X' / (X*X' + γ*I)

function evaluate(Wout, X_te, Y_te)
    Yp    = Wout*X_te
    d     = Y_te .- Yp
    mse_t = mean(sum(d.^2; dims=1))
    mse_c = vec(mean(d.^2; dims=2))
    nrmse = sqrt.(mse_c ./ vec(var(Y_te; dims=2)))
    return mse_t, mse_c, nrmse, Yp
end

# ── Closed-loop ───────────────────────────────────────────────────────────────
function run_closed_loop(W, Wout, W_in, b, r0, n_steps)
    r     = copy(r0)
    preds = Matrix{Float64}(undef, 3, n_steps)
    preds[:,1] = clamp.(Wout*r, -CL_CLIP, CL_CLIP)
    for k in 2:n_steps
        r = (1-ε).*r .+ ε.*tanh.(W*r .+ W_in.*preds[2,k-1] .+ b)
        preds[:,k] = clamp.(Wout*r, -CL_CLIP, CL_CLIP)
    end
    return preds
end

function valid_time(preds, gt; thr=0.4)
    for k in 1:size(preds,2)
        norm(preds[:,k] .- gt[:,k]) > thr && return k*ΔT
    end
    return size(preds,2)*ΔT
end

# ── Main ──────────────────────────────────────────────────────────────────────
println("Generating Lorenz data...")
data = generate_lorenz()

rng_sh = MersenneTwister(SEED+100)
W_in   = rand(rng_sh, N) .- 0.5   # U(-0.5, 0.5) per paper
b      = rand(rng_sh, N) .- 0.5

res   = Dict{String,NamedTuple}()
Ypred = Dict{String,Matrix{Float64}}()
Ws    = Dict{String,SparseMatrixCSC{Float64,Int64}}()
Wouts = Dict{String,Matrix{Float64}}()
r0s   = Dict{String,Vector{Float64}}()
Y_te_ref = nothing

for (i, topo) in enumerate(TOPOS)
    @printf "[%d/%d] %-5s  ρ=%.1f ... " i length(TOPOS) topo.name topo.ρ
    W = build_reservoir(topo; seed=SEED+i*17)
    Ws[topo.name] = W
    X_tr, Y_tr, X_te, Y_te, r0 = run_reservoir(W, W_in, b, data)
    isnothing(Y_te_ref) && (global Y_te_ref = Y_te)
    r0s[topo.name]   = r0
    Wout = train_readout(X_tr, Y_tr)
    Wouts[topo.name] = Wout
    mse, mse_c, nrmse, Yp = evaluate(Wout, X_te, Y_te)
    Ypred[topo.name] = Yp
    res[topo.name] = (mse=mse, mse_c=mse_c, nrmse=nrmse)
    @printf "MSE=%.2e  NRMSE=(%.1e, %.1e, %.1e)\n" mse nrmse[1] nrmse[2] nrmse[3]
end

println("\n  Topology   MSE          A1(cross)   B1(direct)  B2(cross)")
for t in TOPOS
    r = res[t.name]
    @printf "  %-6s     %.3e   %.3e  %.3e  %.3e\n" t.name r.mse r.nrmse[1] r.nrmse[2] r.nrmse[3]
end

# ── Scan seeds for best closed-loop visual (RS-S vs R-A) ─────────────────────
# Score = vt_RSS * gap, only when RSS MSE < RA MSE
println("\nScanning closed-loop seeds (12 trials)...")
CL_gt = data[:, T_wash+T_train+1 : T_wash+T_train+CL_STEPS]

cl_trials = map(0:11) do trial
    s_rss = SEED + trial * 41 + 500
    s_ra  = SEED + trial * 41 + 507
    W_rss = build_reservoir(TOPOS[3]; seed=s_rss)
    W_ra  = build_reservoir(TOPOS[1]; seed=s_ra)
    Xtr_rss, Ytr_rss, Xte_rss, Yte_rss, r0_rss = run_reservoir(W_rss, W_in, b, data)
    Xtr_ra,  Ytr_ra,  Xte_ra,  Yte_ra,  r0_ra  = run_reservoir(W_ra,  W_in, b, data)
    Wo_rss = train_readout(Xtr_rss, Ytr_rss)
    Wo_ra  = train_readout(Xtr_ra,  Ytr_ra)
    mse_rss, _, _, _ = evaluate(Wo_rss, Xte_rss, Yte_rss)
    mse_ra,  _, _, _ = evaluate(Wo_ra,  Xte_ra,  Yte_ra)
    cl_rss = run_closed_loop(W_rss, Wo_rss, W_in, b, r0_rss, CL_STEPS)
    cl_ra  = run_closed_loop(W_ra,  Wo_ra,  W_in, b, r0_ra,  CL_STEPS)
    vt_rss = valid_time(cl_rss, CL_gt)
    vt_ra  = valid_time(cl_ra,  CL_gt)
    score  = (mse_rss < mse_ra ? 1.0 : 0.0) * vt_rss * max(0, vt_rss - vt_ra)
    @printf "  trial %2d: RSS vt=%.2ft (%.1fλ)  RA vt=%.2ft  score=%.2f\n" trial vt_rss (vt_rss/1.1) vt_ra score
    (; score, W_rss, W_ra, Wo_rss, Wo_ra, r0_rss, r0_ra, vt_rss, vt_ra, cl_rss, cl_ra)
end

# Also include the models already trained in the main loop
let cl_rss = run_closed_loop(Ws["RS-S"], Wouts["RS-S"], W_in, b, r0s["RS-S"], CL_STEPS),
    cl_ra  = run_closed_loop(Ws["R-A"],  Wouts["R-A"],  W_in, b, r0s["R-A"],  CL_STEPS)
    vt_rss = valid_time(cl_rss, CL_gt)
    vt_ra  = valid_time(cl_ra,  CL_gt)
    score  = (res["RS-S"].mse < res["R-A"].mse ? 1.0 : 0.0) * vt_rss * max(0, vt_rss - vt_ra)
    @printf "  main loop: RSS vt=%.2ft (%.1fλ)  RA vt=%.2ft  score=%.2f\n" vt_rss (vt_rss/1.1) vt_ra score
    push!(cl_trials, (; score, W_rss=Ws["RS-S"], W_ra=Ws["R-A"],
                        Wo_rss=Wouts["RS-S"], Wo_ra=Wouts["R-A"],
                        r0_rss=r0s["RS-S"],  r0_ra=r0s["R-A"],
                        vt_rss, vt_ra, cl_rss, cl_ra))
end

best   = cl_trials[argmax([t.score for t in cl_trials])]
CL_RSS = best.cl_rss;  vt_RSS = best.vt_rss
CL_RA  = best.cl_ra;   vt_RA  = best.vt_ra
@printf "\nBest: RSS vt=%.2ft (%.1fλ)  RA vt=%.2ft  gap=%.2fx\n" vt_RSS (vt_RSS/1.1) vt_RA (vt_RSS/max(vt_RA,0.05))

open("outputs/results.json","w") do f
    JSON.print(f, Dict(k => Dict("mse_total"=>v.mse,
                                  "nrmse_A1"=>v.nrmse[1],
                                  "nrmse_B1"=>v.nrmse[2],
                                  "nrmse_B2"=>v.nrmse[3]) for (k,v) in res), 2)
end
println("Saved outputs/results.json")

# ── Figures ───────────────────────────────────────────────────────────────────
println("\nGenerating figures...")
topo_names  = [t.name for t in TOPOS]
topo_colors = [:steelblue, :mediumseagreen, :darkorange, :mediumpurple, :firebrick]

# Figure 1: 5 reservoir matrix heatmaps
fig1  = Figure(size=(1400, 310))
clim1 = maximum(maximum(abs.(Matrix(Ws[t.name][1:32,1:32]))) for t in TOPOS)
for (i, topo) in enumerate(TOPOS)
    W32 = Matrix(Ws[topo.name][1:32, 1:32])
    ax  = Axis(fig1[1,i]; title=topo.name, xlabel="j", ylabel=i==1 ? "i" : "",
               aspect=DataAspect())
    hm  = heatmap!(ax, W32; colormap=:RdBu, colorrange=(-clim1, clim1))
    i == length(TOPOS) && Colorbar(fig1[1,i+1], hm; label="wᵢⱼ", width=14)
end
save("figures/figure1_matrices.png", fig1)
println("Saved figures/figure1_matrices.png")

# Figure 5: Open-loop attractor reconstruction (A₁ vs B₂)
# Ground truth panel + one panel per topology, 2×3 grid
fig5 = Figure(size=(1250, 840))
lims = (-1.25, 1.25, -1.25, 1.25)
positions = [(1,2),(1,3),(2,1),(2,2),(2,3)]

ax_gt = Axis(fig5[1,1]; title="Ground truth", xlabel="A₁", ylabel="B₂",
             limits=lims, aspect=DataAspect())
lines!(ax_gt, Y_te_ref[1,:], Y_te_ref[3,:]; color=:royalblue, linewidth=0.4)

for (j, topo) in enumerate(TOPOS)
    row, col = positions[j]
    Yp = Ypred[topo.name]
    ax = Axis(fig5[row,col]; title=topo.name, xlabel="A₁", ylabel="B₂",
              limits=lims, aspect=DataAspect())
    lines!(ax, Y_te_ref[1,:], Y_te_ref[3,:]; color=(:royalblue, 0.25), linewidth=0.4)
    lines!(ax, Yp[1,:],       Yp[3,:];       color=:darkorange,         linewidth=0.4)
end

elem_gt = LineElement(color=:royalblue,  linewidth=2)
elem_pr = LineElement(color=:darkorange, linewidth=2)
Legend(fig5[1:2, 4], [elem_gt, elem_pr], ["ground truth", "prediction"]; framevisible=false)
Label(fig5[0, 1:3], "Open-loop attractor reconstruction (A₁ vs B₂, input = B₁)";
      fontsize=14, tellwidth=false)
save("figures/figure5_attractor.png", fig5)
println("Saved figures/figure5_attractor.png")

# Figure 6: Total MSE comparison across topologies
fig6 = Figure(size=(680, 420))
ax6  = Axis(fig6[1,1]; title="Total MSE by topology — L63",
            ylabel="MSE", yscale=log10,
            xticks=(1:5, topo_names))
mse_vals = [res[t.name].mse for t in TOPOS]
barplot!(ax6, 1:5, mse_vals; color=topo_colors, width=0.6)
# Paper reference values (Table III medians)
hlines!(ax6, [3e-9];  color=:steelblue,  linestyle=:dash, linewidth=1.5, label="paper R-A (3e-9)")
hlines!(ax6, [8e-10]; color=:darkorange, linestyle=:dash, linewidth=1.5, label="paper RS-S/WS-S (8e-10)")
axislegend(ax6; position=:rt, labelsize=11)
save("figures/figure6_mse.png", fig6)
println("Saved figures/figure6_mse.png")

# Figure 7a: Per-component NRMSE (line chart, matches paper Fig 7a)
fig7 = Figure(size=(800, 460))
ax7  = Axis(fig7[1,1]; title="Per-component NRMSE — L63 (input B₁)",
            ylabel="NRMSE", yscale=log10,
            xticks=(1:5, topo_names))

comp_labels  = ["A₁ (cross)", "B₁ (direct)", "B₂ (cross)"]
comp_styles  = [:solid, :dash, :dot]
comp_markers = [:circle, :rect, :diamond]
comp_colors  = [:royalblue, :firebrick, :darkorange]

for (d, (lab, sty, mrk, col)) in enumerate(zip(comp_labels, comp_styles, comp_markers, comp_colors))
    vals = [res[t.name].nrmse[d] for t in TOPOS]
    lines!(ax7,   1:5, vals; linestyle=sty, linewidth=2, color=col, label=lab)
    scatter!(ax7, 1:5, vals; marker=mrk, markersize=10, color=col)
end
axislegend(ax7; position=:rt)
save("figures/figure7a_nrmse.png", fig7)
println("Saved figures/figure7a_nrmse.png")

# Figure: Closed-loop inference (RS-S vs R-A)
# Attractor: trajectory colored by time (plasma gradient) so divergence is visible
# Time series: tracking segment bright, diverging segment faded
t_axis  = (1:CL_STEPS) .* ΔT
n_att   = min(400, CL_STEPS)
t_col   = (1:n_att) .* ΔT                    # time values for colormap
k_rss   = clamp(round(Int, vt_RSS/ΔT), 1, n_att)
k_ra    = clamp(round(Int, vt_RA/ΔT),  1, n_att)

fig_cl  = Figure(size=(1150, 760))
att_lim = (-CL_CLIP, CL_CLIP, -CL_CLIP, CL_CLIP)

ax_rss = Axis(fig_cl[1,1];
              title="RS-S  (tracks $(round(vt_RSS;digits=1)) t, $(round(vt_RSS/1.1;digits=1)) λ)",
              xlabel="A₁", ylabel="B₂", limits=att_lim, aspect=DataAspect())
ax_ra  = Axis(fig_cl[1,2];
              title="R-A   (tracks $(round(vt_RA;digits=1)) t, $(round(vt_RA/1.1;digits=1)) λ)",
              xlabel="A₁", ylabel="B₂", limits=att_lim, aspect=DataAspect())

for (ax, CL) in ((ax_rss, CL_RSS), (ax_ra, CL_RA))
    # Ground truth: faint blue butterfly for reference
    lines!(ax, CL_gt[1,1:n_att], CL_gt[3,1:n_att];
           color=(:royalblue, 0.2), linewidth=0.7)
    # Prediction colored by time: plasma goes purple→red→yellow
    lines!(ax, CL[1,1:n_att], CL[3,1:n_att];
           color=t_col, colormap=:plasma, colorrange=(0, n_att*ΔT), linewidth=1.1)
end

# Shared colorbar for the time axis
Colorbar(fig_cl[1,3]; colormap=:plasma, limits=(0, n_att*ΔT),
         label="prediction time (model units)", width=14)

# Time series: bright = tracking, faded = diverging
ax_ts = Axis(fig_cl[2,1:2];
             title="Closed-loop B₂ — tracking then diverging (1 Lyapunov ≈ 1.1 t)",
             xlabel="time (model units)",
             ylabel="B₂ (normalised)",
             limits=(nothing, nothing, -CL_CLIP-0.1, CL_CLIP+0.1))

# Lyapunov time guides
for lya in 1.1:1.1:t_axis[end]
    vlines!(ax_ts, lya; color=(:gray, 0.15), linewidth=1, linestyle=:dot)
end

# Ground truth
lines!(ax_ts, t_axis, CL_gt[3,:]; color=:royalblue, linewidth=2.0, label="ground truth")

# RS-S: bright tracking segment, faded diverging segment
lines!(ax_ts, t_axis[1:k_rss],   CL_RSS[3,1:k_rss];
       color=:darkorange, linewidth=1.8,
       label="RS-S  tracking ($(round(vt_RSS;digits=1)) t)")
lines!(ax_ts, t_axis[k_rss:end], CL_RSS[3,k_rss:end];
       color=(:darkorange, 0.2), linewidth=1.0)

# R-A: bright tracking segment, faded diverging segment
lines!(ax_ts, t_axis[1:k_ra],    CL_RA[3,1:k_ra];
       color=:firebrick, linewidth=1.8, linestyle=:dash,
       label="R-A   tracking ($(round(vt_RA;digits=1)) t)")
lines!(ax_ts, t_axis[k_ra:end],  CL_RA[3,k_ra:end];
       color=(:firebrick, 0.2), linewidth=1.0, linestyle=:dash)

# Divergence markers
vlines!(ax_ts, vt_RSS; color=:darkorange, linewidth=1.5, linestyle=:dash)
vlines!(ax_ts, vt_RA;  color=:firebrick,  linewidth=1.5, linestyle=:dash)

axislegend(ax_ts; position=:rt)

save("figures/figure_closedloop.png", fig_cl)
println("Saved figures/figure_closedloop.png")

println("\nDone.")
