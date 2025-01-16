### A Pluto.jl notebook ###
# v0.20.4

#> [frontmatter]
#> title = " Report: Learining and Inference on Graphical Models"
#> date = "2025-01-15"
#> 
#>     [[frontmatter.author]]
#>     name = "Eduardo Ascarrunz"

using Markdown
using InteractiveUtils

# ╔═╡ bf6370e0-d373-11ef-1e7b-45131113726b
begin
	using Graphs, SimpleWeightedGraphs
	using GraphMakie, GLMakie
	using Random
	using Distributions, Statistics
	using LinearAlgebra
	using StatsBase
	using Combinatorics
	using Base.Threads
	using CSV, DataFrames
	using Downloads
end

# ╔═╡ 6e081a14-6225-4b8f-9bb4-a54fe93050b2
md"""
# Report: Learining and Inference on Graphical Models

Eduardo Ascarrunz
"""

# ╔═╡ 5703ce81-81be-4c98-b108-bfc0e343cb69
md"""
!!! important

	- The notebook includes its own environment: first execution can take a long time due to dependency download and compilation

	- Cells with function descriptions can be expanded to show the code in edit mode

	- Re-running this notebook will yield different simulation results; values in the text are also updated accordingly

## Introduction

Biological systems are characterized by complex networks of causal relations between numerous and heterogeneous entities. Relations may be weak and difficult to detect, and so complex that they are difficult to apprehend. Hence, it is no surprise that statistical tools for causal discovery are finding use in areas of biology ranging from cell biology (e.g. Sachs et al. 2005) to ecology (e.g. Dee et al. 2023). It is perhaps most critical in the latter case, as the systems involved are often not amenable to experimental interventions, but may provide a wealth of observational data.

In this report I implemented (imperfectly) the Peter-Clark algorithm for inference of undirected skeletons of causal directed acyclic graphs, and applied it to simulated and empirical data. Broadly speaking, the algorithm starts with a complete undirected graph, and iteratively removes edges based on tests of independece. Over successive iterations, the tests are performed conditionally dependent on larger sets of variables. Traditionally, the independence tests are based on partial correlations coefficients, under the assumption of multivariate normal data.

---

This document is provided as a Pluto notebook. These notebooks are reactive like spreadsheets: a modification in one cell triggers the update of other cells that depend on it. Cells can be *presented* in any order; their order of execution and position in the source file are managed by Pluto based on a dependency graph. This seemed appropriate for the subject of the report.

"""

# ╔═╡ 160e3d99-9fcd-473e-a2a2-85e84fba9dc9
Random.seed!(12345678909876554321);

# ╔═╡ e5df4d5c-1305-4c30-8d11-3aa12be45923
md"""
## Data Simulation

I simulated 15 data columns (I refer to data variables as data columns in order to avoid confusion with code variables) in this exercise, corresponding to 15 vertices in the graphs, represented by the variable `n`.
"""

# ╔═╡ b009213d-1ff8-49af-a662-8aea364ba3ce
n = 15

# ╔═╡ 25d4ea7a-c325-4c6e-aeaf-dd127870c073
md"""
I started by creating a random graph called `gsim`, for which I wrote the `randgraph` function. with the Erdos-Renyi model, with an edge probability of 0.3.

This function is based on the example code from the report guidelines, but generates an undirected graph.
"""

# ╔═╡ 4d0d3dd6-e9c5-4295-a9a3-0645081284b5
"""
	randgraph(n, p_edge)

Create a random Erdos-Renyi undirected graph of `n` vertices with edge probability `p_edge`.

Edge lengths are drawn randomly from a uniform distribution in the range [0.1, 1.5].
"""
function randgraph(n, p_edge)
	g = SimpleWeightedGraph(n)
	for i in 1:(n - 1), j in (i + 1):n
		r = rand()
		r < p_edge && add_edge!(g, i, j, r / p_edge + 0.5)
	end

	return g
end

# ╔═╡ 41f9171f-1b7f-48eb-9127-1d0bb82699dd
gsim = randgraph(n, 0.25)

# ╔═╡ 46d26a14-baa3-450c-98c9-375764b9a9e6
md"""
The generated graph has $(n) vertices and $(ne(gsim)) edges.
"""

# ╔═╡ 488e68e9-6f91-4060-b445-6c0dad0a1f13
md"""
Because the graph is undirected, the adjacency matrix `Asim` is already symmetric.
"""

# ╔═╡ a94e608c-c98c-4ffe-8bf6-e33a93b923a3
Asim = adjacency_matrix(gsim);

# ╔═╡ d601ac44-b19e-4398-ae3c-f80e5b09fdf1
heatmap(Asim)

# ╔═╡ 7a539c17-8da2-4086-a849-ebb78263b0fb
md"""
Then, I used the `simulate_dataset` function to generate a small dataset of 20 samples (=rows) called `sim1` and a larger dataset of 1000 samples called `sim2`.
I wrote the `simulate_dataset` function closely following the R example given in the report guidelines.
"""

# ╔═╡ d7b2ab9a-2fb6-479c-a825-21a1bca40f95
"""
	simulate_dataset(A, m)

Simulate a multivarate Gaussian dataset of `m` variables based in the dependence structure implied by an adjacency matrix `A`. 
"""
function simulate_dataset(A, m)
	n = size(A, 1)
	
	X = zeros(m, n)
	for i in 1:n
		pa = A[:, i] .> 0
		if any(pa)
			Xpa = X[:, pa]
			X[:, i] .= (Xpa) * A[pa, i] .+ randn(m)
		else
			X[:, i] .= randn(m)
		end
	end

	return X
end

# ╔═╡ 78e29fc3-2989-4134-ae6c-e216f0724c1e
sim1 = simulate_dataset(Asim, 20);

# ╔═╡ af7e012a-7eb4-4095-98eb-bfed308314ea
sim2 = simulate_dataset(Asim, 1_000);

# ╔═╡ e364a4d3-840f-4539-9114-4eb96f28d3c8
md"""
## Graph Structure Estimation

### Implementation

The Julia package Associations has an implementation of the Peter-Clarke algorithm. However, after some experiments I found it too slow. The performance is strongly affected by the inefficient calculation of partial correlation coefficients.

Therefore, I implemented the basic building blocks of the first Peter-Clarke algorithm by myself.
"""

# ╔═╡ 16be2434-fa51-4fbf-a73c-ba8a179703df
md"""
Although more limited, only performing skeleton estimation, it function is much faster. The main optimization is in the computation of the partial correlation coefficients. I used a pre-processing step to compute the unconditional correlation matrix, and then partial correlation coeffients can be computed with a recursive formula without looking at the original data again. More detailed explanations are given structure and function descriptions below.
"""

# ╔═╡ 36c275be-1319-49c0-aaac-e33f076e074b
"""
    RecursivePartialCorrContext(data)

A context structure for storing pre-processed information for computing partial correlations in some `data`.

Data must be provided in float matrix form, with columns as variables and rows as samples.

# Fields:
- R: Matrix of unconditional pairwise correlation coefficients
- m: Number of samples in the data
- n: Number of columns (=variables) in the data)
"""
struct RecursivePartialCorrContext
    R::Matrix{Float64}
    m::Int
    n::Int

    RecursivePartialCorrContext(data) = new(cor(data), size(data)...)
end

# ╔═╡ 31706282-8f2a-42bd-90d2-daef9977d1ac
"""
	partialcor_recursive(ctx::RecursivePartialCorrContext, x, y, Z)

Compute a partial correlation coefficient from a preprocessed context structure.

`x` and `y` correspond to the indices of the columns in the original data matrix, and `Z` is a (potentially empty) vector with the indices of the conditioning columns. 

This function makes use of the recursive formula, following Kalisch & Bühlmann (2007):

```math
\\rho_{x,y} = \\dfrac{\\rho_{x,y|Z\\setminus{z}} - \\rho_{x,z|Z\\setminus{z}} \\times \\rho_{y,z|Z\\setminus{z}} } { \\sqrt{1 - \\rho_{x,z|Z\\setminus{z}}^2} \\times \\sqrt{1 - \\rho_{y,z|Z\\setminus{z}}^2}}
```

where ``z`` is any element from ``Z``.
"""
function partialcor_recursive(ctx::RecursivePartialCorrContext, x, y, Z, depth = 0)
    R = ctx.R
	l = length(Z)
	depth == l && return R[x, y]
	
	@inbounds z = Z[l - depth]
	ρ̃_xz_Zmz = partialcor_recursive(ctx, x, z, Z, depth + 1)
	ρ̃_yz_Zmz = partialcor_recursive(ctx, y, z, Z, depth + 1)
	ρ̃_xy_Zmz = partialcor_recursive(ctx, x, y, Z, depth + 1)
	
	return (ρ̃_xy_Zmz - ρ̃_xz_Zmz * ρ̃_yz_Zmz) / √((1 - ρ̃_xz_Zmz ^ 2) * (1 - ρ̃_yz_Zmz ^ 2))
end

# ╔═╡ ffe378da-68b5-47ba-8bac-7d8293c6bbcf
md"""
The last necessary component of this function is an independence test, which I also implemented based on Kalisch and Bühlmann (2007).
"""

# ╔═╡ b227f0f1-e7db-4de2-b0ee-766a6da29208
"""
	independence_test(ctx::RecursivePartialCorrContext, x, y, z)

Return the p-value of an conditional independence test

Tests whether two variables of indices `x` and `y` in the original data matrix are independent conditionally on set of variables of indices `Z`.

``H_0: X \\perp Y | Z``

``H_1:  X \\centernot{\\perp} Y | Z``

The test is performed with Fisher's z transformation following Kalisch & Bühlmann (2007).
"""
function independence_test(ctx::RecursivePartialCorrContext, x, y, Z)
    R, m, n = ctx.R, ctx.m, ctx.n
	rhopar = partialcor_recursive(ctx, x, y, Z, 0)
	z = 0.5 * (log((1 + rhopar) /(1 - rhopar)))
	obs = √(m - length(Z) - 3) * abs(z)
	pval = 2 * (1 - cdf(Normal(0, 1), obs))

	return pval
end

# ╔═╡ d7af2120-8885-4dc0-8184-6be79c7f2b26
"""
	peter_clark(data, alpha = 0.05)

Infer a causal graph skeleton using the classical Peter-Clark algorithm.

This implementation is based on the algorithm description from Le et al. (2019).
"""
function peter_clark(data, alpha = 0.05)
    ctx = RecursivePartialCorrContext(data)
    n = ctx.n
	G = complete_graph(n)
	d = 0
	while true
		for e in edges(G)
			x, y = src(e), dst(e)
			if (degree(G, x) - 1) ≥ d
				for Z in powerset(setdiff(neighbors(G, x), y), d, d)
					pval = independence_test(ctx, x, y, Z)
					if pval > alpha
						rem_edge!(G, x, y)
						break
					end
				end
			end
		end
		
		stop_condition_counter = 0
		for e in edges(G)
			x, y = src(e), dst(e)
			stop_condition_counter += length(neighbors(G, x)) - 1 < d
		end
		
		d += 1
		stop_condition_counter == ne(G) && break
	end

	return G
end

# ╔═╡ 2d4068be-1b19-43f0-a8bd-7c9eb73284b3
md"""
## Inference

I used the `peter_clark` function to inter the simulated graph based on the two datasets.
"""

# ╔═╡ 21f1311c-6418-4d11-9c9c-8cc8483707d7
ĝsim1 = peter_clark(sim1, 0.05)

# ╔═╡ e3f75878-5dde-4900-b680-b73883fba54b
ĝsim2 = peter_clark(sim2, 0.05)

# ╔═╡ cec108da-d92a-4b9a-8fa1-0eb73e196cdd
md"""
We can take a look at the reconstructed graphs, next to the original simulated graph.
"""

# ╔═╡ f904927d-a480-4960-8ab3-3db8a35f8d86
md"""
It is difficult to assess the quality of these estimations just by sight, so I compute the confusion matrix based on their adjacency matrices using the function below.
"""

# ╔═╡ 3b12f6c9-bb7a-4b6d-817d-b5d5add57a81
"""
	confusion_matrix(X, X̂)

Compute a confusion matrix between a known variable X and an estimated variable X̂.

The matrix has the following structure:

|                 | Pred. Positive | Pred. Negative |
|-----------------|:--------------:|:--------------:|
| **Actual Positive** | True Positives  | False Negatives |
| **Actual Negative** | False Positives | True Negatives  |
"""
function confusion_matrix(X, X̂)
	M = zeros(Int, 2, 2)
	for (x, x̂) in zip(UpperTriangular(X), UpperTriangular(X̂))
		M[1, 1] += x && x̂
		M[2, 1] += (! x) && x̂
		M[1, 2] += x && (! x̂)
		M[2, 2] += ! (x && x̂)
	end

	return M
end

# ╔═╡ 279e38cc-2371-4250-ab1a-a73cad236b7d
cm_sim1 = confusion_matrix(Asim .> 0, adjacency_matrix(ĝsim1) .> 0);

# ╔═╡ a78b3a2a-2f69-469a-8af9-adaa2a0bfa5f
md"""
Confusion table with the `sim1` dataset:

|                 | Pred. Positive | Pred. Negative |
|-----------------|:--------------:|:--------------:|
| **Actual Positive** | $(cm_sim1[1, 1])  | $(cm_sim1[1, 2]) |
| **Actual Negative** | $(cm_sim1[2, 1]) | $(cm_sim1[2, 2])  |
"""

# ╔═╡ 64e5665e-f85a-493c-8882-e74137725162
cm_sim2 = confusion_matrix(Asim .> 0, adjacency_matrix(ĝsim2) .> 0);

# ╔═╡ 9469f111-b13a-4be2-9666-36db7d2a41ae
md"""
Confusion table with the `sim2` dataset:

|                 | Pred. Positive | Pred. Negative |
|-----------------|:--------------:|:--------------:|
| **Actual Positive** | $(cm_sim2[1, 1])  | $(cm_sim2[1, 2]) |
| **Actual Negative** | $(cm_sim2[2, 1]) | $(cm_sim2[2, 2])  |
"""

# ╔═╡ fa4e40ff-518a-4bf4-9dd9-7e3a8d67472e
md"""
The confusion tables show that the Peter-Clark algorithm performs better in 3 of the 4 quantities of interest with the `sim2` dataset. Reassuringly, this was the expected result, since partial correlations can be estimated more accurately with greater numbers of samples.
"""

# ╔═╡ 632c268e-2188-4a0d-bcab-92a70d6bcf2d
md"""
## Evaluation

In the Peter-Clark algorithm, the alpha parameter is interpreted as a tunable sensitivity threshold for the estimation of the entire graph topology. The focus is not the determination of the "statistical significance" of each edge.

In the estimated graphs shown above, alpha was set to 0.05. I explored the performance of the Peter-Clark algorithm under several values of alpha with both simulated datasets.

I created the variable `alphas` with the following values:
"""

# ╔═╡ 19965fe0-a06d-4483-8845-0c8b2f117b02
begin
	alphas = [5 * 10 ^ i for i in (-5:0.25:-1.0)]
end

# ╔═╡ d2284ea5-412a-4513-9acf-e8e5d1581edd
md"""
I computed precision and recall curves to evaluate the performance with `sim1`. These curves can be derived from the quantities in a confusion matrix. They are defined thus:

```math
Precision = \dfrac{True\ Positives}{True\ Positives + False\ Positives}
```
```math
Recall = \dfrac{True\ Positives}{True\ Positives + False\ Negatives}
```

In other words, precision measures the proportion of edges in the inferred graph that are correct, and recall measures the proportion of edges of the true graph that are recovered in the inferred graph.
"""

# ╔═╡ b3d364cd-91e4-44f5-b3b4-e9926d9d8142
md"""
I created a data frame with columns to store TP, FN, FP, and TN at each alpha value.
"""

# ╔═╡ 7d25e5ed-5e4d-49d4-ad2d-51c69802c130
performance_sim1 = DataFrame(
		α = alphas,
		TP = zeros(Int, length(alphas)),
		FN = zeros(Int, length(alphas)),
		FP = zeros(Int, length(alphas)),
		TN = zeros(Int, length(alphas)),
	);

# ╔═╡ fb0f709b-3226-43cf-b844-b2256df82e14
md"""
And inferred the graph with each value of alpha to fill in the values.
"""

# ╔═╡ 6ab2b9e1-9148-4406-a661-fbbd89973e54
for (i, alpha) in enumerate(alphas)
	ĝ_alpha = peter_clark(sim1, alpha)
	Â_alpha = adjacency_matrix(ĝ_alpha) .> 0
	C = confusion_matrix(Asim .> 0, Â_alpha)
	performance_sim1[i, :TP] = C[1, 1]
	performance_sim1[i, :FN] = C[1, 2]
	performance_sim1[i, :FP] = C[2, 1]
	performance_sim1[i, :TN] = C[2, 2]
end

# ╔═╡ 0e8003d1-a34a-469d-8b6e-60b99a27fc8f
md"""
With that information, I computed precision and recall for all the vaues of alpha.
"""

# ╔═╡ 181b6b30-c0f0-40fc-95c5-25ab694a23ce
performance_sim1.Precision =
	performance_sim1.TP ./ (performance_sim1.TP .+ performance_sim1.FP);

# ╔═╡ b50b7273-8b21-490b-a52d-b37355fe3f47
performance_sim1.Recall =
	performance_sim1.TP ./ (performance_sim1.TP .+ performance_sim1.FN);

# ╔═╡ 3b474bdf-661e-4b5e-8353-685321052cc1
performance_sim1

# ╔═╡ 9dccd44c-7959-49e0-948f-dd487795e40b
begin
	performance_sim2 = DataFrame(
		α = alphas,
		TP = zeros(Int, length(alphas)),
		FN = zeros(Int, length(alphas)),
		FP = zeros(Int, length(alphas)),
		TN = zeros(Int, length(alphas)),
	);
	for (i, alpha) in enumerate(alphas)
		ĝ_alpha = peter_clark(sim2, alpha)
		Â_alpha = adjacency_matrix(ĝ_alpha) .> 0
		C = confusion_matrix(Asim .> 0, Â_alpha)
		performance_sim2[i, :TP] = C[1, 1]
		performance_sim2[i, :FN] = C[1, 2]
		performance_sim2[i, :FP] = C[2, 1]
		performance_sim2[i, :TN] = C[2, 2]
	end
	performance_sim2.Precision =
		performance_sim2.TP ./ (performance_sim2.TP .+ performance_sim2.FP);
	performance_sim2.Recall =
		performance_sim2.TP ./ (performance_sim2.TP .+ performance_sim2.FN);
end;

# ╔═╡ 52e5f1a4-bd3d-4f10-ac45-2c42d641ff0f
md"""
After doing the same for `sim2`, I plotted the curves. In this case, as I was most interested in optimizing alpha, I plotted precision and recall against alpha, rather than agains each other.
"""

# ╔═╡ 63ff8516-05bc-4c37-b80e-21b35b8abecd
begin
	fig3 = Figure()
	fig3ax1 = Axis(fig3[1, 1], title="Performance with sim1")
	fig3ax2 = Axis(fig3[2, 1], title="Performance with sim2")
	lines!(fig3ax1, alphas, performance_sim1.Precision, label = "Precision")
	lines!(fig3ax1, alphas, performance_sim1.Recall, label = "Recall")
	lines!(fig3ax2, alphas, performance_sim2.Precision, label="Precision")
	lines!(fig3ax2, alphas, performance_sim2.Recall, label="Recall")
	ylims!(fig3ax1, 0.0, 1.005)
	ylims!(fig3ax2, 0.0, 1.005)
	fig3ax2.xlabel = "α threshold"
	hidexdecorations!(fig3ax1, ticks=false, grid=false)
	fig3ax1.ylabel = fig3ax2.ylabel = "Precision or Recall"
	axislegend(fig3ax1, position=:lt)
	fig3
end

# ╔═╡ 65cf0d67-b3e9-4402-b58f-603829243ca3
md"""
Again, we can see that `sim2` performs better markedly than `sim1` over the range of values of alpha. In the `sim2` curves, the recall levels hover over $(round(performance_sim2.Recall |> minimum, digits=1)) and precision values are between $(round(performance_sim2.Precision |> minimum, digits=1)) and $(round(performance_sim2.Precision |> maximum, digits=1)). However, the non-monotonic trajectories of the precision and recall curves are quite perplexing. There is the possibility of bug in my code. I did some experiments to check that the results of `partialcor_recursive` are very close to results of partial correlation implementations in the packages Associations and StatsBase (see Appendix for an example). A problem in the `peter_clark` function is possible, and I did not have the time to compare its accuracy to other much slower implementations.

I also considered the possibility that the fact that the classical Peter-Clark algorithm is not invariant to edge order could interact with the alpha parameter variations resulting in the performance fluctuations. However, Colombo & Maathuis (2014, fig. 5) show monotonic responses of the precision to variations of alpha with the classical Peter-Clark algorithm.

Therefore, the most likely scenario is that there is some important detail of implementation that is affecting my Peter-Clark algorithm, and probably depressing its performance. Fortunately, enough signal is captured to display the expected behaviour of better performance with the larger dataset.
"""

# ╔═╡ a5448e5d-2a60-43f0-9eb7-e2b66fcf560e
md"""
## Application to Sachs data

The Sachs dataset comes from Sachs et al. (2005), who used Bayesian network approaches to derive protein-signalling causal graph from multiparameter single-cell data. The data consists of 11 variables representing phosphorilation levels of 11 molecular species in a well-studied signalling pathway. It contains 7466 samples obtained from experimental manipulations performed to capture a wide range of the dynamic behaviour of the system.
"""

# ╔═╡ 3738fe77-a9cf-404b-8484-710bcff167bd
sachs = CSV.read(Downloads.download("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/sachs.data"), DataFrame, header=["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"])

# ╔═╡ 38858138-7b9e-4168-8bb1-4ced661182a9
md"""
Sachs et al. (2005) provided the following graph based on molecular interactions previously reported in the literature, which I took as the "ground truth" to compare my estimations against. In reality, some of the links in this graphs were less well supported than the others; I did not differentiate between them, for simplicity.
"""

# ╔═╡ 5b7a1648-bd64-4453-8545-4cef88fbc33e
begin
	molid = Dict{String,Int}(x => i for (i, x) in enumerate(names(sachs)))   # Molecule name => Vertex index

	gsachs = SimpleDiGraph(11)
	add_edge!(gsachs, molid["PKC"], molid["Jnk"])
	add_edge!(gsachs, molid["PKC"], molid["P38"])
	add_edge!(gsachs, molid["PKC"], molid["Raf"])
	add_edge!(gsachs, molid["PKC"], molid["Mek"])
	add_edge!(gsachs, molid["PKC"], molid["PKA"])
	add_edge!(gsachs, molid["Plcg"], molid["PKC"])
	add_edge!(gsachs, molid["Plcg"], molid["PIP3"])
	add_edge!(gsachs, molid["Plcg"], molid["PIP2"])
	add_edge!(gsachs, molid["PIP3"], molid["PIP2"])
	add_edge!(gsachs, molid["PIP3"], molid["Akt"])
	add_edge!(gsachs, molid["PIP2"], molid["PKC"])
	add_edge!(gsachs, molid["PKA"], molid["Akt"])
	add_edge!(gsachs, molid["PKA"], molid["Jnk"])
	add_edge!(gsachs, molid["PKA"], molid["Mek"])
	add_edge!(gsachs, molid["PKA"], molid["Raf"])
	add_edge!(gsachs, molid["PKA"], molid["Erk"])
	add_edge!(gsachs, molid["PKA"], molid["P38"])
	add_edge!(gsachs, molid["Erk"], molid["Akt"])
	add_edge!(gsachs, molid["Mek"], molid["Erk"]);
	add_edge!(gsachs, molid["Raf"], molid["Mek"])
	gsachs
end

# ╔═╡ 6db3659c-e5ee-40a1-829f-41b2ecb5232b
md"""
As I have some manner of "ground truth", I can apply the same steps used for the simulated data.

1. I computed the adjacency matrix of the "true graph".
"""

# ╔═╡ 9f66b1e6-fe66-4661-8697-6c689235302c
Asachs = adjacency_matrix(gsachs, dir=:both);

# ╔═╡ 5932d03e-2037-43e5-ba4f-0558a51d8d87
md"""
2. I created a data frame to hold the basic performance values.
"""

# ╔═╡ 9e38cb85-4c1d-4b80-b74c-eee535e0182e
performance_sachs = DataFrame(
		α = alphas,
		TP = zeros(Int, length(alphas)),
		FN = zeros(Int, length(alphas)),
		FP = zeros(Int, length(alphas)),
		TN = zeros(Int, length(alphas)),
	);

# ╔═╡ bdfc1cd1-745b-4a59-81c4-49af5c22bec5
md"""
3. I estimated the graph with all the values of alpha to compute the values.
"""

# ╔═╡ 823a6c98-2526-475f-b078-52781fb5c933
begin
	ĝsachs_alphas = Vector{Graph}(undef, length(alphas))
	for (i, alpha) in enumerate(alphas)
		ĝsachs_alphas[i] = peter_clark(Matrix(sachs), alpha)
		Â_alpha = adjacency_matrix(ĝsachs_alphas[i]) .|> Bool
		C = confusion_matrix(Bool.(Asachs), Â_alpha)
		performance_sachs[i, :TP] = C[1, 1]
		performance_sachs[i, :FN] = C[1, 2]
		performance_sachs[i, :FP] = C[2, 1]
		performance_sachs[i, :TN] = C[2, 2]
	end
end;

# ╔═╡ 4d896d68-c878-454b-99b3-8c5faa4b7c97
md"""
4. And I finally computed precision and recall.
"""

# ╔═╡ 0143d7bc-af25-41d1-a6d5-79654abcbe23
performance_sachs.Precision =
	performance_sachs.TP ./ (performance_sachs.TP .+ performance_sachs.FP);

# ╔═╡ 0266b586-0afb-4e5e-b194-b8ca71dd5d67
performance_sachs.Recall = performance_sachs.TP ./ (performance_sachs.TP .+ performance_sachs.FN);

# ╔═╡ 548c6c7d-4870-4b33-b2be-b977943fcc9c
begin
	fig6, fig6ax1 = lines(alphas, performance_sachs.Precision, label="Precision")
	lines!(fig6ax1, alphas, performance_sachs.Recall, label="Recall")
	fig6ax1.xlabel = "α threshold"
	fig6ax1.ylabel = "Precision or Recall"
	axislegend(position=:lt)
	fig6
end

# ╔═╡ 934fe60d-97a8-4aaa-9e75-3c2d7689f64e
md"""
As the curves show, the Peter-Clark algorithm unfortunately picks up too many false edges, producing really mediocre precision values. And, again, the precision curve is not monotonic. At least, the high recall levels are of some consolation.

The best precision values were obtained with alpha = 0.05, so it's a good graph to visualize.
"""

# ╔═╡ e2eae386-4a44-4ae9-99da-29c8aed30091
md"""
## Conclusion

Despite of the troubles with the performance behaviour of the algorithm, which delayed the submission of this report, implementing the method of [Peter] Spirtes and [Clark] Glymour was a stimulating experience that aforded me a better understanding of its workings. With more time, I would be very interested in exploring the true performance of the classical Peter-Clark algorithm in a range of sound implementations, and appreciate its true potential for causal discovery in biological systems.

Another area of exploration that remains is the choice of the alpha value in real research scenarios where a ground truth is unknown. I wonder if there is some principled way to determine a value of alpha based on the magnitudes of the weakest relationships that one wishes to detect.
"""

# ╔═╡ 7c9160bf-980e-47c3-adb3-62a79a6830ac
md"""
## References

Colombo D, Maathuis MH. 2014. Order-independent constraint-based causal structure learning. J. Mach. Learn. Res. 15:3741–3782.

Dee LE, Ferraro PJ, Severen CN, Kimmel KA, Borer ET, Byrnes JEK, Clark AT, Hautier Y, Hector A, Raynaud X, Reich PB, Wright AJ, Arnillas CA, Davies KF, MacDougall A, Mori AS, Smith MD, Adler PB, Bakker JD, Brauman KA, Cowles J, Komatsu K, Knops JMH, McCulley RL, Moore JL, Morgan JW, Ohlert T, Power SA, Sullivan LL, Stevens C, Loreau M. 2023. Clarifying the effect of biodiversity on productivity in natural ecosystems with longitudinal data and methods for causal inference. Nature Communications 14:1–12. DOI: 10.1038/s41467-023-37194-5.

Kalisch M, Bühlman P. 2007. Estimating high-dimensional directed acyclic graphs with the PC-algorithm. Journal of Machine Learning Research 8.

Le TD, Hoang T, Li J, Liu L, Liu H, Hu S. 2019. A Fast PC Algorithm for High Dimensional Causal Discovery with Multi-Core PCs. IEEE/ACM transactions on computational biology and bioinformatics 16:1483–1495. DOI: 10.1109/tcbb.2016.2591526.

Sachs K, Perez O, Pe’er D, Lauffenburger DA, Nolan GP. 2005. Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data. Science 308:523–529. DOI: 10.1126/science.1105809.

"""

# ╔═╡ 50cdc4fb-b49e-4145-abde-2d65309e5451
md"""
## Appendix: Validating the recursive partial correlations function

I took the data from `sim2` and generated random pairs of indices for the x and y variables, and sets of indices for the Z conditional sets. Then I plotted the results of the recursive partial correlation algorithm against the `partialcor` function from the StatsBase package, which is based on linear regression.
"""

# ╔═╡ 43c08653-2461-4d21-b5bd-b94669a206d9
varpairs = [sample(1:15, 2; replace=false) for _ in 1:200]

# ╔═╡ 777f23fc-d729-4ac4-bfff-22438f78e2ed
Z_sets = [setdiff(1:15, x)[2:sample(3:13)] for x in varpairs]

# ╔═╡ 807a9d2a-2ce3-41e8-b812-80be412b9e3a
trial_params = collect(zip(varpairs, Z_sets));

# ╔═╡ 169453eb-fb6d-4699-9781-ed2af1193b3c
@time rho_stats = [partialcor(sim2[:, x], sim2[:, y], sim2[:, Z]) for ((x, y), Z) in trial_params];

# ╔═╡ c682d05b-fc09-42e0-b03c-1725aeb6449a
@time begin
ctx = RecursivePartialCorrContext(sim2)
rho_recursive = [partialcor_recursive(ctx, x, y, Z) for ((x, y), Z) in trial_params]
end;

# ╔═╡ 82b5a57c-013c-45e1-87aa-8c4dd281e7dd
plot(rho_stats, rho_recursive)

# ╔═╡ 4528a800-ccc5-473b-8bf8-8077fd4586ad
md"""
## Backstage: Utility functions
"""

# ╔═╡ 51e53966-d252-4695-afdd-e99128f87792
"""
	gplot!(ax, g::AbstractGraph, labels)
	gplot(g::AbstractGraph, labels)

Plot a graph with custom styling. Labels are an optional string vector. If not given, node indices are used.
"""
function gplot!(ax, g::AbstractGraph, labels=string.(1:nv(g)))
	graphplot!(ax, g, node_color = "#FFFFFF", edge_color = "#555555", node_size=40, nlabels_align = (:center, :center); node_strokewidth=1, arrow_shift=0.5, arrow_size=15, nlabels = labels)
	hidedecorations!(ax)

	return ax
end

# ╔═╡ 32ad86fb-cc07-462d-955b-c20458309eef
begin
	fig1 = Figure()
	ax1 = Axis(fig1[1, 1], title = "Simulated graph")
	gplot!(ax1, gsim)
	fig1
end

# ╔═╡ 051bb113-db0e-4c68-bd6e-8ff10560be9b
begin
	fig2 = Figure(size=(700, 700))
	fig2ga = fig2[1, 1] = GridLayout()
	fig2ax1 = Axis(fig2ga[1, 1:2], title = "Simulated graph")
	fig2ax2 = Axis(fig2ga[2, 1], title = "Graph inferred from sim1")
	fig2ax3 = Axis(fig2ga[2, 2], title = "Graph inferred from sim2")
	# Label(fig2ga[1, 1:2, Top()], "Figure 2. Simulated and estimated graphs", font=:bold, alignmode=Outside())
	gplot!(fig2ax1, gsim)
	gplot!(fig2ax2, ĝsim1)
	gplot!(fig2ax3, ĝsim2)
	fig2
end

# ╔═╡ 137de75c-dd6a-4a69-baa3-0eb275eca5d5
begin
	fig5 = Figure()
	axfig5 = Axis(fig5[1, 1], title="Signalling pathway graph expected from background knowledge")
	gplot!(axfig5, gsachs, names(sachs))
	fig5
end

# ╔═╡ b071293c-f4af-4380-b745-5ce8cf940dfa
begin
	fig7 = Figure()
	fig7ax1 = Axis(fig7[1, 1], title="Ground truth graph from Sachs et al. (2005)")
	fig7ax2 = Axis(fig7[2, 1], title="Estimated graph with α = 0.05")
	gplot!(fig7ax1, gsachs, names(sachs))
	gplot!(fig7ax2, peter_clark(Matrix(sachs), 0.05), names(sachs))
	fig7
end

# ╔═╡ 42f4c548-d032-45f4-ac89-ed27e4b194f0
function gplot(g::AbstractGraph, labels=string.(1:nv(g)))
	fig = Figure()
	ax = Axis(fig[1, 1])
	gplot!(ax, g, labels)
	hidedecorations!(ax)

	return fig
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
GraphMakie = "1ecd5474-83a3-4783-bb4f-06765db800d2"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CSV = "~0.10.15"
Combinatorics = "~1.0.2"
DataFrames = "~1.7.0"
Distributions = "~0.25.116"
GLMakie = "~0.10.18"
GraphMakie = "~0.5.12"
Graphs = "~1.12.0"
SimpleWeightedGraphs = "~1.4.0"
Statistics = "~1.11.1"
StatsBase = "~0.34.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "7bee7f2f64508ec6aaceaefc239221d8797d9408"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "e1371a23fd9816080c828d0ce04373857fe73d33"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.3"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7901a6117656e29fa2c74a58adb682f380922c47"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.116"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f42a5b1e20e009a43c3646635ed81a9fcaccb287"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+2"

[[deps.Extents]]
git-tree-sha1 = "81023caa0021a41712685887db1fc03db26f41f5"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "8cc47f299902e13f90405ddb5bf87e5d474c0d38"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "d52e255138ac21be31fa633200b65e4e71d26802"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW]]
deps = ["GLFW_jll"]
git-tree-sha1 = "7ed24cfc4cb29fb10c0e8cca871ddff54c32a4c3"
uuid = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98"
version = "3.4.3"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GLMakie]]
deps = ["ColorTypes", "Colors", "FileIO", "FixedPointNumbers", "FreeTypeAbstraction", "GLFW", "GeometryBasics", "LinearAlgebra", "Makie", "Markdown", "MeshIO", "ModernGL", "Observables", "PrecompileTools", "Printf", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "8753fba3356131357b5cd02500fe80c3668535d0"
uuid = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
version = "0.10.18"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "59107c179a586f0fe667024c5eb7033e81333271"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.2"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "f4ee66b6b1872a4ca53303fbb51d158af1bf88d4"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.4.0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0224cce99284d997f6880a42ef715a37c99338d1"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.2+2"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.GraphMakie]]
deps = ["DataStructures", "GeometryBasics", "Graphs", "LinearAlgebra", "Makie", "NetworkLayout", "PolynomialRoots", "SimpleTraits", "StaticArrays"]
git-tree-sha1 = "c8c3ece1211905888da48e16f438af85e951ea55"
uuid = "1ecd5474-83a3-4783-bb4f-06765db800d2"
version = "0.5.12"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1dc470db8b1131cfc7fb4c115de89fe391b9e780"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.12.0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "dc6bed05c15523624909b3953686c5f5ffa10adc"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "LinearAlgebra", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "ffb76d09ab0dc9f5a27edac2acec13c74a876cc6"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.21"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

    [deps.IntervalArithmetic.weakdeps]
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3447a92280ecaad1bd93d3fce3d408b6cfff8913"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.0+1"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78e0f4b5270c4ae09c7c5f78e77b904199038945"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+2"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a7f43994b47130e4f491c3b2dbe78fe9e2aed2b3"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.0+2"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d841749621f4dcf0ddc26a27d1f6484dfc37659a"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.2+1"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "b404131d06f7886402758c9ce2214b636eb4d54a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9d630b7fb0be32eeb5e8da515f5e8a26deb457fe"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.2+1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "be3051d08b78206fb5e688e8d70c9e84d0264117"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.21.18"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "9019b391d7d086e841cbeadc13511224bd029ab3"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.8.12"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "f45c8916e8385976e1ccd055c9874560c257ab13"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "14a12d9153b1a1a22d669eede58b2ea2164ff138"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.4.13"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.ModernGL]]
deps = ["Libdl"]
git-tree-sha1 = "b76ea40b5c0f45790ae09492712dd326208c28b2"
uuid = "66fc600b-dfda-50eb-8b99-91cfa97b1301"
version = "1.1.7"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkLayout]]
deps = ["GeometryBasics", "LinearAlgebra", "Random", "Requires", "StaticArrays"]
git-tree-sha1 = "0c51e19351dc1eecc61bc23caaf2262e7ba71973"
uuid = "46757867-2c16-5918-afeb-47bfcb05e46a"
version = "0.4.7"
weakdeps = ["Graphs"]

    [deps.NetworkLayout.extensions]
    NetworkLayoutGraphsExt = "Graphs"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+3"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PolynomialRoots]]
git-tree-sha1 = "5f807b5345093487f733e520a1b7395ee9324825"
uuid = "3a141323-8675-5d76-9d11-e1df1406c778"
version = "1.0.0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "52af86e35dd1b177d051b12681e1c581f53c281b"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "9537ef82c42cdd8c5d443cbc359110cbb36bae10"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.21"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "3c0faa42f2bd3c6d994b06286bba2328eae34027"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.2"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "01915bfcd62be15329c9a07235447a89d588327c"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.1"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "15e637a697345f6743674f1322beefbc5dcd5cfc"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.3+2"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2b0e27d52ec9d8d483e2ca0b72b3cb1a8df5c27a"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+3"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "02054ee01980c90297412e4c809c8694d7323af3"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+3"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee57a273563e273f0f53275101cd41a8153517a"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b9ead2d2bdb27330545eb14234a2e300da61232e"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+3"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+2"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "7dfa0fd9c783d3d0cc43ea1af53d69ba45c447df"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+3"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "ccbb625a89ec6195856a50aa2b668a5c08712c94"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.4.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "35976a1216d6c066ea32cba2150c4fa682b276fc"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.0+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc541bb19ed5b0ede95581fb2e41ecf179527d2"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.6.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─bf6370e0-d373-11ef-1e7b-45131113726b
# ╟─6e081a14-6225-4b8f-9bb4-a54fe93050b2
# ╟─5703ce81-81be-4c98-b108-bfc0e343cb69
# ╠═160e3d99-9fcd-473e-a2a2-85e84fba9dc9
# ╟─e5df4d5c-1305-4c30-8d11-3aa12be45923
# ╟─b009213d-1ff8-49af-a662-8aea364ba3ce
# ╟─25d4ea7a-c325-4c6e-aeaf-dd127870c073
# ╠═4d0d3dd6-e9c5-4295-a9a3-0645081284b5
# ╠═41f9171f-1b7f-48eb-9127-1d0bb82699dd
# ╟─46d26a14-baa3-450c-98c9-375764b9a9e6
# ╟─32ad86fb-cc07-462d-955b-c20458309eef
# ╟─488e68e9-6f91-4060-b445-6c0dad0a1f13
# ╠═a94e608c-c98c-4ffe-8bf6-e33a93b923a3
# ╟─d601ac44-b19e-4398-ae3c-f80e5b09fdf1
# ╟─7a539c17-8da2-4086-a849-ebb78263b0fb
# ╠═d7b2ab9a-2fb6-479c-a825-21a1bca40f95
# ╠═78e29fc3-2989-4134-ae6c-e216f0724c1e
# ╠═af7e012a-7eb4-4095-98eb-bfed308314ea
# ╟─e364a4d3-840f-4539-9114-4eb96f28d3c8
# ╠═d7af2120-8885-4dc0-8184-6be79c7f2b26
# ╟─16be2434-fa51-4fbf-a73c-ba8a179703df
# ╠═36c275be-1319-49c0-aaac-e33f076e074b
# ╠═31706282-8f2a-42bd-90d2-daef9977d1ac
# ╟─ffe378da-68b5-47ba-8bac-7d8293c6bbcf
# ╠═b227f0f1-e7db-4de2-b0ee-766a6da29208
# ╟─2d4068be-1b19-43f0-a8bd-7c9eb73284b3
# ╠═21f1311c-6418-4d11-9c9c-8cc8483707d7
# ╠═e3f75878-5dde-4900-b680-b73883fba54b
# ╟─cec108da-d92a-4b9a-8fa1-0eb73e196cdd
# ╟─051bb113-db0e-4c68-bd6e-8ff10560be9b
# ╟─f904927d-a480-4960-8ab3-3db8a35f8d86
# ╠═3b12f6c9-bb7a-4b6d-817d-b5d5add57a81
# ╠═279e38cc-2371-4250-ab1a-a73cad236b7d
# ╟─a78b3a2a-2f69-469a-8af9-adaa2a0bfa5f
# ╠═64e5665e-f85a-493c-8882-e74137725162
# ╟─9469f111-b13a-4be2-9666-36db7d2a41ae
# ╟─fa4e40ff-518a-4bf4-9dd9-7e3a8d67472e
# ╟─632c268e-2188-4a0d-bcab-92a70d6bcf2d
# ╟─19965fe0-a06d-4483-8845-0c8b2f117b02
# ╟─d2284ea5-412a-4513-9acf-e8e5d1581edd
# ╟─b3d364cd-91e4-44f5-b3b4-e9926d9d8142
# ╠═7d25e5ed-5e4d-49d4-ad2d-51c69802c130
# ╟─fb0f709b-3226-43cf-b844-b2256df82e14
# ╠═6ab2b9e1-9148-4406-a661-fbbd89973e54
# ╟─0e8003d1-a34a-469d-8b6e-60b99a27fc8f
# ╠═181b6b30-c0f0-40fc-95c5-25ab694a23ce
# ╠═b50b7273-8b21-490b-a52d-b37355fe3f47
# ╟─3b474bdf-661e-4b5e-8353-685321052cc1
# ╟─9dccd44c-7959-49e0-948f-dd487795e40b
# ╟─52e5f1a4-bd3d-4f10-ac45-2c42d641ff0f
# ╟─63ff8516-05bc-4c37-b80e-21b35b8abecd
# ╟─65cf0d67-b3e9-4402-b58f-603829243ca3
# ╟─a5448e5d-2a60-43f0-9eb7-e2b66fcf560e
# ╟─3738fe77-a9cf-404b-8484-710bcff167bd
# ╟─38858138-7b9e-4168-8bb1-4ced661182a9
# ╟─5b7a1648-bd64-4453-8545-4cef88fbc33e
# ╟─137de75c-dd6a-4a69-baa3-0eb275eca5d5
# ╟─6db3659c-e5ee-40a1-829f-41b2ecb5232b
# ╠═9f66b1e6-fe66-4661-8697-6c689235302c
# ╟─5932d03e-2037-43e5-ba4f-0558a51d8d87
# ╠═9e38cb85-4c1d-4b80-b74c-eee535e0182e
# ╟─bdfc1cd1-745b-4a59-81c4-49af5c22bec5
# ╠═823a6c98-2526-475f-b078-52781fb5c933
# ╟─4d896d68-c878-454b-99b3-8c5faa4b7c97
# ╠═0143d7bc-af25-41d1-a6d5-79654abcbe23
# ╠═0266b586-0afb-4e5e-b194-b8ca71dd5d67
# ╟─548c6c7d-4870-4b33-b2be-b977943fcc9c
# ╟─934fe60d-97a8-4aaa-9e75-3c2d7689f64e
# ╟─b071293c-f4af-4380-b745-5ce8cf940dfa
# ╟─e2eae386-4a44-4ae9-99da-29c8aed30091
# ╟─7c9160bf-980e-47c3-adb3-62a79a6830ac
# ╟─50cdc4fb-b49e-4145-abde-2d65309e5451
# ╠═43c08653-2461-4d21-b5bd-b94669a206d9
# ╠═777f23fc-d729-4ac4-bfff-22438f78e2ed
# ╠═807a9d2a-2ce3-41e8-b812-80be412b9e3a
# ╠═169453eb-fb6d-4699-9781-ed2af1193b3c
# ╠═c682d05b-fc09-42e0-b03c-1725aeb6449a
# ╠═82b5a57c-013c-45e1-87aa-8c4dd281e7dd
# ╟─4528a800-ccc5-473b-8bf8-8077fd4586ad
# ╠═51e53966-d252-4695-afdd-e99128f87792
# ╟─42f4c548-d032-45f4-ac89-ed27e4b194f0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
