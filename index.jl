### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 6264e41e-e179-11ef-19c1-f135e97db7cc
# ╠═╡ show_logs = false
begin
	using Pkg
	cd(".")
	Pkg.activate(".")
	#Pkg.add(url="https://github.com/patrickm663/HMD.jl")
	#Pkg.add(name="Turing", version="0.34.1")
	#Pkg.add(["CSV", "DataFrames", "Lux", "ADTypes", "Zygote", "Optimisers", "Plots", "ComponentArrays", "Distributions", "Functors", "Tracker", "StatsPlots", "SciMLSensitivity", "OrdinaryDiffEqTsit5"])
end

# ╔═╡ ef51c95e-d5ad-455a-9631-094823b695bb
# ╠═╡ show_logs = false
using ADTypes, Lux, Optimisers, Printf, Random, CSV, Plots, DataFrames, ComponentArrays, HMD, Zygote, Statistics, Distributions, Functors, Turing, Tracker, LinearAlgebra, StatsPlots, AbstractGPs, KernelFunctions

# ╔═╡ f6696102-2894-4d54-be66-32004ea6486d
Turing.setprogress!(true);

# ╔═╡ 50b3b576-d941-4609-8469-6de51cfa1545
begin
	start_year = 1950
	end_year = 1999
	forecast_year = 2016
	extended_forecast_year = 2065
	start_age = 0
	end_age = 99
	age_length = length(start_age:end_age)
	τ₀ = 3
	τ₁ = 8#16
	T = 10
	NN_depth = 4
	cell = LSTMCell #GRUCell
	act = tanh
	list_of_countries = [
    "AUS", "AUT", "BEL", "BGR", "BLR", "CAN", "CHE", "CZE", "DNK", 
    "ESP", "EST", "FIN", "FRATNP", "GBRTENW", "GBR_NIR", "GBR_SCO", 
    "GRC", "HUN", "IRL", "ISL", "ISR", "ITA", "JPN", "LTU", "LUX", 
    "LVA", "NLD", "NOR", "NZL_NM", "POL", "PRT", "RUS", "SVK", 
    "SVN", "SWE", "TWN", "UKR", "USA"
]#HMD.get_countries()
	country = list_of_countries[1]
	gender_ = :Female
	p_ = 0.10#0.25#0.05#0.01
	model_type = "NN"
	# Hard-code
	if model_type == "NN"
		τ₀ = 2
	end

	s_age = std(start_age:end_age)
	m_age = mean(start_age:end_age)
end

# ╔═╡ 9f21c322-25bd-434f-9f9d-1a931e8199ea
begin
	lr = 2e-2
	opt = AdamW(lr)
	n_epochs = 2_000 # Max epochs
end

# ╔═╡ d1eb691b-481f-45d5-b736-7f99f4b0b4d2
sample_type = "MCMC"

# ╔═╡ a0132c58-f427-4c88-ba45-bd8b9d9f98d4
list_of_countries

# ╔═╡ 5378ee86-303b-4269-88f3-6eeccc30cb15
begin
	min_max(x, xmin, xmax) = 2.0f0*(x - xmin) / (xmax - xmin) - 1.0f0
	standardise(x, x_mu, x_sigma) = (x - x_mu) / (x_sigma + 1.0f-6)
end

# ╔═╡ 93779239-cd66-4be2-b70f-c7872a29a29f
function LSTM(in_dims, hidden_dims, out_dims; depth=1, cell=LSTMCell, dropout_p=0.0f0)
	@assert depth > 0
	@assert cell ∈ [LSTMCell, GRUCell, RNNCell]
	if depth == 1
		return Chain(
			Recurrence(cell(in_dims => hidden_dims); return_sequence=false),
			#Dropout(dropout_p),
			Dense(hidden_dims => out_dims, identity)
			)
	elseif depth == 2
		return Chain(
			Recurrence(cell(in_dims => hidden_dims); return_sequence=true),
			Recurrence(cell(hidden_dims => hidden_dims); return_sequence=false),
			#Dropout(dropout_p),
			Dense(hidden_dims => out_dims, identity)
		)
	elseif depth > 2
		return Chain(
			Recurrence(cell(in_dims => hidden_dims); return_sequence=true),
			[Recurrence(cell(hidden_dims => hidden_dims); return_sequence=true) for _ in 1:(depth-2)],
			Recurrence(cell(hidden_dims => hidden_dims); return_sequence=false),
			#Dropout(dropout_p),
			Dense(hidden_dims => out_dims, identity)
		)
	end
end

# ╔═╡ 7970d4a4-48a2-4f9a-860e-cd0d2b999957
function CNN(out_dims, hidden_dims; act=tanh)
	return Chain(
		Conv((3, 3), 1 => hidden_dims, act, stride=1, pad=0),
		MeanPool((2, 2)),
		Conv((3, 3), hidden_dims => hidden_dims, act, stride=1, pad=0),
		MeanPool((2, 2)),
		GlobalMeanPool(),
		FlattenLayer(),
		Chain(
			Dense(hidden_dims => out_dims, identity)
		)
	)
end

# ╔═╡ 8fef40e0-4528-4224-9d9e-6e306b626f7d
#=
function NNODE(in_dims, hidden_dims, out_dims; sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()), dropout_p=0.0f0)
# Source: https://lux.csail.mit.edu/stable/tutorials/intermediate/1_NeuralODE
	
	function NeuralODE(
	        model::Lux.AbstractLuxLayer; solver=Tsit5(), tspan=(0.0f0, 1.0f0), kwargs...)
	    return @compact(; model, solver, tspan, kwargs...) do x, p
	        dudt(u, p, t) = vec(model(reshape(u, size(x)), p))
	        # Note the `p.model` here
	        prob = ODEProblem(ODEFunction{false}(dudt), vec(x), tspan, p.model)
	        @return solve(prob, solver; kwargs...)
	    end
	end

	rbf(x) = exp(-(x^2))

	internal_NN =  Chain(
				Dense(in_dims => hidden_dims, tanh), 
				Dense(hidden_dims => hidden_dims, tanh), 
				Dense(hidden_dims => out_dims, tanh)
	)

	@views diffeqsol_to_array(l::Int, x::ODESolution) = reshape(last(x.u), (l, :))
	@views diffeqsol_to_array(l::Int, x::AbstractMatrix) = reshape(x[:, end], (l, :))
	
    # Construct the Neural ODE Model
    return Chain(
        NeuralODE(internal_NN;
            save_everystep=false, reltol=1.0f-3,
            abstol=1.0f-3, save_start=false, sensealg),
        Base.Fix1(diffeqsol_to_array, out_dim))
end
=#

# ╔═╡ 99851338-fed0-4963-90e0-dc09bb3d480f
function GompertzNN(hidden_dims; show_intermediate=false, act=tanh)

	function Gompertz(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, show_intermediate::Bool)
		# log(mu) = log(alpha(t) + beta(t)exp(log(c(t))*x))
		return @compact(; model₁, model₂, show_intermediate) do x
			year = reshape(x[1, :], 1, :)
			age = x[2, :]
			params = model₁(year)
			α = params[1, :] # 0.0
			β = params[2, :] # 1
			c = log.(params[3, :]) # 2

			age_year = vcat(year, reshape(x[2, :], 1, :))
			log_σ² = model₂(age_year) |> vec
			
			if show_intermediate
				@return params
			else
				@return hcat(log.(α .+ β .* exp.(c .* age)), log_σ²)
			end
		end
	end

	year_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 3, exp)
	)
	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return Gompertz(year_NN, log_sigma_NN, show_intermediate)
end

# ╔═╡ 5beb9242-c2ae-4bb3-89b0-c81ff07d5ffb
function CairnsBlakeDowdNN(hidden_dims; show_intermediate=false, act=tanh)

	function CairnsBlakeDowd(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, show_intermediate::Bool)
		# logit(q) = k1(t) + (x - x̄)*k2(t)
		return @compact(; model₁, model₂, show_intermediate) do x
			year = reshape(x[1, :], 1, :)
			age = x[2, :]
			year_params = model₁(year)
			k1 = year_params[1, :]
			k2 = year_params[2, :] 
			CBD_unlogit = exp.(k1 .+ age .* k2) # Age is already standardised with k2 absorbing 1/σ term
			CBD_qx = CBD_unlogit ./ (1 .+ CBD_unlogit)
			CBD_mx = -log.(1 .- CBD_qx)

			age_year = vcat(year, reshape(x[2, :], 1, :))
			log_σ² = model₂(age_year) |> vec
			
			if show_intermediate
				@return year_params
			else
				@return hcat(log.(CBD_mx), log_σ²)
			end
		end
	end
	
	year_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 2)
	)
	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return CairnsBlakeDowd(year_NN, log_sigma_NN, show_intermediate)
end

# ╔═╡ 760ced0b-17c0-43da-8ff8-e140c34b7d16
function CairnsBlakeDowd2NN(hidden_dims; show_intermediate=false, act=tanh)

	function CairnsBlakeDown2(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, model₃::Lux.AbstractLuxLayer, show_intermediate::Bool)
		# logit(q) = k1(t) + k2(x, t)
		return @compact(; model₁, model₂, model₃, show_intermediate) do x
			year = reshape(x[1, :], 1, :)
			age = reshape(x[2, :], 1, :)
			age_year = vcat(year, age)
			k1 = model₁(year)
			k2 = model₂(age_year)
			CBD_unlogit = exp.(k1 .+ k2)
			CBD_qx = CBD_unlogit ./ (1 .+ CBD_unlogit)
			CBD_mx = -log.(1 .- CBD_qx)

			log_σ² = model₃(age_year) |> vec
			
			if show_intermediate
				@return (k1, k2)
			else
				@return hcat(log.(CBD_mx), log_σ²)
			end
		end
	end
	
	year_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)
	age_year_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)
	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return CairnsBlakeDown2(year_NN, age_year_NN, log_sigma_NN, show_intermediate)
end

# ╔═╡ 395fc7db-7db3-403b-b793-b70749281ef4
function onehot(p, p_vec)
	oh_mat = zeros(length(p_vec), length(p))
	#@show p_vec
	for i ∈ p_vec
		for j ∈ 1:length(p)
			if p[j] == i
				oh_mat[i, j] = 1.0f0
			end
		end
	end
	return oh_mat[1:(end-1), :]
end

# ╔═╡ d8fd292f-7974-4e7b-a795-b263cea45fb7
function GMNN(hidden_dims; show_intermediate=false, act=tanh)

	function GM(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer,  model₃::Lux.AbstractLuxLayer, show_intermediate::Bool)
		# log(μ(x, t, c)) = log(M₁(x, t, c) + exp(M₂(x, t, c)))
		return @compact(; model₁, model₂, model₃, show_intermediate) do x
			year = reshape(x[1, :], 1, :)
			age = reshape(x[2, :], 1, :)
			age_year = vcat(year, age)
			
			NN_params₁ = model₁(age_year)
			NN_params₂ = model₂(age_year)

			
			log_σ² = model₃(age_year) |> vec

			if show_intermediate
				@return (NN_params₁, NN_params₂)
			else
				@return hcat(log.(NN_params₁ .+ NN_params₂), log_σ²)
			end
		end
	end

	NN₁ =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1, softplus)
	)
	
	NN₂ =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1, exp)
	)

	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return GM(NN₁, NN₂, log_sigma_NN, show_intermediate)
end

# ╔═╡ b15917eb-36a6-46c5-b05f-7140118b183a
function classicNN(hidden_dims; act=tanh)

	function classic(model₁::Lux.AbstractLuxLayer)
		return @compact(; model₁) do x
			year = reshape(x[1, :], 1, :)
			age = reshape(x[2, :], 1, :)
			age_year = vcat(year, age)
			
			@return model₁(age_year)'
		end
	end
	
	_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 2, identity)
	)

	return classic(_NN)
end

# ╔═╡ ebc03c34-6b56-43de-babf-6eb4811322a1
function LocalGLMNetNN(hidden_dims; show_intermediate=false, act=tanh)

	function LocalGLMNet(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, show_intermediate::Bool)
		return @compact(; model₁, model₂, show_intermediate) do x
			# LocalGLMNet: h(x) = β₀ + β₁(x)*x + β₂(t)*t
			xs_year = reshape(x[1, :], 1, :)
			xs_age = reshape(x[2, :], 1, :)
			_const = reshape(xs_year ./ xs_year, 1, :) # Create a vector of 1s
			year = x[1, :]
			age = x[2, :]
			params_= model(vcat(_const, xs_year, xs_age))
			α = params_[1, :]
			β₁ = params_[2, :]
			β₂ = params_[3, :]

			age_year = vcat(xs_year, xs_age)

			log_σ² = model₂(age_year) |> vec
			
			if show_intermediate
				@return year_params
			else
				@return hcat(α .+ (β₁ .* year) .+ (β₂ .* age), model₂)
			end
		end
	end
	
	_NN =  Chain(
				Dense(3 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 3, identity)
	)
	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return LocalGLMNet(_NN, log_sigma_NN, show_intermediate)
end

# ╔═╡ 4da18da3-6951-4b71-8a8e-953ecf0c0551
function SilerNN(hidden_dims; show_intermediate=false, act=tanh)

	function Siler(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, show_intermediate::Bool)
		return @compact(; model₁, model₂, show_intermediate) do x
			# Siler: h(x) = a*exp(−bx) + c + d*exp(fx)
			year = reshape(x[1, :], 1, :)
			age = x[2, :]
			year_params = model₁(year)
			α = year_params[1, :]
			β = year_params[2, :]
			γ = year_params[3, :]
			δ = year_params[4, :]
			η = year_params[5, :]

			age_year = vcat(year, reshape(x[2, :], 1, :))

			log_σ² = model₂(age_year) |> vec

			if show_intermediate
				@return year_params
			else
				@return hcat(log.(α .* exp.(-β .* age) .+ γ .+ δ .* exp.(η .* age)), log_σ²)
			end
		end
	end
	
	year_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 5, exp)
	)
	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return Siler(year_NN, log_sigma_NN, show_intermediate)
end

# ╔═╡ fb46cbe5-dbde-4ca5-b500-10e4ff5106af
function HeligmanPollardNN(hidden_dims; show_intermediate=false, act=tanh)
	
	function HeligmanPollard(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, model₃::Lux.AbstractLuxLayer, model₄::Lux.AbstractLuxLayer, show_intermediate::Bool)
		return @compact(; model₁, model₂, model₃, model₄, show_intermediate) do x
			# HP: qx/(1-qx) = A^(x+B)^C + D*exp(-E(log(x/F))^2) + GH^x
			year = reshape(x[1, :], 1, :)
			age = ((x[2, :]) .* s_age) .+ m_age
			
			infant_year_params = model₁(year) # ABC
			A = exp.(infant_year_params[1, :] .- 11.0f0)
			B = softplus.(infant_year_params[2, :])
			C = sigmoid.(infant_year_params[3, :]) .* 0.2f0

			hump_year_params = model₂(year) # DEF			
			D = exp.(hump_year_params[1, :] .- 11.0f0)
			E = softplus.(hump_year_params[2, :])
			F = 18.0f0 .+ softplus.(hump_year_params[3, :])

			secant_year_params = model₃(year)
			G = exp.(secant_year_params[1, :] .- 11.0f0)
			H = 1.0f0 .+ softplus.(secant_year_params[2, :]) .* 0.15f0

			HP = A .^ ((age .+ B) .^ C) .+ D .* exp.(-E .* (log.(age ./ F)) .^ 2) .+ (G .* (H .^ age))
			
			HP_qx = HP ./ (1.0f0 .+ HP)

			HP_mx = -log.(1.0f0 .- HP_qx .+ 1.0f-9)

			# log σ²
			age_year = vcat(year, reshape(x[2, :], 1, :))
			log_σ² = model₄(age_year) |> vec
			
			if show_intermediate
				@return year_params
			else
				@return hcat(log.(HP_mx), log_σ²)
			end
		end
	end
	
	NN_infant =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 3)
	)
	
	NN_hump =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 3)
	)

	NN_secant =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 2)
	)
	
	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return HeligmanPollard(NN_infant, NN_hump, NN_secant, log_sigma_NN, show_intermediate)
end

# ╔═╡ f3dcda6d-9110-40b0-a450-767317dd136a
m_age

# ╔═╡ b046da21-63a9-4556-ab7b-24e1c8f629e0
begin
	foo = SilerNN(8; show_intermediate=true)
	foo_ps, foo_st = Lux.setup(Xoshiro(33), foo)
	foo_ps = foo_ps |> ComponentArray

	X_foo = rand(Xoshiro(1), Uniform(-2, 2), 2, 30)

	Lux.apply(foo, X_foo, foo_ps, foo_st)[1][2, :]

	#year_nn = foo.layers.model
	#year_params, _ = year_nn(X_foo, foo_ps.model, foo_st.model)
end

# ╔═╡ 608363b0-5f75-45c8-8970-5170489a5eeb
function FNN(in_dims, hidden_dims, out_dims; depth=3, act=tanh, outer_act=identity)
	@assert depth > 0
	if depth == 1
		return Chain(
			Dense(in_dims => out_dims, outer_act)
		)
	elseif depth == 2
		return Chain(
			Dense(in_dims => hidden_dims, act),
			Dense(hidden_dims => out_dims, outer_act)
		)
	elseif depth > 2
		return Chain(
			Dense(in_dims => hidden_dims, act),
			[Dense(hidden_dims => hidden_dims, act) for _ in 1:(depth-2)],
			Dense(hidden_dims => out_dims, outer_act)
		)
	end
end

# ╔═╡ 59eed207-e138-44be-a241-d4cbfde0f38c
function get_data(country; T=10, τ₀=3, start_year=start_year, end_year=end_year, _format="LSTM", min_max_scale=true, start_age=start_age, end_age=end_age, p_=0.75, gender=:Female)
	# Apply 'Toy example' data pre-processing
	
    # Load from CSV
    data_ = CSV.read("data/$(country)_Mx_1x1.csv", DataFrame)

	# Split out females, aged 0-98, years start_year-end_year
	start_year = max(start_year, minimum(data_.Year))
	end_year = min(end_year, maximum(data_.Year)) |> f32
	
	@assert start_year < end_year
	@assert end_year - start_year ≥ T

	#TEST.Female .= log.(TEST.Female)

	TEST = DataFrame()

	TEST.Year = zeros(length(start_age:end_age)*length(start_year:extended_forecast_year))
	TEST.Age = zeros(length(start_age:end_age)*length(start_year:extended_forecast_year))

	k = 1
	for y ∈ start_year:extended_forecast_year
		for a ∈ start_age:end_age
			TEST.Year[k] = y
			TEST.Age[k] = a
			k += 1
		end
	end

	leftjoin!(TEST, data_[(start_age .≤ data_.Age .≤ end_age) .&& (start_year .≤ data_.Year .≤ extended_forecast_year), [:Year, :Age, gender]], on = [:Year, :Age])

	TEST = coalesce.(TEST, 0.0)

	for i ∈ 1:size(TEST)[1]
		if TEST[i, gender] == 0
			TEST[i, gender] = 0
		else
			TEST[i, gender] = log(TEST[i, gender])
		end
	end

	sort!(TEST, )

	data = data_[(start_age .≤ data_.Age .≤ end_age) .&& (start_year .≤ data_.Year .≤ end_year), [:Year, :Age, gender]]

	d_sample = rand(Xoshiro(321), Bernoulli(p_), size(data)[1])

	data = data[d_sample, :]

	sort!(data, :Age)

	TEST.Observed .= 0

	for i ∈ 1:size(TEST)[1]
		for j ∈ 1:size(data)[1]
			if TEST.Year[i] == data.Year[j] && TEST.Age[i] == data.Age[j]
				TEST.Observed[i] = 1
			end
		end
	end

	if _format == "LSTM"
	
		# Group age x year and drop the age column (1)
		data = HMD.transform(data, gender)[:, 2:end]
		data = coalesce.(data, 0.0)
	
		# Get average mortality per age
		avg_mortality = mean(Matrix(data), dims=2)
	
		# Change missing data to the mean
		for i ∈ 1:size(data)[1]
			for j ∈ 1:size(data)[2]
				if data[i, j] == 0.0
					if avg_mortality[i, 1] == 0
						data[i, j] = avg_mortality[i+1, 1]
					else
						data[i, j] = avg_mortality[i, 1]
					end
				end
			end
		end
		
		# Convert to Matrix, transpose, take logs, and add extra column to match R code
		data = hcat(start_year:end_year, log.(Matrix(data)'))
	
		# 1 = train, 2 = validation
		X_ = zeros(2, τ₀, T, size(data)[2]-τ₀) # 3, 10, :
		y_ = zeros(2, size(data)[2]-τ₀)
		for i in 1:2
			for j in 1:(size(data)[2]-τ₀)
				X_[i, :, :, j] = data[i:(T+i-1), ((j+1):(j+τ₀))]'
				y_[i, j] = data[T+i, Int(j+1+(τ₀ - 1)/2)]'
			end
		end

		if min_max_scale == true
			# Min-max scale inputs to -1 - 1
			X_1, X_2 = (minimum(X_), maximum(X_))
		
			X_train = min_max.(X_[1, :, :, :], X_1, X_2) |> f32
			X_valid = min_max.(X_[2, :, :, :], X_1, X_2) |> f32
		else
			# Z-transform
			X_1, X_2 = (mean(X_), std(X_))
		
			X_train = standardise.(X_[1, :, :, :], X_1, X_2) |> f32
			X_valid = standardise.(X_[2, :, :, :], X_1, X_2) |> f32
		end
	
		# Negative log mortality so outputs are positive
		y_train = y_[1, :] |> f32
		y_valid = y_[2, :] |> f32
	
		return X_train, reshape(y_train, 1, :), X_valid, reshape(y_valid, 1, :), X_1, X_2, TEST
		
	elseif _format == "NN"
		# Drop missing from training set and validation sets
		# But retain it for a final X test set
		data = data[data[:, gender] .> 0, :]
		
		train = data[data.Year .< end_year, :]
		valid = data[data.Year .≥ end_year, :]

		train = Matrix(train) |> f32
		valid = Matrix(valid) |> f32

		# Recreate the full forecast set but using all ages to test interpolation
		age_length = end_age - start_age + 1

		X_train = train[:, 1:2]
		X_valid = valid[:, 1:2]
		y_train = log.(train[:, end])
		y_valid = log.(valid[:, end])

		validation_ages = X_valid[:, 2]

		if min_max_scale == true
			year_min, year_max = (start_year, end_year-1)
			age_min, age_max = (start_age, end_age)
	
			X_train[:, 1] .= min_max.(X_train[:, 1], year_min, year_max)
			X_valid[:, 1] .= min_max.(X_valid[:, 1], year_min, year_max)
			TEST.Year_std = min_max.(TEST[:, 1], year_mu, year_sigma)
	
			X_train[:, 2] .= min_max.(X_train[:, 2], age_min, age_max)
			X_valid[:, 2] .= min_max.(X_valid[:, 2], age_min, age_max)
			TEST.Age_std = min_max.(TEST[:, 2], age_mu, age_sigma)
		else
			year_mu, year_sigma = (mean(start_year:(end_year-1)), std(start_year:(end_year-1)))
			age_mu, age_sigma = (mean(start_age:end_age), std(start_age:end_age))
	
			X_train[:, 1] .= standardise.(X_train[:, 1], year_mu, year_sigma)
			X_valid[:, 1] .= standardise.(X_valid[:, 1], year_mu, year_sigma)
			TEST.Year_std = standardise.(TEST[:, 1], year_mu, year_sigma)
	
			X_train[:, 2] .= standardise.(X_train[:, 2], age_mu, age_sigma)
			X_valid[:, 2] .= standardise.(X_valid[:, 2], age_mu, age_sigma)
			TEST.Age_std = standardise.(TEST[:, 2], age_mu, age_sigma)
		end

		return X_train', reshape(y_train, 1, :), X_valid', reshape(y_valid, 1, :), TEST

	elseif _format == "CNN"
		# Group age x year and drop the age column (1)
		data = HMD.transform(data, gender)
		data = coalesce.(data[:, 2:end], 0.0)
	
		# Get average mortality per age
		avg_mortality = mean(Matrix(data), dims=2)
	
		# Change missing data to the mean
		for i ∈ 1:size(data)[1]
			for j ∈ 1:size(data)[2]
				if data[i, j] == 0.0
					if avg_mortality[i, 1] == 0
						data[i, j] = avg_mortality[i+1, 1]
					else
						data[i, j] = avg_mortality[i, 1]
					end
				end
			end
		end
		
		 # Convert to Matrix, and take logs
		data = log.(Matrix(data))

		# Number of batches
		m = size(data)[2] - T
		
		X_ = zeros(size(data)[1], T, 1, m)
		y_ = zeros(size(data)[1], m)

		for i ∈ 1:m
			X_[:, :, :, i] = data[:, i:(T+i-1)]
			y_[:, i] = data[:, T+i]
		end

		X_train = X_[:, :, :, 1:(end-1)]
		X_valid = X_[:, :, :, end]

		if min_max_scale == true
			# Min-max scale inputs to -1 - 1
			X_1, X_2 = (minimum(X_), maximum(X_))
		
			X_train = min_max.(X_train, X_1, X_2) |> f32
			X_valid = min_max.(X_valid, X_1, X_2) |> f32
		else
			# Z-transform
			X_1, X_2 = (mean(X_), std(X_))
		
			X_train = standardise.(X_train, X_1, X_2) |> f32
			X_valid = standardise.(X_valid, X_1, X_2) |> f32
		end
	
		# Log mortality so outputs are negative
		y_train = y_[:, 1:(end-1)] |> f32
		y_valid = y_[:, end] |> f32

		age_length = length(start_age:end_age)
	
		return reshape(X_train, age_length, T, 1, m-1), y_train, reshape(X_valid, age_length, T, 1, 1), y_valid, X_1, X_2, TEST
	end
		
end

# ╔═╡ c4f2ff5e-8bea-41e8-8414-8ebe89bff363
function get_data_NEW(
    countries::AbstractVector{String}; # Accepts a list of countries
    start_year=start_year,
    end_year=end_year,
    min_max_scale=false,
    start_age=start_age,
    end_age=end_age,
    p_=0.05,
	filtered=-1,
	genders=[:Female, :Male]
)

    # 1. Initialize global containers
    pop_id_map = Dict{String, Int}()
    all_data = DataFrame() # For training/validation data
    all_test = DataFrame() # For the final, complete grid

    # 2. Loop over all countries and genders to build datasets
    for country in countries
        # Load from CSV (once per country)
        local data_
        try
            data_ = CSV.read("data/$(country)_Mx_1x1.csv", DataFrame)
        catch e
            @warn "Skipping $(country): Could not read file. Error: $e"
            continue
        end
        
        country_start_year = max(start_year, minimum(data_.Year))
        country_end_year = min(end_year, maximum(data_.Year)) |> f64

		#=
        if country_start_year >= country_end_year || (country_end_year - country_start_year < T)
            @warn "Skipping $(country): Not enough valid data in range $(start_year)-$(end_year)."
            continue
        end
		=#

        for gender ∈ genders

            # --- A. Create the full TEST grid ---
            test_pop = DataFrame(
                Year = vec([Float64(y) for y in start_year:extended_forecast_year, a in start_age:end_age]),
                Age = vec([Float64(a) for y in start_year:extended_forecast_year, a in start_age:end_age])
            )
            leftjoin!(test_pop, data_[(start_age .≤ data_.Age .≤ end_age), [:Year, :Age, gender]], on=[:Year, :Age])
			test_pop.Birth_year = test_pop.Year .- test_pop.Age
			test_pop = coalesce.(test_pop, 0.0)
            test_pop.log_mx = [val == 0.0 || ismissing(val) ? 0.0 : log(val) for val in test_pop[:, gender]]
            select!(test_pop, :Year, :Age, :Birth_year, :log_mx) 

            # --- B. Create the training sample ---
            data_pop = data_[(start_age .≤ data_.Age .≤ end_age) .&& (country_start_year .≤ data_.Year .≤ country_end_year), [:Year, :Age, gender]]
			data_pop.Birth_year = data_pop.Year .- data_pop.Age
            
            d_sample = rand(Xoshiro(321), Bernoulli(p_), size(data_pop)[1])
            data_pop = data_pop[d_sample, :]
            
            train_set = Set(zip(data_pop.Year, data_pop.Age, data_pop.Birth_year))
            
            data_pop = data_pop[data_pop[:, gender] .> 0, :]
            data_pop.log_mx = log.(data_pop[:, gender])
            select!(data_pop, :Year, :Age, :Birth_year, :log_mx) 

            # --- C. Assign Population ID ---
            key = "$(country)_$(gender)"
            pop_id = get!(pop_id_map, key, length(pop_id_map) + 1)

			data_pop.PopID = fill(pop_id, nrow(data_pop))
			test_pop.PopID = fill(pop_id, nrow(test_pop))

			test_pop.Observed .= 0
		
			for i ∈ 1:size(test_pop)[1]
				for j ∈ 1:size(data_pop)[1]
					if test_pop.Year[i] == data_pop.Year[j] && test_pop.Age[i] == data_pop.Age[j]
						test_pop.Observed[i] = 1
					end
				end
			end

            # --- D. Vcat to global containers ---
            all_data = vcat(all_data, data_pop)
            all_test = vcat(all_test, test_pop)

        end
    end
    
    # 3. Now, process the *combined* dataset
	if filtered ≠ -1
		filter!(x -> x.PopID ∈ filtered, all_data)
		filter!(x -> x.PopID ∈ filtered, all_test)
	end
    
    # --- E. Split train/valid (full dataset) ---
    train_df = all_data[all_data.Year .≤ end_year, :]
    valid_df = all_data[all_data.Year .== end_year+1, :]


    
    # --- F. Calculate scaling parameters based on `scaling_pop` ---
    scaling_df = deepcopy(train_df)

    scaling_features = Matrix(scaling_df[:, [:Year, :Age, :Birth_year]]) |> f64
    
    year_mu, year_sigma = (mean(scaling_features[:, 1]), std(scaling_features[:, 1]))
    age_mu, age_sigma = (mean(scaling_features[:, 2]), std(scaling_features[:, 2]))
    birth_year_mu, birth_year_sigma = (mean(scaling_features[:, 3]), std(scaling_features[:, 3]))
    
    year_min, year_max = (minimum(scaling_features[:, 1]), maximum(scaling_features[:, 1]))
    age_min, age_max = (minimum(scaling_features[:, 2]), maximum(scaling_features[:, 2]))
	birth_year_min, birth_year_max = (minimum(scaling_features[:, 3]), maximum(scaling_features[:, 3]))

    
    # Target (y) is always :log_mx
    y_train = Float64.(Matrix(train_df[:, [:log_mx]])') # Shape (1, N_train)
    y_valid = Float64.(Matrix(valid_df[:, [:log_mx]])') # Shape (1, N_valid)
    
    # Features (X) depend on pop_mode
    local X_train, X_valid # Declare
    
    feature_cols = [:Year, :Age, :Birth_year, :PopID]
    
    X_train = Float64.(Matrix(train_df[:, feature_cols])) # Shape (N_train, 2 or 3)
    X_valid = Float64.(Matrix(valid_df[:, feature_cols])) # Shape (N_valid, 2 or 3)

    # --- H. Apply Scaling (to X_train, X_valid, all_test) ---
    if min_max_scale
        X_train[:, 1] .= min_max.(X_train[:, 1], year_min, year_max)
        X_valid[:, 1] .= min_max.(X_valid[:, 1], year_min, year_max)
        X_train[:, 2] .= min_max.(X_train[:, 2], age_min, age_max)
        X_valid[:, 2] .= min_max.(X_valid[:, 2], age_min, age_max)
        X_train[:, 3] .= min_max.(X_train[:, 3], birth_year_min, birth_year_max)
        X_valid[:, 3] .= min_max.(X_valid[:, 3], birth_year_min, birth_year_max)
        
        all_test.Year_std = min_max.(all_test.Year, year_min, year_max)
        all_test.Age_std = min_max.(all_test.Age, age_min, age_max)
		all_test.Birth_year__std = min_max.(all_test.Birth_year, birth_year_min, birth_year_max)
    else
        X_train[:, 1] .= standardise.(X_train[:, 1], year_mu, year_sigma)
        X_valid[:, 1] .= standardise.(X_valid[:, 1], year_mu, year_sigma)
        X_train[:, 2] .= standardise.(X_train[:, 2], age_mu, age_sigma)
        X_valid[:, 2] .= standardise.(X_valid[:, 2], age_mu, age_sigma)
		X_train[:, 3] .= standardise.(X_train[:, 3], birth_year_mu, birth_year_sigma)
        X_valid[:, 3] .= standardise.(X_valid[:, 3], birth_year_mu, birth_year_sigma)

        all_test.Year_std = standardise.(all_test.Year, year_mu, year_sigma)
        all_test.Age_std = standardise.(all_test.Age, age_mu, age_sigma)
        all_test.Birth_year_std = standardise.(all_test.Birth_year, birth_year_mu, birth_year_sigma)
    end
    
    # 4. Return combined matrices, the full TEST grid, and the ID map
    # We transpose X to match the (features, n_samples) shape
    return X_train', y_train, X_valid', y_valid, all_test, pop_id_map
end

# ╔═╡ 055dd1ae-39f1-4292-8a6e-00cfd2795688
begin
	X_train, y_train, X_valid, y_valid, TEST, country_embeddings = get_data_NEW(list_of_countries[[20, 22, 23, 32, 35, 38]]; filtered=-1, p_=p_)
	#X_train, y_train, X_valid, y_valid, TEST, country_embeddings = get_data_NEW([country])
	country_embed_dims = length(country_embeddings)
end

# ╔═╡ 7513dbee-3d55-49d8-bcf4-3003cd13dbb5
function LeeCarterNN(hidden_dims; show_intermediate=false, act=tanh, N_countries=country_embed_dims, embed_dims=5)

	function LeeCarter(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, model₃::Lux.AbstractLuxLayer, show_intermediate::Bool)
		# log(mu) = alpha(x) + beta(x)*kappa(t)
		return @compact(; model₁, model₂, model₃, show_intermediate) do x
			year = reshape(x[1, :], 1, :)
			age = reshape(x[2, :], 1, :)
			
			age_params = model₁(age)
			year_params = model₂(year)
			α = age_params[1, :]
			β = exp.(age_params[2, :])
			κ = year_params[1, :]

			age_year = vcat(year, age)
			log_σ² = model₃(age_year) |> vec
			
			if show_intermediate
				@return (age_params, year_params)
			else
				@return hcat(α .+ β .* κ, log_σ²)
			end
		end
	end

	age_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 2)
	)
	
	year_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return LeeCarter(age_NN, year_NN, log_sigma_NN, show_intermediate)
end

# ╔═╡ 0841707c-3942-412c-b729-1778a81929d3
Float64.(reshape(X_train[3, :], 1, :) .== unique(TEST.PopID)[2:end])

# ╔═╡ 7e061810-6edc-4b45-bb1d-8c6769d37a2d
function LeeCarterNN_gender(hidden_dims; show_intermediate=false, act=tanh, N_countries=country_embed_dims, embed_dims=4, indicator_region=unique(TEST.PopID))

	
	function LeeCarter(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, model₃::Lux.AbstractLuxLayer, region_model::Lux.AbstractLuxLayer, show_intermediate::Bool)
		# log(mu) = alpha(x) + beta(x)*kappa(t)
		return @compact(; model₁, model₂, model₃, region_model, show_intermediate) do x
			year = reshape(x[1, :], 1, :)
			age = reshape(x[2, :], 1, :)
			region = Int.(x[4, :])
			region_params = region_model(region)
			
			age_params = model₁(age)
			year_params = model₂(year)
			
			α = age_params[1, :] .* region_params[1, :]
			β = exp.(age_params[2, :] .* region_params[2, :])
			κ = year_params[1, :] .* region_params[3, :]

			age_year = vcat(year, age)
			log_σ² = (model₃(age_year)[1, :] .* region_params[4, :]) |> vec
			
			if show_intermediate
				@return (age_params, year_params, region_params)
			else
				@return hcat(α .+ β .* κ, log_σ²)
			end
		end
	end

	region_NN = Embedding(
		N_countries => embed_dims
	)

	age_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 2)
	)
	
	year_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return LeeCarter(age_NN, year_NN, log_sigma_NN, region_NN, show_intermediate)
end

# ╔═╡ 9fb77dbb-2244-4bd8-a29c-5ef84f117ce5
function LeeCarterNN_embed(hidden_dims; show_intermediate=false, act=tanh, N_countries=country_embed_dims, embed_dims=5)

	function LeeCarter(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, model₃::Lux.AbstractLuxLayer, country_embedding::Lux.AbstractLuxLayer, show_intermediate::Bool)
		# log(mu) = alpha(x) + beta(x)*kappa(t)
		return @compact(; model₁, model₂, model₃, country_embedding, show_intermediate) do x
			year = reshape(x[1, :], 1, :)
			age = reshape(x[2, :], 1, :)
			
			pop = Int.(reshape(x[3, :], 1, :))
			pop_embed = reshape(country_embedding(pop), embed_dims, :)
			
			age_params = model₁(vcat(age, pop_embed))
			year_params = model₂(vcat(year, pop_embed))
			α = age_params[1, :]
			β = exp.(age_params[2, :])
			κ = year_params[1, :]

			age_year = vcat(year, age, pop_embed)
			log_σ² = model₃(age_year) |> vec
			
			if show_intermediate
				@return (age_params, year_params, pop_embed)
			else
				@return hcat(α .+ β .* κ, log_σ²)
			end
		end
	end

	country_embedding = Lux.Embedding(N_countries => embed_dims)
	
	age_NN =  Chain(
				Dense(1+embed_dims => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 2)
	)
	
	year_NN =  Chain(
				Dense(1+embed_dims => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	log_sigma_NN =  Chain(
				Dense(2+embed_dims => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return LeeCarter(age_NN, year_NN, log_sigma_NN, country_embedding, show_intermediate)
end

# ╔═╡ 52139393-d898-45d5-bc86-54fba2e41bba
function RenshawHabermanNN_gender(hidden_dims; show_intermediate=false, act=tanh, N_countries=country_embed_dims, embed_dims=5)
	
	function RenshawHaberman(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, model₃::Lux.AbstractLuxLayer, model₄::Lux.AbstractLuxLayer, region_model::Lux.AbstractLuxLayer, show_intermediate::Bool)
		# log(mu) = alpha(x) + beta₁(x)*kappa(t) + beta₂(x)*gamma(t-x)
		return @compact(; model₁, model₂, model₃, model₄, region_model, show_intermediate) do x
			year = reshape(x[1, :], 1, :)
			age = reshape(x[2, :], 1, :)
			birth_year = reshape(x[3, :], 1, :)
			region = Int.(x[4, :])
			region_params = region_model(region)

			age_params = model₁(age)
			year_params = model₂(year)
			birth_params = model₃(birth_year)
			
			α = age_params[1, :] .* region_params[1, :]
			β₁ = exp.(age_params[2, :] .* region_params[2, :])
			κ = year_params[1, :] .* region_params[3, :]
			β₂ = exp.(age_params[3, :] .* region_params[2, :])
			γ = birth_params[1, :] .* region_params[4, :]

			age_year = vcat(year, age, birth_year)
			log_σ² = (model₄(age_year)[1, :] .* region_params[5, :]) |> vec
			
			if show_intermediate
				@return (age_params, year_params, region_params)
			else
				@return hcat(α .+ (β₁ .* κ) .+ (β₂ .* γ), log_σ²)
			end
		end
	end

	region_NN = Embedding(
		N_countries => embed_dims
	)

	age_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 3)
	)
	
	year_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	birth_year_NN =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	log_sigma_NN =  Chain(
				Dense(3 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return RenshawHaberman(age_NN, year_NN, birth_year_NN, log_sigma_NN, region_NN, show_intermediate)
end

# ╔═╡ 7565da20-a7b5-4a12-b75c-141e1374af72
hcat(reshape(X_train[1, :], :, 1), reshape(X_train[2, :], :, 1))'

# ╔═╡ ec25dfee-6929-4c94-a023-1ca59fe8ad0b
X_train

# ╔═╡ e4917c43-df80-40ea-9908-29bf46334c46
function HeligmanPollardNN_gender(hidden_dims; show_intermediate=false, act=tanh, N_countries=country_embed_dims, embed_dims=9)
	
	function HeligmanPollard(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, model₃::Lux.AbstractLuxLayer, model₄::Lux.AbstractLuxLayer, model_region::Lux.AbstractLuxLayer, show_intermediate::Bool)
		return @compact(; model₁, model₂, model₃, model₄, model_region, show_intermediate) do x
			# HP: qx/(1-qx) = A^(x+B)^c + D*exp(-E(log(x/F))^2) + GH^x
			year = reshape(x[1, :], 1, :)
			age = max.(((x[2, :]) .* s_age) .+ m_age, 1.0f-9)
			region = Int.(x[4, :])
			region_params = model_region(region)
			
			infant_year_params = model₁(year) # ABC
			A = exp.((infant_year_params[1, :] .* region_params[1, :]) .- 11.0f0)
			B = softplus.(infant_year_params[2, :] .* region_params[2, :])
			C = sigmoid.(infant_year_params[3, :] .* region_params[3, :]) .* 0.2f0

			hump_year_params = model₂(year) # DEF			
			D = exp.((hump_year_params[1, :] .* region_params[4, :]) .- 11.0f0)
			E = softplus.(hump_year_params[2, :] .* region_params[5, :])
			F = 18.0f0 .+ softplus.(hump_year_params[3, :] .* region_params[6, :])

			secant_year_params = model₃(year) # GH
			G = exp.((secant_year_params[1, :] .* region_params[7, :]) .- 11.0f0)
			H = 1.0f0 .+ softplus.(secant_year_params[2, :] .* region_params[8, :]) .* 0.15f0

			HP = A .^ ((age .+ B) .^ C) .+ D .* exp.(-E .* (log.(age ./ F)) .^ 2) .+ (G .* (H .^ age))
			
			HP_qx = HP ./ (1.0f0 .+ HP .+ 1.0f-9)

			HP_mx = -log.(1.0f0 .- HP_qx .+ 1.0f-9)

			# log σ²
			age_year = vcat(year, reshape(x[2, :], 1, :))
			log_σ² = (model₄(age_year)[1, :] .* region_params[9, :]) |> vec
			
			if show_intermediate
				@return year_params
			else
				@return hcat(log.(HP_mx), log_σ²)
			end
		end
	end

	region_NN = Embedding(
		N_countries => embed_dims
	)
	
	NN_infant =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 3)
	)
	
	NN_hump =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 3)
	)

	NN_secant =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 2)
	)
	
	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return HeligmanPollard(NN_infant, NN_hump, NN_secant, log_sigma_NN, region_NN, show_intermediate)
end

# ╔═╡ eb111789-8e19-46b7-b0d3-0d569bba2d01
function KostakiNN_gender(hidden_dims; show_intermediate=false, act=tanh, N_countries=country_embed_dims, embed_dims=10)
	
	function Kostaki(model₁::Lux.AbstractLuxLayer, model₂::Lux.AbstractLuxLayer, model₃::Lux.AbstractLuxLayer, model₄::Lux.AbstractLuxLayer, model_region::Lux.AbstractLuxLayer, show_intermediate::Bool)
		return @compact(; model₁, model₂, model₃, model₄, model_region, show_intermediate) do x
			# K: qx/(1-qx) = A^(x+B)^C + D*exp(-E(log(x/F))^2 + Kx) + GH^x
			year = reshape(x[1, :], 1, :)
			age = max.(((x[2, :]) .* s_age) .+ m_age, 1.0f-9)
			region = Int.(x[3, :])
			region_params = model_region(region)
			
			infant_year_params = model₁(year) # ABC
			A = exp.((infant_year_params[1, :] .* region_params[1, :]) .- 11.0f0)
			B = softplus.(infant_year_params[2, :] .* region_params[2, :])
			C = sigmoid.(infant_year_params[3, :] .* region_params[3, :]) .* 0.2f0

			hump_year_params = model₂(year) # DEFK
			D = exp.((hump_year_params[1, :] .* region_params[4, :]) .- 11.0f0)
			E = softplus.(hump_year_params[2, :] .* region_params[5, :])
			F = 18.0f0 .+ softplus.(hump_year_params[3, :] .* region_params[6, :])
			K = hump_year_params[4, :] .* region_params[7, :]

			secant_year_params = model₃(year) # GH
			G = exp.((secant_year_params[1, :] .* region_params[8, :]) .- 11.0f0)
			H = 1.0f0 .+ softplus.(secant_year_params[2, :] .* region_params[9, :]) .* 0.15f0

			HP = A .^ ((age .+ B) .^ C) .+ D .* exp.(-E .* ((log.(age ./ F)) .^ 2) .+ (K .* age)) .+ (G .* (H .^ age))
			
			HP_qx = HP ./ (1.0f0 .+ HP .+ 1.0f-9)

			HP_mx = -log.(1.0f0 .- HP_qx .+ 1.0f-9)

			# log σ²
			age_year = vcat(year, reshape(x[2, :], 1, :))
			log_σ² = (model₄(age_year)[1, :] .* region_params[10, :]) |> vec
			
			if show_intermediate
				@return hcat(A, B, C, D, E, F, G, H, K)
			else
				@return hcat(log.(HP_mx), log_σ²)
			end
		end
	end

	region_NN = Embedding(
		N_countries => embed_dims
	)
	
	NN_infant =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 3)
	)
	
	NN_hump =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 4)
	)

	NN_secant =  Chain(
				Dense(1 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 2)
	)
	
	log_sigma_NN =  Chain(
				Dense(2 => hidden_dims, act), 
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => hidden_dims, act),
				Dense(hidden_dims => 1)
	)

	return Kostaki(NN_infant, NN_hump, NN_secant, log_sigma_NN, region_NN, show_intermediate)
end

# ╔═╡ 80e4878c-2fc6-483a-90b9-5bb3f362f749


# ╔═╡ a2be51e0-f646-47db-bb3b-daa672fd4f44
country_embeddings

# ╔═╡ ad84dffe-d03b-4e5d-8856-347f4e0e4332
list_of_countries[[20, 22, 32, 35, 38]]

# ╔═╡ 40711e69-c166-4a2a-8d37-88f0c241fcce
list_of_countries

# ╔═╡ 7b5558e7-30c8-4069-9175-6dd79d27c8f5
#=
if model_type == "NN"
	X_train, y_train, X_valid, y_valid, TEST = get_data(country; T=T, τ₀=τ₀, _format=model_type,  min_max_scale=false, start_year=start_year, p_=p_, gender=gender_)
else
	X_train, y_train, X_valid, y_valid, x_1, x_2, y_test = get_data(country; T=T, τ₀=τ₀, _format=model_type,  min_max_scale=false, start_year=start_year, p_=p_, gender=gender_)
end
=#

# ╔═╡ d12da7dd-e186-4329-939e-92cd8702f2e6
X_train

# ╔═╡ 8e5c8f3f-cfee-4dbc-baf5-f392b26e0e54
TEST

# ╔═╡ 03acd606-0d03-457c-933c-2f1ff0caded6
begin
# 1. SETUP: Mappings

# Invert to get ID -> Name
id_to_name = Dict(value => key for (key, value) in country_embeddings)

# 2. FILTER DATA
# Filter for Odd IDs (Females)
df_odd = filter(row -> isodd(row.PopID), TEST)
df_odd = filter(row -> row.Year .≤ end_year, df_odd)
unique_ids = sort(unique(df_odd.PopID))

# Define global color limits so all plots share the same scale
z_min, z_max = extrema(df_odd.log_mx)

# 3. PLOTTING LOOP
plot_list = []

for pid in unique_ids
    # Get data for this country
    sub_df = filter(row -> row.PopID == pid, df_odd)
    
    # SORTING IS CRITICAL for reshaping
    # We sort by Age then Year to ensure the matrix fills correctly
    sort!(sub_df, [:Year, :Age])
    
    # Extract axes
    x_ages = sort(unique(sub_df.Age))
    y_years = sort(unique(sub_df.Year))
    
    # Reshape z-vector into a Matrix (Years x Ages) for Plots.jl
    # Dimensions must be [length(y_years), length(x_ages)]
    z_matrix = reshape(sub_df.log_mx, length(x_ages), length(y_years))'
    
    # Create the single plot
    p = heatmap(
        x_ages, 
        y_years, 
        z_matrix, 
        title = id_to_name[pid],
        xlabel = "Age",
        ylabel = "Year",
        c = :magma,       # Colormap
        clims = (z_min, z_max), # Shared color scale
        colorbar = true   # Keep colorbar on individual plots for readability
    )
    
    push!(plot_list, p)
end

# 4. COMBINE PLOTS
# We use a 3x2 layout. Since there are 5 plots, the 6th slot will be empty.
final_plot = plot(plot_list..., layout = (3, 2), size = (1000, 1200), plot_title="Log Mortality Rates (Females)")

# Display
final_plot
end

# ╔═╡ 1b83a76d-2587-4e3c-a000-901a5a165716
add_var_to_y(yt) = vcat(yt, repeat([log.(var(yt))], size(yt, 2))')

# ╔═╡ 564d77df-85c1-406a-8964-3b14fca6328c
function main(tstate, vjp, data, epochs; early_stopping=true)
	y_train_t = add_var_to_y(y_train)
	loss_function = MSELoss()
	train_losses = []
	#valid_losses = []
	
    for epoch in 1:epochs

        _, l, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)
		# Use testmode to turn off Dropout
		train_pred = (Lux.apply(tstate.model, X_train, tstate.parameters, Lux.testmode(tstate.states)))[1]
		train_loss = MSELoss()(y_train_t', train_pred)
		push!(train_losses, train_loss)
		
		#valid_pred = (Lux.apply(tstate.model, X_valid, tstate.parameters, Lux.testmode(tstate.states)))[1]
		#valid_loss = MSELoss()(reshape(y_valid, :, 1), valid_pred)
		#push!(valid_losses, valid_loss)

        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Train Loss: %.5g \n" epoch train_loss
		end

		# Custom early-stopping
		if early_stopping == true
			if epoch > 150
				if valid_losses[end] > valid_losses[end-50]
					@printf "Early Stopping at Epoch: %3d \t" epoch
					return tstate, train_losses, valid_losses
				end
	        end
		end
    end
	
    return tstate, train_losses#, valid_losses
end

# ╔═╡ b7d6736b-776e-49d1-ae18-4a64b07a1a24
begin
	if model_type == "NN"
		#discrete_model = FNN(τ₀, τ₁, 1; depth=NN_depth, act=act, outer_act=identity)
		#discrete_model = GompertzNN(τ₁)
		#discrete_model = LeeCarterNN(τ₁; act=act)
		discrete_model = LeeCarterNN_gender(τ₁; act=act)
		#discrete_model = RenshawHabermanNN_gender(τ₁; act=act)
		#discrete_model = SilerNN(τ₁)
		#discrete_model = HeligmanPollardNN(τ₁)
		#discrete_model = HeligmanPollardNN_gender(τ₁)
		#discrete_model = KostakiNN_gender(τ₁)
		#discrete_model = CairnsBlakeDowdNN(τ₁)
		#discrete_model = GMNN(τ₁)
		#discrete_model = CairnsBlakeDowd2NN(τ₁)
		#discrete_model = classicNN(τ₁; act=act)
		#discrete_model = LocalGLMNetNN(τ₁)
	elseif model_type == "LSTM"
		discrete_model = LSTM(τ₀, τ₁, 1; depth=NN_depth, cell=cell)
		#discrete_model = NNODE(τ₀, τ₁, 1; sensealg=TrackerAdjoint())
	elseif model_type == "CNN"
		discrete_model = CNN(age_length, τ₁)
	end
	
	ps, st = Lux.setup(Xoshiro(12345), discrete_model)
	ps = ps |> ComponentArray

	tstate = Lux.Training.TrainState(discrete_model, ps, st, opt)
	
	ad_rule = AutoZygote()
	#ad_rule = AutoEnzyme()

	

	if model_type == "NN"
		try
			@time tstate, train_losses = main(tstate, ad_rule, (X_train, add_var_to_y(y_train)), n_epochs; early_stopping=false)
		catch
			@time tstate, train_losses = main(tstate, ad_rule, (X_train, add_var_to_y(y_train)'), n_epochs; early_stopping=false)
		end
	else
		@time tstate, train_losses = main(tstate, ad_rule, (X_train, y_train), n_epochs; early_stopping=false)
	end
end

# ╔═╡ 864a7e7f-d4e3-4ada-b81a-61d5574c7138
add_var_to_y(y_train)

# ╔═╡ 1fe630ea-6926-4260-b549-12d11856a74a
reshape(y_train, :, 1)

# ╔═╡ b34c44fd-fed8-4111-984e-39e98bfe1481
reshape(add_var_to_y(y_train), 2, :)'

# ╔═╡ 7096f53d-8f11-415d-b419-4c28917d6dc2
X_train

# ╔═╡ 096a8a27-ddad-4534-94e4-71f755d5f561
discrete_model

# ╔═╡ bb6602f4-927e-473e-b8da-957395ed7617
begin
	if model_type == "NN"
		plot(title="Losses: τ₁=$τ₁, act=$act, FNN, depth=$NN_depth", xlab="Epochs", ylab="MSE", ylim=(0.0, max(0.21, 1.5*minimum(train_losses))))
	elseif model_type == "LSTM"
		plot(title="Losses: $(country)\n τ₀=$τ₀, τ₁=$τ₁, T=$T, cell=$cell, depth=$NN_depth", xlab="Epochs", ylab="MSE", ylim=(0.0, max(0.21, 1.5*minimum(train_losses))))
	elseif model_type == "CNN"
		plot(title="Losses: $(country)\n T=$T, CNN", xlab="Epochs", ylab="MSE", ylim=(0.0, max(0.21, 1.5*minimum(train_losses))))
	end
	scatter!(1:length(train_losses), train_losses, label="Training")
	#scatter!(1:length(valid_losses), valid_losses, label="Validation")
	hline!([0.1], col=:black, label=false)
end

# ╔═╡ e006827b-2f8b-4b7a-9669-2fb64fadc129
vector_to_parameters(ps_new, ps_) = ComponentArray(ps_new, getaxes(ps_))

# ╔═╡ bbb80aa6-096c-4fae-bc1d-2e2f5ba28b2d
function predict(train_state, Xs)
	st_ = Lux.testmode(train_state.states)
	return (Lux.apply(train_state.model, Xs, train_state.parameters, st_))[1]
end

# ╔═╡ 4fb91714-4412-49c3-b466-8ac3ee439047
function predict(train_state, Xs, params)
	st_ = Lux.testmode(train_state.states)
	return (Lux.apply(train_state.model, Xs, vector_to_parameters(params, train_state.parameters), st_))[1]
end

# ╔═╡ b69e3b36-d0e6-46f3-962f-d2f840064260
tstate.model

# ╔═╡ 3c1d261e-3043-4b0b-965f-5c48e5466316
#heatmap(X_valid[:, :, 1, 1]', xlab="Age (x)", ylab="Year (t)", title="Scaled Mortality Data")

# ╔═╡ d0f95b2e-2326-44fe-8078-5c89d13cb8d5
#heatmap(tstate.model[1](X_train, tstate.parameters.layer_1, Lux.testmode(tstate.states))[1][:, :, 3, 5], xlab="Year (t)", ylab="Age (x)", title="Convolutional Layer #1:\nChannel 1; Time step 1965-1974", titlefontsize=10)

# ╔═╡ cac4f05e-e40a-4814-bde1-0c72c047e2ff
#heatmap(tstate.model[2](X_valid, tstate.parameters.layer_1, Lux.testmode(tstate.states))[1][:, :, 1, 1]', xlab="Age (x)", ylab="Year (t)", title="Pooling Layer:\nMeanPool")

# ╔═╡ c49191f4-f092-44c9-9497-e0772b43b158
#heatmap(tstate.model[3](X_valid, tstate.parameters.layer_1, Lux.testmode(tstate.states))[1][:, :, 1, 1]', xlab="Age (x)", ylab="Year (t)", title="Convolution Layer #2:\nChannel 1")

# ╔═╡ 2fa273ec-86e1-4e2c-a589-e79a09e0089d
function predict(train_state::Training.TrainState, Xs, time_step, x_1, x_2, standariser)
	X_ = deepcopy(Xs)
	pred_full = zeros(time_step, 101-τ₀+1)
	for i ∈ 1:time_step
		pred = predict(train_state, X_)
		pred_full[i, :] = pred
		pred_scale = standariser.(pred, x_1, x_2)
		# Shift downwards
		for j ∈ 1:(T-1)
			X_[:, j, :] = X_[:, j+1, :]
		end
		for k ∈ 1:τ₀
			X_[k, end, :] = pred_scale
		end
	end
	return pred_full
end

# ╔═╡ 8b935b73-b129-4b65-808a-bccace7d0838
TEST

# ╔═╡ 2598323d-8e4b-4789-8f77-97cc99b94b29
if model_type == "NN"
	#BNN_arch = FNN(τ₀, τ₁, 1; depth=NN_depth, act=act, outer_act=identity)
	#BNN_arch = GompertzNN(τ₁; act=act)
	#BNN_arch = LeeCarterNN(τ₁; act=act)
	BNN_arch = LeeCarterNN_gender(τ₁; act=act)
	#BNN_arch = RenshawHabermanNN_gender(τ₁; act=act)
	#BNN_arch = SilerNN(τ₁; act=act)
	#BNN_arch = HeligmanPollardNN(τ₁; act=act)
	#BNN_arch = HeligmanPollardNN_gender(τ₁; act=act)
	#BNN_arch = KostakiNN_gender(τ₁; act=act)
	#BNN_arch = CairnsBlakeDowdNN(τ₁; act=act)
	#BNN_arch = GMNN(τ₁; act=act)
	#BNN_arch = CairnsBlakeDowd2NN(τ₁; act=act)
	#BNN_arch = classicNN(τ₁; act=act)
	#BNN_arch = LocalGLMNetNN(τ₁; act=act)
elseif model_type == "LSTM"
	BNN_arch = LSTM(τ₀, τ₁, 1; depth=NN_depth, cell=cell)
	#BNN_arch = NNODE(τ₀, τ₁, 1; sensealg=TrackerAdjoint())
elseif model_type == "CNN"
	BNN_arch = CNN(age_length, τ₁; act=act)
end

# ╔═╡ 0711bfc1-4337-4ae4-a5b0-c7b08dae2190
begin
	N_samples = 200#5_000#1_500#1_000
	half_N_samples = max(10, Int(N_samples/2))
	N_chains = 4#2#4#1
end

# ╔═╡ 5cd9678e-6dd7-4290-ad48-3377735f7696
function path_finder(m, iters, type_; n_ch=N_chains, ad=AutoZygote())
	if type_ == "pathfinder"
		result_multi = multipathfinder(m, iters; nruns=n_ch, rng=Xoshiro(789))
		return collect.(eachrow(result_multi.draws_transformed.value[1:n_ch, :, 1]))
	elseif type_ == "VI"
		qo = Turing.Variational.meanfield(m)
		advi = ADVI(10, iters; adtype=ad)
		opt = Turing.Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
		q_fr = vi(m, advi, qo; optimizer=opt)

		draws_matrix = rand(Xoshiro(23), q_fr, n_ch)

		return [draws_matrix[:, i] for i in 1:n_ch]
	elseif type_ == "MAP"
		return maximum_a_posteriori(m).values.array
	end
end

# ╔═╡ 78baeae1-36cb-47a4-b82a-fba776e19635
function BNN(BNN_arch, N_samples)
	ps_BNN, st_BNN = Lux.setup(Xoshiro(12345), BNN_arch) |> f64
	ps_BNN = ps_BNN |> ComponentArray

	DNN_params = vcat(tstate.parameters...) |> f64

	Nloglikelihood(y_, μ_, σ_) = loglikelihood(MvNormal(μ_, (σ_ .^ 2) .* I), y_)

	@model function bayes_nn(xs, ys, BNN_arch, ps_BNN, sig, n_params, ::Type{T} = Float64) where {T}

		# Priors
		## Sample the parameters
		#σ ~ truncated(Normal(0, 1); lower=1e-4)
		#σ ~ truncated(Normal(0, 1); lower=0.25, upper=0.65)
		#σ ~ InverseGamma(12, 5.5)
		#sig ~ InverseGamma(2.0, 0.9)
		#α ~ Uniform(0.90, 1.0)
		#parameters ~ MvNormal(zeros(n_params), (sig^2) .* I)
		#parameters ~ filldist(Laplace(0.0, (sig^2)), n_params)
		parameters ~ MvNormal(DNN_params, (sig^2) .* I)
		
		## Forward NN to make predictions
		#log_μ_and_sigma, st_BNN = Lux.apply(BNN_arch, xs, vector_to_parameters(parameters .* α, ps_BNN), st_BNN)
		log_μ_and_sigma, st_BNN = Lux.apply(BNN_arch, xs, vector_to_parameters(parameters, ps_BNN), st_BNN)
		μ = log_μ_and_sigma[:, 1]
		σ = sqrt.(exp.(log_μ_and_sigma[:, 2]))
	
		## Likelihood
		#=
		for i ∈ 1:size(ys)[2]
			Turing.@addlogprob! Nloglikelihood(ys[:, i], μ[:, i], σ[:, i])
		end
		=#
		ys ~ MvNormal(μ, I * (σ .^ 2))

		#return nothing
	end

	n_params = Lux.parameterlength(BNN_arch)
	sig = 0.25#0.85#0.025

	if model_type == "NN"
		BNN_inference = bayes_nn(X_train |> f64, reshape(y_train, :, 1) |> f64, BNN_arch, ps_BNN, sig, n_params)
	else
		BNN_inference = bayes_nn(X_train |> f64, y_train |> f64, BNN_arch, ps_BNN, sig, n_params)
	end
	ad = AutoTracker()
	#ad = AutoEnzyme()
	#ad = AutoZygote()
	#ad = AutoReverseDiff(; compile=true)
	#ad = AutoMooncake()

	#sampling_alg = Turing.HMCDA(0.9, 1.0; adtype=ad)
	sampling_alg = Turing.NUTS(0.9; adtype=ad)
	#sampling_alg = Turing.SGLD(; stepsize=PolynomialStepsize(1e-6), adtype=ad)
	#sampling_alg = Turing.SGHMC(; learning_rate=1e-7, momentum_decay=0.6, adtype=ad)

	#map_estimate = maximum_a_posteriori(BNN_inference)

	if sample_type == "MCMC"
		#@time init_p_ = path_finder(BNN_inference, 1_000, "VI")
		
		#chains = sample(Xoshiro(22), BNN_inference, sampling_alg, N_samples; discard_adapt=false, initial_params=map_estimate.values.array)
		chains_prior = 0.0#sample(Xoshiro(22), BNN_inference, Prior(), N_samples; discard_adapt=false)
		#chains = sample(Xoshiro(22), BNN_inference, sampling_alg, MCMCThreads(), N_samples, N_chains; discard_adapt=false, initial_params=init_p_)
		#chains = sample(Xoshiro(22), BNN_inference, sampling_alg, N_samples; discard_adapt=false, initial_params=init_p_)
		chains = sample(Xoshiro(22), BNN_inference, sampling_alg, MCMCThreads(), N_samples, N_chains; discard_adapt=false)
		#chains = sample(Xoshiro(22), BNN_inference, Prior(), MCMCThreads(), N_samples, N_chains; discard_adapt=false)

		return chains, ps_BNN, st_BNN, chains_prior
	elseif sample_type == "VI"
		qo = q_fullrank_gaussian(BNN_inference)
		#qo = q_meanfield_gaussian(BNN_inference)
		q_fr, _, _, _ = vi(Xoshiro(22), BNN_inference, qo, N_samples; adtype=ad, show_progress=false)
		z = rand(Xoshiro(22), q_fr, N_samples*4)
		
		varinf = Turing.DynamicPPL.VarInfo(BNN_inference)
		vns_and_values = Turing.DynamicPPL.varname_and_value_leaves(
			Turing.DynamicPPL.values_as(varinf, OrderedDict),
		)
		varnames = map(first, vns_and_values)

		chains = Chains(reshape(z', (size(z, 2), size(z, 1), 1)), varnames)

		return chains, ps_BNN, st_BNN, 0.0
	end

end

# ╔═╡ 60fe0f3c-89e0-4996-8af6-7424b772cefa
#chains, ps_BNN, st_BNN, chains_prior = BNN(BNN_arch, N_samples)

# ╔═╡ 0042643c-bf69-4a8d-b727-c0c875c77973
summarize(chains)

# ╔═╡ a18df247-d38f-4def-b56f-7b025ca57e2f
StatsPlots.plot(chains[1:end, 1:75:end, :])

# ╔═╡ ef5320bf-0b65-45da-9a9b-7c52dca56733
d_chains = describe(chains[1:end, :, :])

# ╔═╡ c5f772b2-ab1d-4612-8d49-fcffea9772b7
d_chains[1][1:end, :rhat] |> mean

# ╔═╡ 930e4cb8-5374-465c-8288-493bfeb9c333
d_chains[1][1:end, :ess_bulk] |> mean

# ╔═╡ f378bc31-9a5f-4cda-9754-929c4dc89b3d
d_chains[1][1:end, :ess_per_sec] |> mean

# ╔═╡ ca031860-7d8c-4cb0-8f00-c7e4bf232316
gelmandiag(chains)[1:end, :psrf] |> mean

# ╔═╡ e3d59361-1d40-41dd-88b7-4cddcdf69d79
# Extract all weight and bias parameters.
θ = MCMCChains.group(chains, :parameters).value;

# ╔═╡ 89458806-9364-40a4-a7db-c57d0d6751ad
argmax(chains[:lp])[1]

# ╔═╡ c3ce3d5f-1fd7-4523-9184-1a51eba6a75f
begin
	idx = argmax(chains[:lp])[1]
	ch_idx = argmax(chains[:lp])[2]
end

# ╔═╡ 0e0860eb-5fe6-451b-9ecf-40c48f49a233
begin
	θ_MAP = θ[idx, :, ch_idx]
	ps_MAP = vector_to_parameters(θ_MAP, ps_BNN)
end

# ╔═╡ 2a172202-dea1-4101-a49a-4a8d64c68e3c
function predict(m, xs, single_chain, p_, st_)
	st_test = Lux.testmode(st_)
	return (Lux.apply(m, xs |> f64, vector_to_parameters(single_chain, p_) |> f64, st_test)[1])
end

# ╔═╡ b9886727-893a-47b5-9df6-2c55e3811406
#θ[idx, :, 1]

# ╔═╡ 06de1912-a662-4988-8f95-a78322757f2f
function predict(m, xs, chains, p_, st_, N_sims)
	posterior_samples = sample(Xoshiro(1111), chains[half_N_samples:end, :, :], N_sims)
	#α_sample = Matrix(MCMCChains.group(posterior_samples, :α).value[:, :, 1])
	θ_sample = Matrix(MCMCChains.group(posterior_samples, :parameters).value[:, :, 1])
	σ_sample = Matrix(MCMCChains.group(posterior_samples, :σ).value[:, :, 1])
	
	st_test = Lux.testmode(st_)

	function fwd_pass(par, q)
		res = (Lux.apply(m, xs |> f64, vector_to_parameters(par, p_) |> f64, st_test)[1])
		res_μ = res[:, 1]
		res_σ² = exp.(res[:, 2])
		return vec(rand(Xoshiro(1111+q), MvNormal(res_μ, I * res_σ²)))
	end

	#y_pred_f = hcat([fwd_pass(θ_sample[k, :] .* α_sample[k, :], k) for k ∈ 1:N_sims]...)'
	y_pred_f = hcat([fwd_pass(θ_sample[k, :], k) for k ∈ 1:N_sims]...)'

	return quantile.(eachcol(y_pred_f), 0.5) |> vec, quantile.(eachcol(y_pred_f), 0.025) |> vec, quantile.(eachcol(y_pred_f), 1-0.025) |> vec
end

# ╔═╡ c1ca74d8-8c90-427b-8bf2-41ab1d13bcf3
function predict(m, Xs, single_chain, p_, st_, time_step, x_1, x_2, standariser)
	X_ = deepcopy(Xs)
	pred_full = zeros(time_step, 99-τ₀+1)

	for i ∈ 1:time_step
		pred = predict(m, X_, single_chain, p_, st_)
		pred_full[i, :] = pred
		pred_scale = standariser.(pred, x_1, x_2)
		# Shift downwards
		for j ∈ 1:(T-1)
			X_[:, j, :] = X_[:, j+1, :]
		end
		for k ∈ 1:τ₀
			X_[k, end, :] = pred_scale
		end
	end

	return pred_full
end

# ╔═╡ 950c5db7-fabc-4da1-a14d-5f8d83f9f4f2
function predict(m::Chain, xs, chains, p_, st_, time_step, x_1, x_2, standariser, N_sims)
	posterior_samples = sample(Xoshiro(1111), chains[half_N_samples:end, :, :], N_sims)
	#α_sample = Matrix(MCMCChains.group(posterior_samples, :α).value[:, :, 1])
	θ_sample = Matrix(MCMCChains.group(posterior_samples, :parameters).value[:, :, 1])
	
	
	st_test = Lux.testmode(st_)

	fwd_pass(par) = predict(m, xs, par, p_, st_test, time_step, x_1, x_2, standariser)

	y_pred_f = zeros(time_step, size(xs)[3], N_sims)
	
	for k ∈ 1:N_sims
		#y_pred_f[:, :, k] = fwd_pass(θ_sample[k, :] .* α_sample[k, :])
		y_pred_f[:, :, k] = fwd_pass(θ_sample[k, :])
	end

	p_50 = zeros(time_step, size(xs)[3])
	p_lb = zeros(time_step, size(xs)[3])
	p_ub = zeros(time_step, size(xs)[3])
	for i ∈ 1:time_step
		y_pred_temp = reshape(y_pred_f[i, :, :], size(xs)[3], N_sims)
		p_50[i, :] = quantile.(eachrow(y_pred_temp), 0.5)
		p_lb[i, :] = quantile.(eachrow(y_pred_temp), 0.025)
		p_ub[i, :] = quantile.(eachrow(y_pred_temp), 0.975)
	end

	return p_50, p_lb, p_ub
end

# ╔═╡ 3c47c725-8751-4088-bc99-39c3cc653377
predict(tstate, X_train)

# ╔═╡ c0047459-5f14-4af7-b541-8238763d4a70
y_pred_valid = predict(tstate, Matrix(TEST[end_year .< TEST.Year .≤ forecast_year  .&& TEST[:, :log_mx] .< 0, ["Year_std", "Age_std", "Birth_year_std", "PopID"]])')[:, 1]

# ╔═╡ 7e8e62f5-28a1-4153-892a-fc8988130f4b
@info "MSE Testing x10e-4: $(mean((exp.(Matrix(TEST[end_year .< TEST.Year .≤ forecast_year .&& TEST[:, :log_mx] .< 0, ["log_mx"]]) |> vec) .- exp.(y_pred_valid)) .^ 2) * 1e4)"

# ╔═╡ e1e33d8e-0fd1-42d9-84ff-3b5ca4161c11
@info "Log RMSE Testing: $(mean(((Matrix(TEST[end_year .< TEST.Year .≤ forecast_year .&& TEST[:, :log_mx] .< 0, ["log_mx"]]) |> vec) .- (y_pred_valid)) .^ 2) |> sqrt)"

# ╔═╡ f1f84de0-488a-4bad-a2a4-64965d493dc7
y_pred_train = predict(tstate, Matrix(TEST[end_year .≥ TEST.Year .&& TEST[:, :log_mx] .< 0, ["Year_std", "Age_std", "Birth_year_std", "PopID"]])')[:, 1]

# ╔═╡ c8bca677-24d5-4bcc-b881-e0f43f208ca9
@info "MSE Training x10e-4: $(mean((exp.(Matrix(TEST[end_year .≥ TEST.Year .&& TEST[:, :log_mx] .< 0, ["log_mx"]]) |> vec) .- exp.(y_pred_train)) .^ 2) * 1e4)"

# ╔═╡ 36452b3e-4f2a-4f20-aec2-228b8d685c82
@info "Log RMSE Training: $(mean(((Matrix(TEST[end_year .≥ TEST.Year .&& TEST[:, :log_mx] .< 0, ["log_mx"]]) |> vec) .- (y_pred_train)) .^ 2) |> sqrt)"

# ╔═╡ f099a171-e1ba-4d74-87a3-010e0e9ff27a
if model_type ≠ "NN"
	forecast = predict(tstate, X_valid, (extended_forecast_year-end_year+1), x_1, x_2, standardise)
end

# ╔═╡ b8fc95a6-b495-4b64-a784-906937d19135
predict(BNN_arch, X_train, θ_MAP, ps_MAP, st_BNN)[:, 2]

# ╔═╡ c8fa5746-8eec-45a1-bda5-871ef71de684
@info "Observed Data: $(sum(TEST.Observed))"

# ╔═╡ 09f6c249-dbce-40bf-a879-39c85650a8cb
@info "All Data: $(sum(TEST.Year .≤ end_year))"

# ╔═╡ 0b8f579c-36e9-4660-9793-c537bb5e275d
TEST

# ╔═╡ 7c48fa09-5061-43e6-8483-961b3d2cee0c
begin
	actual_set_train = TEST[TEST.Year .≤ end_year .&& TEST[:, :log_mx] .< 0, ["log_mx"]]
	train_set = Matrix(TEST[TEST.Year .≤ end_year .&& TEST[:, :log_mx] .< 0, ["Year_std", "Age_std", "Birth_year_std", "PopID"]])'
	actual_set_test = TEST[end_year .< TEST.Year .≤ forecast_year .&& TEST[:, :log_mx] .< 0, ["log_mx"]]
	test_set = Matrix(TEST[end_year .< TEST.Year .≤ forecast_year .&& TEST[:, :log_mx] .< 0, ["Year_std", "Age_std", "Birth_year_std", "PopID"]])'
	
	pred_set_train, lb_train, ub_train = predict(BNN_arch, train_set, chains, ps_BNN, st_BNN, 10_000)
	pred_set_train_MAP = predict(BNN_arch, train_set, θ_MAP, ps_MAP, st_BNN)[:, 1]
	pred_set_test, lb_test, ub_test = predict(BNN_arch, test_set, chains, ps_BNN, st_BNN, 10_000)
	pred_set_test_MAP = predict(BNN_arch, test_set, θ_MAP, ps_MAP, st_BNN)[:, 1]

	mse_exp(actual, pred) = mean(((exp.(actual) .- exp.(pred)) .^ 2).log_mx)
	mse_log(actual, pred) = mean(((actual .- pred) .^ 2).log_mx)
	mae_exp(actual, pred) = mean(abs.(exp.(actual) .- exp.(pred)).log_mx)
	mae_log(actual, pred) = mean(abs.(actual .- pred).log_mx)
	rmse_exp(actual, pred) = sqrt(mse_exp(actual, pred))
	rmse_log(actual, pred) = sqrt(mse_log(actual, pred))

	mwip_exp(u, l) = mean(exp.(u) .- exp.(l))
	mwip_log(u, l) = mean(u .- l)
	picp_exp(u, l, A) = mean(exp.(l) .< exp.(A.log_mx) .< exp.(u))
	picp_log(u, l, A) = mean(l .< A.log_mx .< u)


	@info "MPIW (train): $(mwip_exp(ub_train, lb_train))"
	@info "MPIW (test): $(mwip_exp(ub_test, lb_test))"

	@info "PICP (train): $(picp_exp(ub_train, lb_train, actual_set_train) * 1e2)%"
	@info "PICP (test): $(picp_exp(ub_test, lb_test, actual_set_test) * 1e2)%"


	@info "MSE Training Set: $(1e4 * mse_exp(actual_set_train, pred_set_train))×10e-4"
	@info "Log RMSE Training Set: $(rmse_log(actual_set_train, pred_set_train))"
	
	@info "MSE Training Set (MAP): $(1e4 * mse_exp(actual_set_train, pred_set_train_MAP))×10e-4"
	@info "Log RMSE Training Set (MAP): $(rmse_log(actual_set_train, pred_set_train_MAP))"
	
	@info "MSE Testing Set: $(1e4 * mse_exp(actual_set_test, pred_set_test))×10e-4"
	@info "Log RMSE Testing Set: $(rmse_log(actual_set_test, pred_set_test))"

	@info "MSE Testing Set (MAP): $(1e4 * mse_exp(actual_set_test, pred_set_test_MAP))×10e-4"
	@info "Log RMSE Testing Set (MAP): $(rmse_log(actual_set_test, pred_set_test_MAP))"
	
	@info "MAE Training Set: $(1e3 * mae_exp(actual_set_train, pred_set_train))×10e-3"
	@info "Log MAE Training Set: $(mae_log(actual_set_train, pred_set_train))"

	@info "MAE Training Set (MAP): $(1e3 * mae_exp(actual_set_train, pred_set_train_MAP))×10e-3"
	@info "Log MAE Training Set (MAP): $(mae_log(actual_set_train, pred_set_train_MAP))"
	
	@info "MAE Testing Set: $(1e3 * mae_exp(actual_set_test, pred_set_test))×10e-3"
	@info "Log MAE Testing Set: $(mae_log(actual_set_test, pred_set_test))"

	@info "MAE Testing Set (MAP): $(1e3 * mae_exp(actual_set_test, pred_set_test_MAP))×10e-3"
	@info "Log MAE Testing Set (MAP): $(mae_log(actual_set_test, pred_set_test_MAP))"
end

# ╔═╡ c59ed9df-f944-4ee6-881e-2986dc8b1d3d
begin
	if model_type == "NN"
		pop_id = 1
		country_name = [k for (k,v) in country_embeddings if v == pop_id][1]
		# Get first year of validation/test set
		X_test_valid = Matrix(TEST[TEST.Year .== end_year .&& TEST.PopID .== pop_id, ["Year_std", "Age_std", "Birth_year_std", "PopID"]])'
		y_pred_test = predict(tstate, X_test_valid)[:, 1]
		y_actuals_plotting = TEST[TEST.Year .== end_year .&& TEST.PopID .== pop_id .&& TEST.log_mx .< 0.0, ["Age", "log_mx"]]
		y_actuals_obs_plotting = TEST[TEST.Year .== end_year .&& TEST.PopID .== pop_id .&& TEST.Observed .== 1 .&& TEST.log_mx .< 0.0, ["Age", "log_mx"]]
		

		plot(title="Validation Set ($end_year): $(country_name)\n Lee-Carter NN", xlab="Age", ylab="log μ", legend=:topleft)#, ylim=(-12.0, 0.0))
		
		median_pred, lb_pred, ub_pred = predict(BNN_arch, X_test_valid, chains, ps_BNN, st_BNN, 10_000)

		MAP_pred = predict(BNN_arch, X_test_valid, θ_MAP, ps_MAP, st_BNN)[:, 1]

		
		plot!(start_age:end_age, vec(y_pred_test), label="Predicted: AdamW", width=2, color=:blue)
		plot!(start_age:end_age, vec(median_pred), label="Predicted: BNN Median", width=2, color=:red)
		plot!(start_age:end_age, lb_pred, fillrange=ub_pred, label="Predicted: BNN 95% CI", width=0.9, color=:red, alpha=0.2)
		plot!(start_age:end_age, vec(MAP_pred), label="Predicted: BNN MAP", width=2, color=:red, linestyle=:dash)
		scatter!(y_actuals_plotting.Age, y_actuals_plotting.log_mx, label="Actual: Complete", color=:orange)
		scatter!(y_actuals_obs_plotting.Age, y_actuals_obs_plotting.log_mx, label="Actual: Observed", color=:black)
		
	elseif model_type == "LSTM"
		#start_age = τ₀ - 2
		
		plot(title="Validation Set ($end_year): $(country)\n τ₀=$τ₀, τ₁=$τ₁, T=$T, cell=$cell, depth=$NN_depth", xlab="Age", ylab="log μ", legend=:topleft, )#ylim=(-12.0, 0.0))
		
		median_pred, lb_pred, ub_pred = predict(BNN_arch, X_valid, chains, ps_BNN, st_BNN, 1_000)
		
		#plot!((start_age+1):(end_age-1), vec(y_pred_valid), label="Predicted: ADAM", width=2, color=:blue)
		plot!((start_age+1):(end_age-1), vec(median_pred), label="Predicted: BNN Median", width=2, color=:red)
		plot!((start_age+1):(end_age-1), lb_pred, fillrange=ub_pred, label="Predicted: BNN 95% CI", width=0.9, color=:red, alpha=0.2)
		scatter!((start_age+1):(end_age-1), vec(y_valid'), label="Observed", color=:orange)

	elseif model_type == "CNN"
		
		plot(title="Validation Set ($end_year): $(country)\nT=$T, CNN", xlab="Age", ylab="log μ", legend=:topleft)#, ylim=(-12.0, 0.0))
		
		median_pred, lb_pred, ub_pred = predict(BNN_arch, X_valid, chains, ps_BNN, st_BNN, 1_000)
		
		plot!(start_age:end_age, vec(y_pred_valid), label="Predicted: ADAM", width=2, color=:blue)
		plot!(start_age:end_age, vec(median_pred), label="Predicted: BNN Median", width=2, color=:red)
		plot!(start_age:end_age, lb_pred, fillrange=ub_pred, label="Predicted: BNN 95% CI", width=0.9, color=:red, alpha=0.2)
		scatter!(start_age:end_age, vec(y_valid'), label="Observed", color=:orange)
	end
end

# ╔═╡ 57f6b96c-98e4-4a68-942c-13d1036ce643
begin
	actual_set_train_ci = TEST[TEST.Year .≤ end_year .&& TEST[:, :log_mx] .< 0 .&& TEST.PopID .== pop_id, ["log_mx"]]
	train_set_ci = Matrix(TEST[TEST.Year .≤ end_year .&& TEST[:, :log_mx] .< 0 .&& TEST.PopID .== pop_id, ["Year_std", "Age_std", "Birth_year_std", "PopID"]])'
	actual_set_test_ci = TEST[end_year .< TEST.Year .≤ forecast_year .&& TEST[:, :log_mx] .< 0 .&& TEST.PopID .== pop_id, ["log_mx"]]
	test_set_ci = Matrix(TEST[end_year .< TEST.Year .≤ forecast_year .&& TEST[:, :log_mx] .< 0 .&& TEST.PopID .== pop_id, ["Year_std", "Age_std", "Birth_year_std", "PopID"]])'
	
	pred_set_train_ci, lb_train_ci, ub_train_ci = predict(BNN_arch, train_set_ci, chains, ps_BNN, st_BNN, 10_000)
	pred_set_train_MAP_ci = predict(BNN_arch, train_set_ci, θ_MAP, ps_MAP, st_BNN)[:, 1]
	pred_set_test_ci, lb_test_ci, ub_test_ci = predict(BNN_arch, test_set_ci, chains, ps_BNN, st_BNN, 10_000)
	pred_set_test_MAP_ci = predict(BNN_arch, test_set_ci, θ_MAP, ps_MAP, st_BNN)[:, 1]


	@info "MPIW (train): $(mwip_exp(ub_train_ci, lb_train_ci))"
	@info "MPIW (test): $(mwip_exp(ub_test_ci, lb_test_ci))"

	@info "PICP (train): $(picp_exp(ub_train_ci, lb_train_ci, actual_set_train_ci) * 1e2)%"
	@info "PICP (test): $(picp_exp(ub_test_ci, lb_test_ci, actual_set_test_ci) * 1e2)%"


	@info "MSE Training Set: $(1e4 * mse_exp(actual_set_train_ci, pred_set_train_ci))×10e-4"
	@info "Log RMSE Training Set: $(rmse_log(actual_set_train_ci, pred_set_train_ci))"
	
	@info "MSE Training Set (MAP): $(1e4 * mse_exp(actual_set_train_ci, pred_set_train_MAP_ci))×10e-4"
	@info "Log RMSE Training Set (MAP): $(rmse_log(actual_set_train_ci, pred_set_train_MAP_ci))"
	
	@info "MSE Testing Set: $(1e4 * mse_exp(actual_set_test_ci, pred_set_test_ci))×10e-4"
	@info "Log RMSE Testing Set: $(rmse_log(actual_set_test_ci, pred_set_test_ci))"

	@info "MSE Testing Set (MAP): $(1e4 * mse_exp(actual_set_test_ci, pred_set_test_MAP_ci))×10e-4"
	@info "Log RMSE Testing Set (MAP): $(rmse_log(actual_set_test_ci, pred_set_test_MAP_ci))"
	
	@info "MAE Training Set: $(1e3 * mae_exp(actual_set_train_ci, pred_set_train_ci))×10e-3"
	@info "Log MAE Training Set: $(mae_log(actual_set_train_ci, pred_set_train_ci))"

	@info "MAE Training Set (MAP): $(1e3 * mae_exp(actual_set_train_ci, pred_set_train_MAP_ci))"
	@info "Log MAE Training Set (MAP): $(mae_log(actual_set_train_ci, pred_set_train_MAP_ci))"
	
	@info "MAE Testing Set: $(1e3 * mae_exp(actual_set_test_ci, pred_set_test_ci))×10e-3"
	@info "Log MAE Testing Set: $(mae_log(actual_set_test_ci, pred_set_test_ci))"

	@info "MAE Testing Set (MAP): $(1e3 * mae_exp(actual_set_test_ci, pred_set_test_MAP_ci))×10e-3"
	@info "Log MAE Testing Set (MAP): $(mae_log(actual_set_test_ci, pred_set_test_MAP_ci))"
end

# ╔═╡ d826f3b4-8a5f-4f99-88fb-d9b8420c6d89
begin
	forecast_year_ = 1980
	if model_type ≠ "NN"
		plot(title="Year $forecast_year_: $(country)\n τ₀=$τ₀, τ₁=$τ₁, T=$T, cell=$cell, depth=$NN_depth, gender=$gender_", xlab="Year", ylab="log μ", legend=:topleft)

		median_forecast, lb_forecast, ub_forecast = predict(BNN_arch, X_valid, chains, ps_BNN, st_BNN, (extended_forecast_year-end_year+1), x_1, x_2, standardise, 1_000)
		
		plot!(start_age:end_age, vec(forecast[forecast_year_-end_year+1, :]'), label="Forecast: $opt", color=:blue, width=2)
		plot!(start_age:end_age, vec(median_forecast[forecast_year_-end_year+1, :]'), label="Forecast: BNN Median", color=:red, width=2)
		plot!(start_age:end_age, vec(lb_forecast[forecast_year_-end_year+1, :]'), fillrange=vec(ub_forecast[forecast_year_-end_year+1, :]'), label="Forecast: BNN 95% CI", color=:red, width=0.9, alpha=0.2)
		scatter!(y_test[y_test.Year .== forecast_year_, :Age], y_test[y_test.Year .== forecast_year_, gender_], label="Actual", color=:orange)
	else
		X_test_forecast = Matrix(TEST[TEST.Year .== forecast_year_ .&& TEST.PopID .== pop_id, ["Year_std", "Age_std", "Birth_year_std", "PopID"]])'
		y_pred_forecast = predict(tstate, X_test_forecast)[:, 1]
		y_actuals_forecast = TEST[TEST.Year .== forecast_year_ .&& TEST.PopID .== pop_id, ["Age", "log_mx"]]
		y_actuals_obs_forecast = TEST[TEST.Year .== forecast_year_ .&& TEST.PopID .== pop_id .&& TEST.Observed .== 1 .&& TEST.log_mx .< 0.0, ["Age", "log_mx"]]

		MAP_forecast = predict(BNN_arch, X_test_forecast, θ_MAP, ps_MAP, st_BNN)[:, 1]
		
		plot(title="Forecast (Year $forecast_year_): $(country)\n τ₁=$τ₁, act=$act, FNN, depth=$NN_depth", xlab="Age", ylab="log μ", legend=:best, ylim=(-12.0, 0.0))
		plot(title="Year $forecast_year_: $(country_name)\n Lee-Carter NN", xlab="Age", ylab="log μ", legend=:best, ylim=(-12.0, -0.001))
		
		median_forecast, lb_forecast, ub_forecast = predict(BNN_arch, X_test_forecast, chains, ps_BNN, st_BNN, 10_000)

		plot!(start_age:end_age, vec(y_pred_forecast), label="Predicted: AdamW", width=2, color=:blue)
		plot!(start_age:end_age, vec(median_forecast), label="Predicted: BNN Median", width=2, color=:red)
		plot!(start_age:end_age, vec(MAP_forecast), label="Predicted: BNN MAP", width=2, color=:red, linestyle=:dash)
		plot!(start_age:end_age, vec(lb_forecast), fillrange=vec(ub_forecast), label="Predicted: BNN 95% CI", color=:red, width=0.9, alpha=0.2)
		scatter!(y_actuals_forecast.Age, y_actuals_forecast.log_mx, label="Actual: Complete", color=:orange)
		scatter!(y_actuals_obs_forecast.Age, y_actuals_obs_forecast.log_mx, label="Actual: Observed", color=:black)
	end
end

# ╔═╡ c1b667d2-1411-4637-9309-967309cc30e6
begin
	forecast_age = 65
	if model_type ≠ "NN"
		adj_forecast_age = forecast_age + Int((3 - τ₀)/2)
		@assert forecast_age ≥ (τ₀ + 1)/2 - 1
		
		plot(title="Forecast (Age $forecast_age): $(country)\n τ₀=$τ₀, τ₁=$τ₁, T=$T, cell=$cell, depth=$NN_depth", xlab="Year", ylab="log μ", legend=:best)
	
		median_forecast2, lb_forecast2, ub_forecast2 = predict(BNN_arch, X_valid, chains, ps_BNN, st_BNN, (extended_forecast_year-end_year+1), x_1, x_2, standardise, 1_000)
	
		plot!(end_year:extended_forecast_year, forecast[:, adj_forecast_age], label="Forecast: $opt", color=:blue, width=2)
		plot!(end_year:extended_forecast_year, median_forecast2[:, adj_forecast_age], label="Forecast: BNN Median", color=:red, width=2)
		plot!(end_year:extended_forecast_year, lb_forecast2[:, adj_forecast_age], fillrange=ub_forecast2[:, adj_forecast_age], label="Forecast: BNN 95% CI", color=:red, width=0.9, alpha=0.2)
		scatter!(y_test[y_test.Age .== forecast_age, :Year], y_test[y_test.Age .== forecast_age, :Female], label="Actual", color=:orange)
	else
		X_test_forecast_ = Matrix(TEST[TEST.Age .== forecast_age .&& TEST.PopID .== pop_id, ["Year_std", "Age_std", "Birth_year_std", "PopID"]])'
		y_pred_forecast_ = predict(tstate, X_test_forecast_)[:, 1]
		MAP_forecast_ = predict(BNN_arch, X_test_forecast_, θ_MAP, ps_MAP, st_BNN)[:, 1]
		y_actuals_forecast_ = TEST[TEST.Age .== forecast_age .&& TEST.PopID .== pop_id .&& TEST.log_mx .< 0.0, ["Year", "log_mx"]]
		y_actuals_obs_forecast_ = TEST[TEST.Age .== forecast_age .&& TEST.PopID .== pop_id .&& TEST.log_mx .< 0.0 .&& TEST.Observed .== 1, ["Year", "log_mx"]]
		
		plot(title="Forecast for Age $forecast_age: $(country_name)\n Lee-Carter NN", xlab="Year", ylab="log μ", legend=:best)
		
		median_forecast2, lb_forecast2, ub_forecast2 = predict(BNN_arch, X_test_forecast_, chains, ps_BNN, st_BNN, 10_000)

		plot!(max(start_year, TEST.Year[1]):extended_forecast_year, y_pred_forecast_, label="Forecast: AdamW", color=:blue, width=2)
		plot!(max(start_year, TEST.Year[1]):extended_forecast_year, median_forecast2, label="Forecast: BNN Median", color=:red, width=2)
		plot!(max(start_year, TEST.Year[1]):extended_forecast_year, MAP_forecast_, label="Forecast: BNN MAP", color=:red, width=2, linestyle=:dash)
		plot!(max(start_year, TEST.Year[1]):extended_forecast_year, lb_forecast2, fillrange=ub_forecast2, label="Forecast: BNN 95% CI", color=:red, width=0.9, alpha=0.2)
		scatter!(y_actuals_forecast_.Year, y_actuals_forecast_.log_mx, label="Actual: Complete", color=:orange)
		scatter!(y_actuals_obs_forecast_.Year, y_actuals_obs_forecast_.log_mx,  label="Actual: Observed", color=:black)
	end
end

# ╔═╡ 01f3db0b-ffe7-49e0-83e3-df989126a945


# ╔═╡ 0ff0b225-01df-43c3-a755-481bda77c647
function predict__(m, xs, chains, p_, st_, N_sims, idx₁, idx₂)
	posterior_samples = sample(Xoshiro(1111), chains[10:end, :, :], N_sims)
	#α_sample = Matrix(MCMCChains.group(posterior_samples, :α).value[:, :, 1])
	θ_sample = Matrix(MCMCChains.group(posterior_samples, :parameters).value[:, :, 1])
	#σ_sample = Matrix(MCMCChains.group(posterior_samples, :σ).value[:, :, 1])

	_, st_ = Lux.setup(Xoshiro(12345), m) |> f64
	
	st_test = Lux.testmode(st_)

	function fwd_pass(par, idx₁, idx₂)
		if idx₁ == 1 && idx₂ == 2
			return exp.(vec(Lux.apply(m, xs |> f64, vector_to_parameters(par, p_) |> f64, st_test)[1][idx₁][idx₂, :]))
		else
			return vec(Lux.apply(m, xs |> f64, vector_to_parameters(par, p_) |> f64, st_test)[1][idx₁][idx₂, :])
		end
	end

	#y_pred_f = hcat([fwd_pass(θ_sample[k, :] .* α_sample[k, :], σ_sample[k, :][1], k, idx₁, idx₂) for k ∈ 1:N_sims]...)'
	y_pred_f = hcat([fwd_pass(θ_sample[k, :], idx₁, idx₂) for k ∈ 1:N_sims]...)'

	return quantile.(eachcol(y_pred_f), 0.5) |> vec, quantile.(eachcol(y_pred_f), 0.025) |> vec, quantile.(eachcol(y_pred_f), 1-0.025) |> vec
end

# ╔═╡ 3a8da928-5284-4ab3-8782-7ae678e0a49c
begin
	BNN_arch_ = LeeCarterNN_gender(τ₁; show_intermediate=true)
	X_test_forecast__ = Matrix(TEST[TEST.Age .== 65, ["Year_std", "Age_std"]])'
	median_forecast2_, lb_forecast2_, ub_forecast2_ = predict__(BNN_arch_, X_test_forecast__, chains, ps_BNN, st_BNN, 1_000, 2, 1)

	plot(xlab="Year", ylab="κ(t)", title="$(country)")
	plot!(max(start_year, TEST.Year[1]):extended_forecast_year, median_forecast2_,  label="BNN Median", color=:red, width=2)
	plot!(max(start_year, TEST.Year[1]):extended_forecast_year, lb_forecast2_, fillrange=ub_forecast2_, label="BNN 95% CI", color=:red, width=0.9, alpha=0.2, ylim=(minimum(median_forecast2_) - 0.001, 1.1*maximum(median_forecast2_)))
end

# ╔═╡ d26c22aa-d417-4268-ae7c-15345a63ed3b
BNN_arch_

# ╔═╡ bc381caa-73a3-4203-8742-0051e44a0464
begin
	X_test_forecast___ = Matrix(TEST[TEST.Year .== 2001, ["Year_std", "Age_std"]])'

	plt1 = begin
		median_forecast2__, lb_forecast2__, ub_forecast2__ = predict__(BNN_arch_, X_test_forecast___, chains, ps_BNN, st_BNN, 1_000, 1, 1)
		plot(xlab="Age", ylab="α(x)", title="$(country)")
		plot!(start_age:end_age, median_forecast2__,  label="BNN Median", color=:red, width=2)
		plot!(start_age:end_age, lb_forecast2__, fillrange=ub_forecast2__, label="BNN 95% CI", color=:red, width=0.9, alpha=0.2, ylim=(minimum(median_forecast2__) - 0.5, 1.1*maximum(median_forecast2__)))
	end

	plt2 = begin
		median_forecast2__, lb_forecast2__, ub_forecast2__ = predict__(BNN_arch_, X_test_forecast___, chains, ps_BNN, st_BNN, 1_000, 1, 2)
		plot(xlab="Age", ylab="β(x)", title="$(country)")
		plot!(start_age:end_age, median_forecast2__,  label="BNN Median", color=:red, width=2)
		plot!(start_age:end_age, lb_forecast2__, fillrange=ub_forecast2__, label="BNN 95% CI", color=:red, width=0.9, alpha=0.2, ylim=(minimum(median_forecast2__) - 0.1, 1.1*maximum(median_forecast2__)))
	end

	plot(plt1, plt2)
end

# ╔═╡ ab54a223-999e-424a-89bf-c66c629bd21b
heatmap((predict__(BNN_arch_, X_test_forecast___, chains, ps_BNN, st_BNN, 1_000, 1, 1)[1] .+ median_forecast2__ * exp.(median_forecast2_)')', xlab="Age", ylab="Time")

# ╔═╡ ea49af3a-d66d-4c0f-8660-9c5b4bbc6087
heatmap((median_forecast2__ * exp.(median_forecast2_)')', xlab="Age", ylab="Time")

# ╔═╡ 38dd4c3f-ca41-47dc-9ce0-ba4edfb69a51
begin
	kernel_ = Matern32Kernel() + LinearKernel()
end

# ╔═╡ f2d1e8da-460a-43b0-a965-600da15fadfa
if size(unique(X_train, dims=1), 1) != size(X_train, 1)
    println("CRITICAL WARNING: Your training data sample contains duplicate input rows!")
    # You might want to resample or remove duplicates here
end

# ╔═╡ dbda01f1-f36d-4582-87f5-affe42d840b1
function GP_model(kernel_)
	@model function gp_nd(X, y)
	    # Get the number of input dimensions from the data
	    n_dims = size(X, 1)
	
	    # --- Priors ---
	    # Prior for n lengthscales, one for each input dimension (ARD)
	    log_l ~ MvNormal(zeros(n_dims), 0.5 * I)
	
	    # Prior for the kernel variance (scalar, doesn't change)
	    σ_f ~ LogNormal(0.0, 0.5)
	
	    # Prior for the observation noise (scalar, doesn't change)
	    σ_n ~ LogNormal(-2.0, 1.0)
	
	    # --- Model construction ---
	    # Construct the kernel with n lengthscales
	    kernel = (σ_f^2) * with_lengthscale(kernel_, exp.(log_l))
	
	    # Construct the GP
	    gp = GP(kernel)
	
	    # Define the likelihood
	    y ~ gp(X, (σ_n^2) + 1e-6)
	end

	GP_inference = gp_nd(X_train |> f64, vec(y_train |> f64))
	#GP_inference = gp_nd(rand(2, 500) |> f64, vec(rand(1, 500) |> f64))

	#ad = AutoTracker()
	ad = AutoForwardDiff()
	sampling_alg = NUTS(0.9; adtype=ad)

	chains = sample(Xoshiro(202), GP_inference, sampling_alg, 2_000; discard_adapt=false)

	return chains
end

# ╔═╡ adc162e2-74d3-4e46-aea3-1cb0ef0fe532
#gp_model = GP_model(kernel_)

# ╔═╡ f89610b6-1075-4857-8b87-2d546ed918ad
describe(gp_model)

# ╔═╡ 5efd6c9b-c149-44ba-922a-1014ffe9eab8
StatsPlots.plot(gp_model[half_N_samples:end, :, :])

# ╔═╡ e35a4819-4288-437f-9cd5-a9581ec5737e
function predict_gp(X_test_, X_train_, y_train_, s_f, s_n, l_, kern)
	kernel = (s_f^2) * with_lengthscale(kern, l_)
	gp_ = GP(kernel)
	posterior_ = posterior(gp_(X_train_, (s_n^2)), y_train_)

	return mean(posterior_(X_test_)), std(posterior_(X_test_))
end

# ╔═╡ 2def0485-a36b-4f96-806f-74025f2bc589
function predict_gp(ch_gp, X_test, X_train, y_train, N_sims, kernel_)
	posterior_samples = sample(Xoshiro(1111), ch_gp[100:end, :, :], N_sims)
	σ_f_sample = Matrix(MCMCChains.group(posterior_samples, :σ_f).value[:, :, 1])
	σ_n_sample = Matrix(MCMCChains.group(posterior_samples, :σ_n).value[:, :, 1])
	l_ = exp.(Matrix(MCMCChains.group(posterior_samples, :log_l).value[:, :, 1]))

	p_v = reshape(vcat([predict_gp(X_test, X_train, y_train, σ_f_sample[i], σ_n_sample[i], l_[i, :], kernel_)[1] for i ∈ 1:N_sims]...), :, N_sims)
	s_v = reshape(vcat([predict_gp(X_test, X_train, y_train, σ_f_sample[i], σ_n_sample[i], l_[i, :], kernel_)[2] for i ∈ 1:N_sims]...), :, N_sims)

	return quantile.(eachrow(p_v), 0.5), quantile.(eachrow(p_v), 0.025), quantile.(eachrow(p_v), 0.975), quantile.(eachrow(s_v), 0.5)	
end

# ╔═╡ 1bdaa6b8-4245-4f59-8d63-00676b42623e
begin
	@info "MSE Training Set: $(1e4 * mean((exp.(TEST[TEST.Year .≤ end_year .&& TEST[:, gender_]  .< 0, gender_]) .- exp.(predict_gp(gp_model, Matrix(TEST[TEST.Year .≤ end_year .&& TEST[:, gender_]  .< 0, ["Year_std", "Age_std"]])', X_train, vec(y_train), 1_000, kernel_)[1])) .^ 2))×10e-4"
	@info "MSE Testing Set: $(1e4 * mean((exp.(TEST[end_year .< TEST.Year .≤ forecast_year .&& TEST[:, gender_]  .< 0, gender_]) .- exp.(predict_gp(gp_model, Matrix(TEST[end_year .< TEST.Year .≤ forecast_year .&& TEST[:, gender_]  .< 0, ["Year_std", "Age_std"]])', X_train, vec(y_train), 1_000, kernel_)[1])) .^ 2))×10e-4"
end

# ╔═╡ de4b5788-8d8d-44e4-8152-75bdfe07b664
begin
	forecast_year__ = 1980
	
	X_test_forecast____ = Matrix(TEST[TEST.Year .== forecast_year__, ["Year_std", "Age_std"]])'
	
	gp_output_mean, gp_output_lb, gp_output_ub, gp_output_s = predict_gp(gp_model, X_test_forecast____, X_train, vec(y_train), 1_000, kernel_)
	
	plot(start_age:end_age, gp_output_mean, width=2, color=:red, label="Median", ribbon=gp_output_s)
	plot!(start_age:end_age, gp_output_lb, fillrange=gp_output_ub, color=:red, fillalpha=0.2, label="95% CI", title="$forecast_year__")
	scatter!(TEST[TEST.Year .== forecast_year__ .&& TEST[:, gender_] .< 0, :Age], TEST[TEST.Year .== forecast_year__ .&& TEST[:, gender_] .< 0, gender_], label="Actual: Complete", color=:orange)
	scatter!(TEST[TEST.Year .== forecast_year__ .&& TEST.Observed .== 1 .&& TEST[:, gender_]  .< 0, :Age], TEST[TEST.Year .== forecast_year__ .&& TEST.Observed .== 1 .&& TEST[:, gender_]  .< 0, gender_], label="Actual: Observed", color=:black)

end

# ╔═╡ Cell order:
# ╠═6264e41e-e179-11ef-19c1-f135e97db7cc
# ╠═ef51c95e-d5ad-455a-9631-094823b695bb
# ╠═f6696102-2894-4d54-be66-32004ea6486d
# ╠═50b3b576-d941-4609-8469-6de51cfa1545
# ╠═9f21c322-25bd-434f-9f9d-1a931e8199ea
# ╠═d1eb691b-481f-45d5-b736-7f99f4b0b4d2
# ╠═a0132c58-f427-4c88-ba45-bd8b9d9f98d4
# ╠═5378ee86-303b-4269-88f3-6eeccc30cb15
# ╠═93779239-cd66-4be2-b70f-c7872a29a29f
# ╠═7970d4a4-48a2-4f9a-860e-cd0d2b999957
# ╠═8fef40e0-4528-4224-9d9e-6e306b626f7d
# ╠═99851338-fed0-4963-90e0-dc09bb3d480f
# ╠═5beb9242-c2ae-4bb3-89b0-c81ff07d5ffb
# ╠═760ced0b-17c0-43da-8ff8-e140c34b7d16
# ╠═395fc7db-7db3-403b-b793-b70749281ef4
# ╠═7513dbee-3d55-49d8-bcf4-3003cd13dbb5
# ╠═0841707c-3942-412c-b729-1778a81929d3
# ╠═7e061810-6edc-4b45-bb1d-8c6769d37a2d
# ╠═9fb77dbb-2244-4bd8-a29c-5ef84f117ce5
# ╠═52139393-d898-45d5-bc86-54fba2e41bba
# ╠═d8fd292f-7974-4e7b-a795-b263cea45fb7
# ╠═7565da20-a7b5-4a12-b75c-141e1374af72
# ╠═ec25dfee-6929-4c94-a023-1ca59fe8ad0b
# ╠═b15917eb-36a6-46c5-b05f-7140118b183a
# ╠═ebc03c34-6b56-43de-babf-6eb4811322a1
# ╠═4da18da3-6951-4b71-8a8e-953ecf0c0551
# ╠═fb46cbe5-dbde-4ca5-b500-10e4ff5106af
# ╠═f3dcda6d-9110-40b0-a450-767317dd136a
# ╠═e4917c43-df80-40ea-9908-29bf46334c46
# ╠═eb111789-8e19-46b7-b0d3-0d569bba2d01
# ╠═b046da21-63a9-4556-ab7b-24e1c8f629e0
# ╠═608363b0-5f75-45c8-8970-5170489a5eeb
# ╠═59eed207-e138-44be-a241-d4cbfde0f38c
# ╠═c4f2ff5e-8bea-41e8-8414-8ebe89bff363
# ╠═055dd1ae-39f1-4292-8a6e-00cfd2795688
# ╠═80e4878c-2fc6-483a-90b9-5bb3f362f749
# ╠═a2be51e0-f646-47db-bb3b-daa672fd4f44
# ╠═ad84dffe-d03b-4e5d-8856-347f4e0e4332
# ╠═40711e69-c166-4a2a-8d37-88f0c241fcce
# ╠═7b5558e7-30c8-4069-9175-6dd79d27c8f5
# ╠═d12da7dd-e186-4329-939e-92cd8702f2e6
# ╠═8e5c8f3f-cfee-4dbc-baf5-f392b26e0e54
# ╠═03acd606-0d03-457c-933c-2f1ff0caded6
# ╠═564d77df-85c1-406a-8964-3b14fca6328c
# ╠═1b83a76d-2587-4e3c-a000-901a5a165716
# ╠═b7d6736b-776e-49d1-ae18-4a64b07a1a24
# ╠═864a7e7f-d4e3-4ada-b81a-61d5574c7138
# ╠═1fe630ea-6926-4260-b549-12d11856a74a
# ╠═b34c44fd-fed8-4111-984e-39e98bfe1481
# ╠═7096f53d-8f11-415d-b419-4c28917d6dc2
# ╠═3c47c725-8751-4088-bc99-39c3cc653377
# ╠═096a8a27-ddad-4534-94e4-71f755d5f561
# ╠═bb6602f4-927e-473e-b8da-957395ed7617
# ╠═e006827b-2f8b-4b7a-9669-2fb64fadc129
# ╠═bbb80aa6-096c-4fae-bc1d-2e2f5ba28b2d
# ╠═4fb91714-4412-49c3-b466-8ac3ee439047
# ╠═b69e3b36-d0e6-46f3-962f-d2f840064260
# ╠═3c1d261e-3043-4b0b-965f-5c48e5466316
# ╠═d0f95b2e-2326-44fe-8078-5c89d13cb8d5
# ╠═cac4f05e-e40a-4814-bde1-0c72c047e2ff
# ╠═c49191f4-f092-44c9-9497-e0772b43b158
# ╠═2fa273ec-86e1-4e2c-a589-e79a09e0089d
# ╠═8b935b73-b129-4b65-808a-bccace7d0838
# ╠═c0047459-5f14-4af7-b541-8238763d4a70
# ╠═f1f84de0-488a-4bad-a2a4-64965d493dc7
# ╠═c8bca677-24d5-4bcc-b881-e0f43f208ca9
# ╠═36452b3e-4f2a-4f20-aec2-228b8d685c82
# ╠═7e8e62f5-28a1-4153-892a-fc8988130f4b
# ╠═e1e33d8e-0fd1-42d9-84ff-3b5ca4161c11
# ╠═f099a171-e1ba-4d74-87a3-010e0e9ff27a
# ╠═2598323d-8e4b-4789-8f77-97cc99b94b29
# ╠═0711bfc1-4337-4ae4-a5b0-c7b08dae2190
# ╠═5cd9678e-6dd7-4290-ad48-3377735f7696
# ╠═78baeae1-36cb-47a4-b82a-fba776e19635
# ╠═60fe0f3c-89e0-4996-8af6-7424b772cefa
# ╠═0042643c-bf69-4a8d-b727-c0c875c77973
# ╠═a18df247-d38f-4def-b56f-7b025ca57e2f
# ╠═ef5320bf-0b65-45da-9a9b-7c52dca56733
# ╠═c5f772b2-ab1d-4612-8d49-fcffea9772b7
# ╠═930e4cb8-5374-465c-8288-493bfeb9c333
# ╠═f378bc31-9a5f-4cda-9754-929c4dc89b3d
# ╠═ca031860-7d8c-4cb0-8f00-c7e4bf232316
# ╠═e3d59361-1d40-41dd-88b7-4cddcdf69d79
# ╠═89458806-9364-40a4-a7db-c57d0d6751ad
# ╠═c3ce3d5f-1fd7-4523-9184-1a51eba6a75f
# ╠═0e0860eb-5fe6-451b-9ecf-40c48f49a233
# ╠═2a172202-dea1-4101-a49a-4a8d64c68e3c
# ╠═b9886727-893a-47b5-9df6-2c55e3811406
# ╠═b8fc95a6-b495-4b64-a784-906937d19135
# ╠═06de1912-a662-4988-8f95-a78322757f2f
# ╠═c1ca74d8-8c90-427b-8bf2-41ab1d13bcf3
# ╠═950c5db7-fabc-4da1-a14d-5f8d83f9f4f2
# ╠═c8fa5746-8eec-45a1-bda5-871ef71de684
# ╠═09f6c249-dbce-40bf-a879-39c85650a8cb
# ╠═0b8f579c-36e9-4660-9793-c537bb5e275d
# ╠═7c48fa09-5061-43e6-8483-961b3d2cee0c
# ╠═c59ed9df-f944-4ee6-881e-2986dc8b1d3d
# ╠═57f6b96c-98e4-4a68-942c-13d1036ce643
# ╠═d826f3b4-8a5f-4f99-88fb-d9b8420c6d89
# ╠═c1b667d2-1411-4637-9309-967309cc30e6
# ╠═01f3db0b-ffe7-49e0-83e3-df989126a945
# ╠═0ff0b225-01df-43c3-a755-481bda77c647
# ╠═3a8da928-5284-4ab3-8782-7ae678e0a49c
# ╠═d26c22aa-d417-4268-ae7c-15345a63ed3b
# ╠═bc381caa-73a3-4203-8742-0051e44a0464
# ╠═ab54a223-999e-424a-89bf-c66c629bd21b
# ╠═ea49af3a-d66d-4c0f-8660-9c5b4bbc6087
# ╠═38dd4c3f-ca41-47dc-9ce0-ba4edfb69a51
# ╠═f2d1e8da-460a-43b0-a965-600da15fadfa
# ╠═dbda01f1-f36d-4582-87f5-affe42d840b1
# ╠═adc162e2-74d3-4e46-aea3-1cb0ef0fe532
# ╠═f89610b6-1075-4857-8b87-2d546ed918ad
# ╠═5efd6c9b-c149-44ba-922a-1014ffe9eab8
# ╠═e35a4819-4288-437f-9cd5-a9581ec5737e
# ╠═2def0485-a36b-4f96-806f-74025f2bc589
# ╠═1bdaa6b8-4245-4f59-8d63-00676b42623e
# ╠═de4b5788-8d8d-44e4-8152-75bdfe07b664
