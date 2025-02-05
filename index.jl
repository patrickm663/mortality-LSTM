### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 6264e41e-e179-11ef-19c1-f135e97db7cc
# ╠═╡ show_logs = false
begin
	using Pkg
	cd(".")
	Pkg.activate(".")
	Pkg.add(url="https://github.com/patrickm663/HMD.jl")
	Pkg.add(["CSV", "DataFrames", "Lux", "ADTypes", "Zygote", "Optimisers", "Plots", "ComponentArrays", "Distributions", "Functors", "Tracker", "Turing"])
end

# ╔═╡ ef51c95e-d5ad-455a-9631-094823b695bb
using ADTypes, Lux, Optimisers, Printf, Random, CSV, Plots, DataFrames, ComponentArrays, HMD, Zygote, Statistics, Distributions, Functors, Turing, Tracker

# ╔═╡ 50b3b576-d941-4609-8469-6de51cfa1545
begin
	start_year = 1990
	end_year = 2001
	forecast_year = 2019
	extended_forecast_year = 2035
	τ₀ = 1
	τ₁ = 8
	T = 10
	NN_depth = 4
	cell = GRUCell #LSTMCell
	act = tanh
	list_of_countries = HMD.get_countries()
	country = list_of_countries["Japan"]
	lr = 0.05
	opt = Adam(lr)#NAdam(lr)
	model_type = "LSTM"
	# Hard-code
	if model_type == "NN"
		τ₀ = 2
	end
end

# ╔═╡ a0132c58-f427-4c88-ba45-bd8b9d9f98d4
list_of_countries

# ╔═╡ 5378ee86-303b-4269-88f3-6eeccc30cb15
begin
	min_max(x, xmin, xmax) = 2*(x - xmin) / (xmax - xmin) - 1
	standardise(x, x_mu, x_sigma) = (x - x_mu) / x_sigma
end

# ╔═╡ 93779239-cd66-4be2-b70f-c7872a29a29f
function LSTM(in_dims, hidden_dims, out_dims; depth=1, cell=LSTMCell)
	@assert depth > 0
	@assert cell ∈ [LSTMCell, GRUCell, RNNCell]
	if depth == 1
		return Chain(
			Recurrence(cell(in_dims => hidden_dims); return_sequence=false),
			Dense(hidden_dims => out_dims, exp)
		)
	elseif depth == 2
		return Chain(
			Recurrence(cell(in_dims => hidden_dims); return_sequence=true),
			Recurrence(cell(hidden_dims => hidden_dims); return_sequence=false),
			Dense(hidden_dims => out_dims, exp)
		)
	elseif depth > 2
		return Chain(
			Recurrence(cell(in_dims => hidden_dims); return_sequence=true),
			[Recurrence(cell(hidden_dims => hidden_dims); return_sequence=true) for _ in 1:(depth-2)],
			Recurrence(cell(hidden_dims => hidden_dims); return_sequence=false),
			Dense(hidden_dims => out_dims, exp)
		)
	end
end

# ╔═╡ 608363b0-5f75-45c8-8970-5170489a5eeb
function FNN(in_dims, hidden_dims, out_dims; depth=2, act=tanh, outer_act=exp)
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
function get_data(country; T=10, τ₀=3, start_year=1990, end_year=2001, LSTM_format=true, min_max_scale=true)
	# Apply 'Toy example' data pre-processing
	
    # Load from CSV
    data = CSV.read("data/$(country)_Mx_1x1.csv", DataFrame)

	# Split out females, aged 0-98, years start_year-end_year
	start_year = max(start_year, minimum(data.Year))
	end_year = min(end_year, maximum(data.Year)) |> f32

	@show start_year, end_year
	
	@assert start_year < end_year
	@assert end_year - start_year ≥ T

	y_test = data[(data.Age .≤ 98) .&& (end_year .≤ data.Year .≤ forecast_year) .&& (data.Female .> 0.0), [:Year, :Age, :Female]]
	y_test.Female .= log.(y_test.Female)
	
	data = data[(data.Age .≤ 98) .&& (start_year .≤ data.Year .≤ end_year), [:Year, :Age, :Female]]

	if LSTM_format == true
	
		# Group age x year and drop the age column (1)
		data = HMD.transform(data, :Female)[:, 2:end]
	
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
		y_train = -y_[1, :] |> f32
		y_valid = -y_[2, :] |> f32
	
		return X_train, reshape(y_train, 1, :), X_valid, reshape(y_valid, 1, :), X_1, X_2, y_test
		
	else
		# Drop missing from training set and validation sets
		# But retain it for a final X test set
		data = data[data.Female .> 0, :]
		
		train = data[data.Year .< end_year, :]
		valid = data[data.Year .≥ end_year, :]

		train = Matrix(train) |> f32
		valid = Matrix(valid) |> f32

		# Recreate the validation set but using all ages to test interpolation
		X_test = hcat(repeat([end_year * 1.0], 99), 0:98) |> f32

		X_train = train[:, 1:2]
		X_valid = valid[:, 1:2]
		y_train = -log.(train[:, end])
		y_valid = -log.(valid[:, end])

		validation_ages = X_valid[:, 2]

		if min_max_scale == true
			year_min, year_max = extrema(X_train[:, 1])
			age_min, age_max = extrema(X_train[:, 2])
	
			X_train[:, 1] .= min_max.(X_train[:, 1], year_min, year_max)
			X_valid[:, 1] .= min_max.(X_valid[:, 1], year_min, year_max)
			X_test[:, 1] .= min_max.(X_test[:, 1], year_min, year_max)
	
			X_train[:, 2] .= min_max.(X_train[:, 2], age_min, age_max)
			X_valid[:, 2] .= min_max.(X_valid[:, 2], age_min, age_max)
			X_test[:, 2] .= min_max.(X_test[:, 2], age_min, age_max)
		else
			year_mu, year_sigma = (mean(X_train[:, 1]), std(X_train[:, 1]))
			age_mu, age_sigma = (mean(X_train[:, 2]), std(X_train[:, 2]))
	
			X_train[:, 1] .= standardise.(X_train[:, 1], year_mu, year_sigma)
			X_valid[:, 1] .= standardise.(X_valid[:, 1], year_mu, year_sigma)
			X_test[:, 1] .= standardise.(X_test[:, 1], year_mu, year_sigma)
	
			X_train[:, 2] .= standardise.(X_train[:, 2], age_mu, age_sigma)
			X_valid[:, 2] .= standardise.(X_valid[:, 2], age_mu, age_sigma)
			X_test[:, 2] .= standardise.(X_test[:, 2], age_mu, age_sigma)
		end

		return X_train', reshape(y_train, 1, :), X_valid', reshape(y_valid, 1, :), X_test', validation_ages, y_test
		
	end
		
end

# ╔═╡ 7b5558e7-30c8-4069-9175-6dd79d27c8f5
if model_type == "NN"
	X_train, y_train, X_valid, y_valid, X_test, X_valid_ages, y_test = get_data(country; T=T, τ₀=τ₀, LSTM_format=false,  min_max_scale=true, start_year=start_year)
else
	X_train, y_train, X_valid, y_valid, x_1, x_2, y_test = get_data(country; T=T, τ₀=τ₀, LSTM_format=true,  min_max_scale=true, start_year=start_year)
end

# ╔═╡ 564d77df-85c1-406a-8964-3b14fca6328c
function main(tstate, vjp, data, epochs; early_stopping=true)
	loss_function = MSELoss()
	train_losses = []
	valid_losses = []
	
    for epoch in 1:epochs
        _, train_loss, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)
		push!(train_losses, train_loss)
		valid_pred = (Lux.apply(tstate.model, X_valid, tstate.parameters, tstate.states))[1]
		valid_loss = mean((y_valid .- valid_pred) .^ 2)
		push!(valid_losses, valid_loss)

        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Train Loss: %.5g \t Valid Loss: %.5g \n" epoch train_loss valid_loss
		end

		# Custom early-stopping
		if early_stopping == true
			if epoch > 150 && train_losses[end] < 0.10
				if valid_losses[end] > valid_losses[end-10]
					@printf "Early Stopping at Epoch: %3d \t" epoch
					return tstate, train_losses, valid_losses
				end
	        end
		end
    end
	
    return tstate, train_losses, valid_losses
end

# ╔═╡ b7d6736b-776e-49d1-ae18-4a64b07a1a24
begin
	if model_type == "NN"
		model = FNN(τ₀, τ₁, 1; depth=NN_depth, act=act, outer_act=exp)
	else
		model = LSTM(τ₀, τ₁, 1; depth=NN_depth, cell=cell)
	end
	
	ps, st = Lux.setup(Xoshiro(12345), model)

	tstate = Training.TrainState(model, ps, st, opt)
	
	ad_rule = AutoZygote()

	n_epochs = 750 # Max epochs
	@time tstate, train_losses, valid_losses = main(tstate, ad_rule, (X_train, y_train), n_epochs; early_stopping=true)
end

# ╔═╡ 096a8a27-ddad-4534-94e4-71f755d5f561
model

# ╔═╡ bb6602f4-927e-473e-b8da-957395ed7617
begin
	if model_type == "NN"
		plot(title="Losses: $(country)\n τ₁=$τ₁, act=$act, FNN, depth=$NN_depth", xlab="Epochs", ylab="MSE", ylim=(0.0, max(0.21, 1.5*minimum(valid_losses))))
	else
		plot(title="Losses: $(country)\n τ₀=$τ₀, τ₁=$τ₁, T=$T, cell=$cell, depth=$NN_depth", xlab="Epochs", ylab="MSE", ylim=(0.0, max(0.21, 1.5*minimum(valid_losses))))
	end
	scatter!(1:length(train_losses), train_losses, label="Training")
	scatter!(1:length(valid_losses), valid_losses, label="Validation")
	hline!([0.1], col=:black, label=false)
end

# ╔═╡ e006827b-2f8b-4b7a-9669-2fb64fadc129
function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

# ╔═╡ 3ed8e1da-6ac1-48fa-ab1d-8272025d5aad
list_of_params = vcat(ComponentArray(tstate.parameters)...)

# ╔═╡ 930cdb5c-f80e-4f1e-b745-fbc42e7a47d7
sampled_params = hcat([rand(Xoshiro(1), Normal(i, 0.05*std(list_of_params)), 1_000) for i ∈ list_of_params]...) |> f32

# ╔═╡ 42a12287-4469-465c-9626-20de6708ce38
ps_sample = sampled_params[1, :]

# ╔═╡ bbb80aa6-096c-4fae-bc1d-2e2f5ba28b2d
function predict(train_state, Xs)
	st_ = Lux.testmode(train_state.states)
	return -1 .* (Lux.apply(train_state.model, Xs, train_state.parameters, st_))[1]
end

# ╔═╡ 4fb91714-4412-49c3-b466-8ac3ee439047
function predict(train_state, Xs, params)
	st_ = Lux.testmode(train_state.states)
	return -1 .* (Lux.apply(train_state.model, Xs, vector_to_parameters(params, train_state.parameters), st_))[1]
end

# ╔═╡ 7c0f7137-a868-41a9-831d-6298189d27e4
function predict(train_state, Xs, time_step, x_1, x_2, standariser)
	X_ = deepcopy(Xs)
	pred_full = zeros(time_step, 99-τ₀+1)
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

# ╔═╡ 2fa273ec-86e1-4e2c-a589-e79a09e0089d
function predict(train_state, Xs, params, time_step, x_1, x_2, standariser)
	X_ = deepcopy(Xs)
	pred_full = zeros(time_step, 99-τ₀+1)
	for i ∈ 1:time_step
		pred = predict(train_state, X_, params)
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

# ╔═╡ c0047459-5f14-4af7-b541-8238763d4a70
y_pred_valid = predict(tstate, X_valid)

# ╔═╡ f1f84de0-488a-4bad-a2a4-64965d493dc7
y_pred_train = predict(tstate, X_train)

# ╔═╡ 7e8e62f5-28a1-4153-892a-fc8988130f4b
mean((exp.(-y_valid) .- exp.(y_pred_valid)) .^ 2)

# ╔═╡ c8bca677-24d5-4bcc-b881-e0f43f208ca9
mean((exp.(-y_train) .- exp.(y_pred_train)) .^ 2)

# ╔═╡ c59ed9df-f944-4ee6-881e-2986dc8b1d3d
begin
	if model_type == "NN"
		y_pred_test = predict(tstate, X_test)
		start_age = 0
		end_age = 98
		
		plot(title="Validation Set ($end_year): $(country)\n τ₁=$τ₁, act=$act, FNN, depth=$NN_depth", xlab="Age", ylab="log μ", legend=:bottomright, ylim=(-12.0, 0.0))
		
		#for i ∈ 1:1_000
		#	sample_pred = predict(tstate, X_test, sampled_params[i, :])
		#	plot!(start_age:end_age, vec(sample_pred), label="", width=0.05, alpha=0.5, color=:gray)
		#end
		plot!(start_age:end_age, vec(y_pred_test), label="Predicted", width=2, color=:blue)
		scatter!(X_valid_ages, vec(-y_valid'), label="Observed", color=:orange)
	else
		start_age = τ₀ - 2
		end_age = 97
		
		plot(title="Validation Set ($end_year): $(country)\n τ₀=$τ₀, τ₁=$τ₁, T=$T, cell=$cell, depth=$NN_depth", xlab="Age", ylab="log μ", legend=:bottomright, ylim=(-12.0, 0.0))
		
		#for i ∈ 1:1_000
		#	sample_pred = predict(tstate, X_valid, sampled_params[i, :])
		#	plot!(start_age:end_age, vec(sample_pred), label="", width=0.05, alpha=0.5, color=:gray)
		#end
		plot!(start_age:end_age, vec(y_pred_valid), label="Predicted", width=2, color=:blue)
		scatter!(start_age:end_age, vec(-y_valid'), label="Observed", color=:orange)
	end
end

# ╔═╡ f099a171-e1ba-4d74-87a3-010e0e9ff27a
if model_type ≠ "NN"
	forecast = predict(tstate, X_valid, (extended_forecast_year-end_year+1), x_1, x_2, min_max)
end

# ╔═╡ d826f3b4-8a5f-4f99-88fb-d9b8420c6d89
begin
if model_type ≠ "NN"
	forecast_year_ = 2018
	plot(title="Forecast (Year $forecast_year_): $(country)\n τ₀=$τ₀, τ₁=$τ₁, T=$T, cell=$cell, depth=$NN_depth", xlab="Year", ylab="log μ", legend=:bottomright)
	#for i ∈ 1:250
	#	sample_forecast = predict(tstate, X_valid, sampled_params[i, :], (extended_forecast_year-end_year+1), x_1, x_2, min_max)
	#	plot!(start_age:end_age, vec(sample_forecast[forecast_year_-end_year+1, :]'), label="", width=0.05, alpha=0.5, color=:gray)
	#end
	plot!(start_age:end_age, vec(forecast[forecast_year_-end_year+1, :]'), label="Forecast", color=:blue, width=2)
	scatter!(y_test[y_test.Year .== forecast_year_, :Age], y_test[y_test.Year .== forecast_year_, :Female], label="Actual", color=:orange)
end
end

# ╔═╡ c1b667d2-1411-4637-9309-967309cc30e6
begin
if model_type ≠ "NN"
		forecast_age = 65
		adj_forecast_age = forecast_age + Int((3 - τ₀)/2)
		@assert forecast_age ≥ (τ₀ + 1)/2 - 1
		
		plot(title="Forecast (Age $forecast_age): $(country)\n τ₀=$τ₀, τ₁=$τ₁, T=$T, cell=$cell, depth=$NN_depth", xlab="Year", ylab="log μ", legend=:topright)
		#for i ∈ 1:100
		#	sample_forecast = predict(tstate, X_valid, sampled_params[i, :], (extended_forecast_year-end_year+1), x_1, x_2, min_max)
		#	plot!(end_year:extended_forecast_year, sample_forecast[:, forecast_age+1], label="", width=0.05, alpha=0.5, color=:gray)
		#end
		plot!(end_year:extended_forecast_year, forecast[:, adj_forecast_age], label="Forecast", color=:blue, width=2)
		scatter!(y_test[y_test.Age .== forecast_age, :Year], y_test[y_test.Age .== forecast_age, :Female], label="Actual", color=:orange)
	end
end

# ╔═╡ Cell order:
# ╠═6264e41e-e179-11ef-19c1-f135e97db7cc
# ╠═ef51c95e-d5ad-455a-9631-094823b695bb
# ╠═50b3b576-d941-4609-8469-6de51cfa1545
# ╠═a0132c58-f427-4c88-ba45-bd8b9d9f98d4
# ╠═5378ee86-303b-4269-88f3-6eeccc30cb15
# ╠═93779239-cd66-4be2-b70f-c7872a29a29f
# ╠═608363b0-5f75-45c8-8970-5170489a5eeb
# ╠═59eed207-e138-44be-a241-d4cbfde0f38c
# ╠═7b5558e7-30c8-4069-9175-6dd79d27c8f5
# ╠═564d77df-85c1-406a-8964-3b14fca6328c
# ╠═b7d6736b-776e-49d1-ae18-4a64b07a1a24
# ╠═096a8a27-ddad-4534-94e4-71f755d5f561
# ╠═bb6602f4-927e-473e-b8da-957395ed7617
# ╠═e006827b-2f8b-4b7a-9669-2fb64fadc129
# ╠═3ed8e1da-6ac1-48fa-ab1d-8272025d5aad
# ╠═42a12287-4469-465c-9626-20de6708ce38
# ╠═930cdb5c-f80e-4f1e-b745-fbc42e7a47d7
# ╠═bbb80aa6-096c-4fae-bc1d-2e2f5ba28b2d
# ╠═4fb91714-4412-49c3-b466-8ac3ee439047
# ╠═7c0f7137-a868-41a9-831d-6298189d27e4
# ╠═2fa273ec-86e1-4e2c-a589-e79a09e0089d
# ╠═c0047459-5f14-4af7-b541-8238763d4a70
# ╠═f1f84de0-488a-4bad-a2a4-64965d493dc7
# ╠═7e8e62f5-28a1-4153-892a-fc8988130f4b
# ╠═c8bca677-24d5-4bcc-b881-e0f43f208ca9
# ╠═c59ed9df-f944-4ee6-881e-2986dc8b1d3d
# ╠═f099a171-e1ba-4d74-87a3-010e0e9ff27a
# ╠═d826f3b4-8a5f-4f99-88fb-d9b8420c6d89
# ╠═c1b667d2-1411-4637-9309-967309cc30e6
