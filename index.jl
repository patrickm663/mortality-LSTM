### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 6264e41e-e179-11ef-19c1-f135e97db7cc
begin
	cd(".")
	using Pkg
	# Not registered
	Pkg.add(url="https://github.com/patrickm663/HMD.jl")
	Pkg.instantiate()
	Pkg.add(["CSV", "DataFrames", "Lux", "ADTypes", "Zygote", "Optimisers", "Plots", "ComponentArrays", "MLUtils"])
end

# ╔═╡ ef51c95e-d5ad-455a-9631-094823b695bb
using ADTypes, Lux, Optimisers, Printf, Random, CSV, Plots, DataFrames, ComponentArrays, MLUtils, HMD, Zygote, Statistics

# ╔═╡ 93779239-cd66-4be2-b70f-c7872a29a29f
function LSTM1(in_dims, hidden_dims, out_dims)
    #lstm_cell = LSTMCell(in_dims => hidden_dims)
    #regressor = Dense(hidden_dims => out_dims, exp)
    #return @compact(; lstm_cell, regressor) do x::AbstractArray{T, 3} where {T}
    #    x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
    #    y, carry = lstm_cell(x_init)
    #    for x in x_rest
    #        y, carry = lstm_cell((x, carry))
    #    end
    #    @return vec(regressor(y))
    #end
	return Chain(
		Recurrence(
			GRUCell(in_dims => hidden_dims);
			return_sequence=false
			),
		Dense(hidden_dims => out_dims, exp)
		)
end

# ╔═╡ fea5ab5a-847b-4c16-a318-2993b6f3662f
LSTM1(3, 5, 1)

# ╔═╡ 6e4c5b3e-5daf-47db-b0ef-df1a69d249be
rng = Xoshiro(222)

# ╔═╡ e250694d-4d5b-405d-846b-033088335737
md"""
```
library ( keras )
T
<- 10
tau0 <- 3
tau1 <- 5
# length of time series x_1 ,... , x_T
# dimension inputs x_t
# dimension of the neurons z_t ^(1) in the first RNN layer
Input <- layer_input ( shape = c (T , tau0 ) , dtype = ’ float32 ’ , name = ’ Input ’)
Output = Input % >%
layer_lstm ( units = tau1 , activation = ’ tanh ’ , r e c u r r e n t _ a c t i v a t i o n = ’ tanh ’ , name = ’ LSTM1 ’) % >%
layer_dense ( units =1 , activation = k_exp , name =" Output ")
model <- keras_model ( inputs = list ( Input ) , outputs = c ( Output ))
model % >% compile ( loss = ’ mean_squared_error ’ , optimizer = ’ nadam ’)
summary ( model )
```
"""

# ╔═╡ 9a0a2da5-53ed-46c2-8eb0-131d3a11f2c2
md"""
Each rate is neighboured by the previous and sequential rate (age-wise). The target is the mortality rate per age in the year 2 000.
"""

# ╔═╡ a5e6a3ca-9c2b-4064-8e9e-0ea2e91d7aed
md"""
```R
mort_rates <- all_mort [ which ( al l _m or t$ G en de r ==" Female ") , c (" Year " , " Age " , " logmx ")]
mort_rates <- dcast ( mort_rates , Year ~ Age , value . var =" logmx ")
T0 <- 10
tau0 <- 3
# lookback period
# dimension of x_t ( should be odd for our application )
toy_rates <- as . matrix ( mort_rates [ which ( m or t_ r at es $ Ye ar % in % c (2001 - T0 -1):2001)) ,])
# note that the first column in toy_rates is the " Year "
xt <- array ( NA , c (2 , ncol ( toy_rates ) - tau0 , T0 , tau0 ))
YT <- array ( NA , c (2 , ncol ( toy_rates ) - tau0 ))
for ( i in 1:2){ for ( a0 in 1:( ncol ( toy_rates ) - tau0 )){
xt [i , a0 , ,] <- toy_rates [ c ( i :( T0 +i -1)) , c (( a0 +1):( a0 + tau0 ))]
YT [i , a0 ] <- toy_rates [ T0 +i , a0 +1+( tau0 -1)/2]
}}
```
"""

# ╔═╡ 59eed207-e138-44be-a241-d4cbfde0f38c
function get_data(; T=10, τ₀=3)
	# Apply 'Toy example' data pre-processing
	
    # Load from CSV
    data = CSV.read("data/CHE_Mx_1x1.csv", DataFrame)

	# Split out females, aged 0-98, years 1990-2001
	data = data[(data.Age .≤ 98) .&& (1990 .≤ data.Year .≤ 2001), [:Year, :Age, :Female]]

	# Group age x year and drop the age column (1)
	data = HMD.transform(data, :Female)[:, 2:end]
	# Add for numerical stability
	data .= data .+ 1e-9
	
	 # Convert to Matrix, transpose, take logs, and add extra column to match R code
	data = hcat(1990:2001, log.(Matrix(data)'))

	# 1 = train, 2 = validation
	X_ = zeros(2, τ₀, T, size(data)[2]-τ₀) # 3, 10, :
	y_ = zeros(2, size(data)[2]-τ₀)
	for i in 1:2
		for j in 1:(size(data)[2]-τ₀)
			X_[i, :, :, j] = data[i:(T+i-1), ((j+1):(j+τ₀))]'
			y_[i, j] = data[T+i, Int(j+1+(τ₀ - 1)/2)]'
		end
	end

	# Min-max scale inputs to -1 - 1
	
	X_min, X_max = (minimum(X_), maximum(X_))

	min_max(x, xmin, xmax) = 2*(x - xmin) / (xmax - xmin) - 1
	X_train = min_max.(X_[1, :, :, :], X_min, X_max) |> f32
	X_valid = min_max.(X_[2, :, :, :], X_min, X_max) |> f32

	# Negative log mortality so outputs are positive
	y_train = -y_[1, :] |> f32
	y_valid = -y_[2, :] |> f32

	return X_train, reshape(y_train, 1, :), X_valid, reshape(y_valid, 1, :), (X_min, X_max)
end

# ╔═╡ 7b5558e7-30c8-4069-9175-6dd79d27c8f5
X_train, y_train, X_valid, y_valid, min_max_scale = get_data()

# ╔═╡ 564d77df-85c1-406a-8964-3b14fca6328c
function main(tstate::Training.TrainState, vjp, data, epochs)
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
    end
    return tstate, train_losses, valid_losses
end

# ╔═╡ b7d6736b-776e-49d1-ae18-4a64b07a1a24
begin
	model = LSTM1(3, 5, 1)
	ps, st = Lux.setup(rng, model)

	opt = Adam(0.01f0)
	tstate = Training.TrainState(model, ps, st, opt)
	
	ad_rule = AutoZygote()

	n_epochs = 1_000
	@time tstate, train_losses, valid_losses = main(tstate, ad_rule, (X_train, y_train), n_epochs)
end

# ╔═╡ bb6602f4-927e-473e-b8da-957395ed7617
begin
plot(10:n_epochs, train_losses[10:end], xlab="Epochs", ylab="MSE", ylim=(0.0, 0.30), label="Training")
plot!(10:n_epochs, valid_losses[10:end], label="Validation")
end

# ╔═╡ bbb80aa6-096c-4fae-bc1d-2e2f5ba28b2d
function predict(train_state, Xs)
	st_ = Lux.testmode(train_state.states)
	return -1 .* (Lux.apply(train_state.model, Xs, train_state.parameters, st_))[1]
end

# ╔═╡ 828445ab-f12c-4702-a008-9cf7091c084a
y_pred_valid = predict(tstate, X_valid)

# ╔═╡ f1f84de0-488a-4bad-a2a4-64965d493dc7
y_pred_train = predict(tstate, X_train)

# ╔═╡ 7e8e62f5-28a1-4153-892a-fc8988130f4b
mean((exp.(-y_valid) .- exp.(y_pred_valid)) .^ 2)

# ╔═╡ c8bca677-24d5-4bcc-b881-e0f43f208ca9
mean((exp.(-y_train) .- exp.(y_pred_train)) .^ 2)

# ╔═╡ c59ed9df-f944-4ee6-881e-2986dc8b1d3d
begin
	plot(1:97, vec(y_pred_valid), label="Predicted", width=2, title="Validation Set")
	scatter!(1:97, -vec(y_valid'), label="Observed")
end

# ╔═╡ 13601f30-29d5-40f3-a8c2-18b8a25a4070
begin
	plot(1:97, vec(y_pred_train), label="Predicted", width=2, title="Training Set")
	scatter!(1:97, -vec(y_train'), label="Observed")
end

# ╔═╡ 78133d49-6e4d-4506-9e0b-81cd058048c6


# ╔═╡ 7844b633-e043-489a-9627-3f87e13d296d


# ╔═╡ Cell order:
# ╠═6264e41e-e179-11ef-19c1-f135e97db7cc
# ╠═ef51c95e-d5ad-455a-9631-094823b695bb
# ╠═fea5ab5a-847b-4c16-a318-2993b6f3662f
# ╠═93779239-cd66-4be2-b70f-c7872a29a29f
# ╠═6e4c5b3e-5daf-47db-b0ef-df1a69d249be
# ╟─e250694d-4d5b-405d-846b-033088335737
# ╟─9a0a2da5-53ed-46c2-8eb0-131d3a11f2c2
# ╟─a5e6a3ca-9c2b-4064-8e9e-0ea2e91d7aed
# ╠═59eed207-e138-44be-a241-d4cbfde0f38c
# ╠═7b5558e7-30c8-4069-9175-6dd79d27c8f5
# ╠═564d77df-85c1-406a-8964-3b14fca6328c
# ╠═b7d6736b-776e-49d1-ae18-4a64b07a1a24
# ╠═bb6602f4-927e-473e-b8da-957395ed7617
# ╠═bbb80aa6-096c-4fae-bc1d-2e2f5ba28b2d
# ╠═828445ab-f12c-4702-a008-9cf7091c084a
# ╠═f1f84de0-488a-4bad-a2a4-64965d493dc7
# ╠═7e8e62f5-28a1-4153-892a-fc8988130f4b
# ╠═c8bca677-24d5-4bcc-b881-e0f43f208ca9
# ╠═c59ed9df-f944-4ee6-881e-2986dc8b1d3d
# ╠═13601f30-29d5-40f3-a8c2-18b8a25a4070
# ╠═78133d49-6e4d-4506-9e0b-81cd058048c6
# ╠═7844b633-e043-489a-9627-3f87e13d296d
