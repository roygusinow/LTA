# let's do a simulation of an HMM first!
using DataFrames, Distributions, Random, LinearAlgebra

# include("../structures/helping_structures.jl")

function gen_covariate_data(
	N::Int64,
	k::Int64;
	constant = false,
	seed = 1,
)
	# create covariate data for simulation

	Random.seed!(seed)

	# just make the dat for simulation
	patient_mat = Matrix{Union{Missing, Float64}}(undef, N, k)
	patient_mat[:, 1] .= 1
	for i in 2:k
		patient_mat[:, i] .= rand(MersenneTwister(seed + i), Bernoulli(0.1*i/3), N)
	end

	if constant
		patient_mat = repeat(patient_mat[1, :]', N, 1)
	end

	patient_df = DataFrame(patient_mat, :auto)

	return patient_df
end

# get (beta) model params based on seed
function gen_beta_initial(;
	seed = 1,
	n_states = n_states, k = k,
	ref_state = 1,
)

	beta_initial = rand(MersenneTwister(seed), Uniform(-2, 2), n_states, k)
	beta_initial[ref_state, :] .= 0

	return beta_initial
end

function gen_beta_transition(;
	seed = 1,
	n_states = n_states, k = k,
	ref_state = 1,
)

	beta_transition = rand(
		MersenneTwister(seed),
		Uniform(-2, 2),
		(n_states, k, n_states))
	beta_transition[ref_state, :, :] .= 0

	return beta_transition
end

function gen_beta_bernoulli_emission(; seed = 1, n_states = n_states, k = k, n_sympt = n_sympt)
	beta_emission = rand(MersenneTwister(seed + 2), Uniform(-2, 2), n_states, k, n_sympt)
	return beta_emission
end

function gen_model_params(; seed, n_states, covariate_tup, n_obs_tup, ref_state)

	# generate model params for simulation

	rho_initial = gen_rho_weight_param(seed + 1, length(covariate_tup[:initial]))
	rho_trans = gen_rho_weight_param(seed + 2, length(covariate_tup[:trans]))

	# handle the case where there are no covariates
	if length(covariate_tup[:initial]) == 0
		k_size_initial = 1
	else
		k_size_initial = 2
	end

	if length(covariate_tup[:trans]) == 0
		k_size_trans = 1
	else
		k_size_trans = 2
	end

	beta_initial = gen_beta_initial(;
		seed = seed + 3,
		n_states = n_states, k = k_size_initial,
		ref_state = ref_state)
	beta_transition = gen_beta_transition(;
		seed = seed + 4,
		n_states = n_states, k = k_size_trans,
		ref_state = ref_state)

	model_params = (
		beta_initial = beta_initial,
		beta_transition = beta_transition,
		rho_initial = rho_initial,
		rho_trans = rho_trans
	)

	# handle different emmissions
	klem = length(covariate_tup[:em])
	@assert klem == 0 "Currently no covariates supported for emissions!";
	emissions = NamedTuple()
	for dist in keys(n_obs_tup)
		if dist == :bernoulli
			nm = n_obs_tup[:bernoulli]
			@assert nm > 0 "No bernoulli symptoms specified!";
			beta_bernoulli = gen_beta_bernoulli_emission(; 
				seed = seed + 5, n_states = n_states, k = 1, n_sympt = nm)
			emissions = (; emissions..., beta_bernoulli = beta_bernoulli)
		elseif dist == :gaussian	
			nm = n_obs_tup[:gaussian]
			@assert nm > 0 "No gaussian symptoms specified!";
			beta_gaussian = gen_beta_gaussion_emission(; 
				seed = seed + 6, n_states = n_states, n_obs_cont = nm)
			emissions = (; emissions..., beta_gaussian = beta_gaussian)
		elseif dist == :ordinal	
			nm = length(n_obs_tup[:ordinal])
			@assert nm > 0 "No ordinal symptoms specified!";
			# currently stable n_levels
			beta_ordinal= gen_beta_ordinal_emission(; 
				seed = seed + 6, n_states = n_states, n_levels = n_obs_tup[:ordinal])
			emissions = (; emissions..., beta_ordinal = beta_ordinal)
		end
	end
	model_params = (; model_params..., emissions = emissions)

	return model_params
end


# get model params
function gen_emission_prob(
	seed::Int64 = 2,
	dims::Vector{Int64} = [7, 9])

	# get the emision
	prob_mat = Matrix{Float64}(undef, dims[1], dims[2])
	for row in 1:dims[1]
		prob_mat[row, :] = rand(MersenneTwister(seed + row), Dirichlet(ones(dims[2]) / dims[2]))
	end

	return (prob_mat)
end

function get_emission_mat(
	beta::Array{<:Real},
	x_vec::Vector{<:Union{Missing, Float64}},
)

	# get complete emission matrix for a given sample
	out_mat = zeros(eltype(beta), size(beta, 1), size(beta, 3))
	for sample_idx in 1:length(x_vec)
		out_mat += beta[:, sample_idx, :] .* x_vec[sample_idx]
	end
	sig = 1 ./ (1 .+ exp.(.-out_mat))

	return sig

end

function get_bernoulli_probs(
	beta::Array{<:Real},
	x_vec::Vector{<:Union{Missing, Float64}},
)
	# get complete emission matrix for a given sample, adding the intercept
	if length(x_vec) == 0
		intercept_vec = [1.0]
	else
		intercept_vec = [1.0, x_vec]
	end

	out = get_emission_mat(
		beta,
		intercept_vec
	)

	return out
end

function gen_beta_gaussion_emission(; seed::Int64 = 2, n_states = 3, n_obs_cont = 2)
	# generate a random emission matrix for the continuous outcomes

	true_gaussian_emission = (
		means = rand(MersenneTwister(seed), Uniform(0.2, 0.8), (n_states, n_obs_cont)), # assign mean
		stds = rand(MersenneTwister(seed + 1), Uniform(0.05, 0.1), (n_states, n_obs_cont)) # assign standard deviation
	)
	beta_gaussian_emission = convert_gauss_2_form(true_gaussian_emission) # convert to beta form

	return beta_gaussian_emission
end

function convert_means_2_form(means::Array{<:Real})
	# convert means to the form of the beta matrix
	out = log.(means ./ (1 .- means))
	
	return out
end

function convert_std_2_form(stds::Array{<:Real})
	out = log.(stds)
	return out
end

function convert_means_2_true(means::Array{<:Real})
	out = 1 ./ (1 .+ exp.(-means))
	return out
end

function convert_std_2_true(stds::Array{<:Real})
	out = exp.(stds)
	return out
end

function convert_gauss_2_true(x::NamedTuple)
	# convert gaussian means and stds back to their non-parameterised forms
	out = (
		means = convert_means_2_true(x.means),
		stds = convert_std_2_true(x.stds)
	)
	return out
end

function convert_gauss_2_form(x::NamedTuple)
	
	# convert gaussian means and stds to the form of the beta matrix
	out = (
		means = convert_means_2_form(x.means),
		stds = convert_std_2_form(x.stds)
	)

	return out
end

function softplus(x)
	return log(1 + exp(x))
end

function convert_ordinal_spacings_2_intercepts(x::Vector{<:Real})

	ordinal_intercepts = Vector{eltype(x)}(undef, length(x))

	ordinal_intercepts[1] = x[1] # use the first value as base
	for i in 2:length(x)
		ordinal_intercepts[i] = ordinal_intercepts[i - 1] + softplus(x[i])
	end
	return ordinal_intercepts
end

function convert_ordinal_spacings_2_probs(x::Vector{<:Real})

	# first, convert spacings to intercepts to ensure positivity and ordering
	# print("hit")
	ordinal_intercepts = convert_ordinal_spacings_2_intercepts(x)
	# ordinal_intercepts = cumsum(x)
	# ordinal_intercepts = x
	
	# convert ordinal intercepts to probabilities
	out = Vector{eltype(ordinal_intercepts)}(undef, length(ordinal_intercepts) + 1)
	out[1] = 1 / (1 + exp(-ordinal_intercepts[1]))
	for i in 1:length(ordinal_intercepts)-1
		out[i + 1] = (1 / (1 + exp(-ordinal_intercepts[i + 1]))) - (1 / (1 + exp(-ordinal_intercepts[i])))
	end
	out[end] = 1 - (1 / (1 + exp(-ordinal_intercepts[end])))

	return out
end

function convert_ordinal_2_true(x::NamedTuple)

	# convert ordinal intercepts to probabilities of every state
	true_ordinal_probs = NamedTuple()
	for i in 1:length(x)

		# could be better formulated...
		true_ordinal_probs = (; true_ordinal_probs..., 
			keys(x)[i] => collect(transpose(hcat(
				[convert_ordinal_spacings_2_probs(x[i][state, :]) for state in 1:size(x[i], 1)]...))))
	end

	return true_ordinal_probs
end

function gen_beta_ordinal_emission(; 
	seed, n_states, n_levels::Vector{Int64})

	beta_ordinal = NamedTuple()
	for i in 1:length(n_levels)
		beta_ordinal = (; beta_ordinal..., 
			Symbol("symptom_" * string(i)) => rand(MersenneTwister(seed + i), Uniform(-2, 2), (n_states, n_levels[i] - 1)))
	end

	return beta_ordinal
end

function multinom_reg(
	beta_mat::Matrix{<:Real},
	# x_vec::Vector{<:Union{Missing, Float64}},
	x_vec::Vector{<:Real}
)

	# x_vec_reg = x_vec[1:size(beta_mat, 2)]
	x_vec_reg = x_vec
	# get multinomial probs from beta coefs and a example vector    
	exp_arg = exp.(beta_mat * x_vec_reg)
	# probs_vec = exp_arg ./ (1 + sum(exp_arg[beta_mat[:, 1].!=0])) # sum only non references
	probs_vec = exp_arg ./ (1 + sum(exp_arg[2:end])) # sum only non references -- assuming its at 1st ref

	return probs_vec
end

function get_transition_mat(
	beta::Array{<:Real},
	# x_vec::Vector{<:Union{Missing, Float64}},
	x_vec::Vector{<:Real}
)
	# get complete transition matrix for a given sample
	transition_mat = [multinom_reg(beta[:, :, b_mat_idx], x_vec) for b_mat_idx in 1:size(beta, 3)]

	# return transition_mat
	return (reduce(vcat, transition_mat'))
end

# drawing
function sample_single_bernoulli(
	state::Int64,
	emission_mat::Matrix{Float64}
)
	# draw observations from state
	si = size(emission_mat, 2)
	obs_vec = Vector{Float64}(undef, si)

	for symptom_ind in 1:si
		bern_dist = Bernoulli(emission_mat[state, symptom_ind]) # random bernoulli draw!
		obs_vec[symptom_ind] = rand(bern_dist)
	end

	return Int.(obs_vec)
end

function sample_bernoulli(
	states::Vector{Int64}; 
	bernoulli_probability_mat::Matrix{Float64}
	)

	T = length(states)
	bernoulli_observations = Matrix{Int64}(undef, T, size(bernoulli_probability_mat, 2))
	for time in 1:T
		bernoulli_observations[time, :] = sample_single_bernoulli(states[time], bernoulli_probability_mat)
	end
	
	return bernoulli_observations
end

function sample_single_gaussian(
	state::Int64;
	means::Array{Float64},
	stds::Array{Float64}
	)
	# draw observations from state
	si = size(means, 2)
	obs_vec = Vector{Float64}(undef, si)

	# standard_dev = 0.1
	for symptom_ind in 1:si
		norm_dist = Normal(
			means[state, symptom_ind], 
			stds[state, symptom_ind]
		) # random norm draw!
		obs_vec[symptom_ind] = rand(norm_dist)
	end

	return obs_vec

end

function sample_gaussian(
	states::Vector{Int64}; 
	true_gaussian_emission::NamedTuple
	)

	T = length(states)
	gaussian_observations = Matrix{Float64}(undef, T, size(true_gaussian_emission.means, 2))
	for time in 1:T
		gaussian_observations[time, :] = sample_single_gaussian(states[time], means = true_gaussian_emission.means, stds = true_gaussian_emission.stds)
	end
	
	return gaussian_observations
end

function sample_single_ordinal(
	state::Int64;
	true_ordinal_probs::NamedTuple
	)

	# draw observations from state
	si = length(true_ordinal_probs)
	obs_vec = Vector{Int64}(undef, si)

	# standard_dev = 0.1
	for symptom_ind in 1:si
		prob_vec = true_ordinal_probs[symptom_ind][state, :]
		obs_vec[symptom_ind] = rand(Distributions.Categorical(prob_vec))
	end

	return obs_vec
end

function sample_ordinal(
	states::Vector{Int64}; 
	true_ordinal_probs::NamedTuple)

	T = length(states)
	ordinal_observations = Matrix{Int64}(undef, T, length(true_ordinal_probs))
	for time in 1:T
		ordinal_observations[time, :] = sample_single_ordinal(states[time]; true_ordinal_probs = true_ordinal_probs)
	end

	return ordinal_observations
end

function draw_cat_prob(prob_vec::Vector{Float64})
    # Check if exactly one entry is NaN and all others are zero
    if count(isnan, prob_vec) == 1 && all(iszero, prob_vec[.!isnan.(prob_vec)])
        # Identify the index of the NaN
        idx = findfirst(isnan, prob_vec)
        # Replace that single NaN with 1.0
        prob_vec .= 0.0
        prob_vec[idx] = 1.0
    end

    return rand(Distributions.Categorical(prob_vec))
end

# simulation
function sampleHMM_states(
	initial_vec::Vector{Float64},
	transition_mat::Matrix{Float64},
	T::Int64)

	#Initialize states and observations
	state = Vector{Int64}(undef, T)

	#Sample initial s from initial distribution
	state[1] = draw_cat_prob(initial_vec)

	#Loop over Time Index
	for time in 2:T
		state[time] = draw_cat_prob(transition_mat[state[time-1], :])
	end

	return state
end

function simulate_HMM_sample(;
	model_params::NamedTuple,
	covariate_vec::Vector{<:Union{Missing, Float64}},
	covariate_idx::NamedTuple,
	T::Int64
)
	# simulate a single sample of HMM

	# simulate state trajectory first
	initial_probs = get_rho_beta_initial_states(model_params.beta_initial, model_params.rho_initial, covariate_vec[covariate_idx[:initial]])
	transition_probs = get_rho_beta_transition_mat(model_params.beta_transition, model_params.rho_trans, covariate_vec[covariate_idx[:trans]])
	state = sampleHMM_states(
		initial_probs,
		transition_probs,
		T)

	# then sample observations
	observations = NamedTuple()
	for dist in keys(model_params.emissions)
		if dist == :beta_bernoulli
			bernoulli_probability_mat = get_bernoulli_probs(model_params.emissions.beta_bernoulli, covariate_vec[covariate_idx[:em]])
			bernoulli_observations = sample_bernoulli(state; bernoulli_probability_mat = bernoulli_probability_mat)
			observations = (; observations..., bernoulli_observations = bernoulli_observations)
		elseif dist == :beta_gaussian	
			true_gaussian_emission = convert_gauss_2_true(model_params.emissions.beta_gaussian)
			gaussian_observations = sample_gaussian(state; true_gaussian_emission = true_gaussian_emission)
			observations = (; observations..., gaussian_observations = gaussian_observations)
		elseif dist == :beta_ordinal
			true_ordinal_probs = convert_ordinal_2_true(model_params.emissions.beta_ordinal)
			ordinal_observations = sample_ordinal(state; true_ordinal_probs = true_ordinal_probs)
			observations = (; observations..., ordinal_observations = ordinal_observations)
		end
	end

	return state, observations
end

function convert_covariate_2_df_indices(
	df::DataFrame,
	covariate_tup::NamedTuple
)
	df_headers = names(df)
	covariate_idx = convert_covariate_2_df_indices(
		df_headers,
		covariate_tup
	)
	
	return covariate_idx
end

function convert_covariate_2_df_indices(
	df_headers::Vector{String},
	covariate_tup::NamedTuple;
)

	# get the indices of the covariates in the dataframe

	covariate_idx = (
		initial = length(covariate_tup[:initial]) != 0 ? sum(hcat([df_headers .== var_name for var_name in covariate_tup[:initial]]), dims = 1)[1] .== 1 : [],
		trans = length(covariate_tup[:trans]) != 0 ? sum(hcat([df_headers .== var_name for var_name in covariate_tup[:trans]]), dims = 1)[1] .== 1 : [],
		em = length(covariate_tup[:em]) != 0 ? sum(hcat([df_headers .== var_name for var_name in covariate_tup[:em]]), dims = 1)[1] .== 1 : [],
		)
	
	return covariate_idx
end

function simulate(;
	model_params::NamedTuple,
	covariate_df::DataFrame,
	covariate_tup::NamedTuple,
	T::Int64
)

	covariate_mat = Matrix(covariate_df)
	covariate_idx = convert_covariate_2_df_indices(
		covariate_df,
		covariate_tup
	)
	N = size(covariate_mat, 1)

	states = Matrix{Float64}(undef, N, T)

	# observations
	bernoulli_observations = Array{Int64}(undef, N, T, 0)
	gaussian_observations = Array{Float64}(undef, N, T, 0)
	ordinal_observations = Array{Int64}(undef, N, T, 0)
	for dist in keys(model_params.emissions)
		if dist == :beta_bernoulli
			bernoulli_observations = Array{Int64}(undef, N, T, size(model_params.emissions.beta_bernoulli, 3))
		elseif dist == :beta_gaussian
			gaussian_observations = Array{Float64}(undef, N, T, size(model_params.emissions.beta_gaussian.means, 2))
		elseif dist == :beta_ordinal
			ordinal_observations = Array{Int64}(undef, N, T, length(model_params.emissions.beta_ordinal))
		end
	end

	# simulate each sample
	for i in 1:N
		state, observation = simulate_HMM_sample(;
			model_params = model_params,
			covariate_vec = covariate_mat[i, :],
			covariate_idx = covariate_idx,
			T = T
		)
		states[i, :] = state

		for dist in keys(model_params.emissions)
			if dist == :beta_bernoulli
				bernoulli_observations[i, :, :] = observation.bernoulli_observations
			elseif dist == :beta_gaussian
				gaussian_observations[i, :, :] = observation.gaussian_observations
			elseif dist == :beta_ordinal
				ordinal_observations[i, :, :] = observation.ordinal_observations
			end
		end
	end

	# saving appropirately
	observations = NamedTuple()
	for dist in keys(model_params.emissions)
		if dist == :beta_bernoulli
			observations = (; observations..., bernoulli_observations = bernoulli_observations)
		elseif dist == :beta_gaussian
			observations = (; observations..., gaussian_observations = gaussian_observations)
		elseif dist == :beta_ordinal
			observations = (; observations..., ordinal_observations = ordinal_observations)
		end
	end

	return states, observations
end

function remove_observations(
	observation_df::Array{<:Real};
	percentage_missing = 0.05,
)

	erased_observation_df = deepcopy(observation_df)
	missing_mask = rand(
		MersenneTwister(2),
		Bernoulli(percentage_missing),
		size(observation_df))
	erased_observation_df[missing_mask] .= -1

	return erased_observation_df
end

function remove_patient_covars(
	patient_mat::Matrix{Float64};
	percentage_missing = 0.05,
	col_idx = 2
)

	erased_patient_mat = deepcopy(patient_mat)
	missing_mask = rand(
		MersenneTwister(2),
		Bernoulli(percentage_missing),
		size(patient_mat))
	possible_cols = collect(1:size(erased_patient_mat, 2))
	missing_mask[:, setdiff(possible_cols, col_idx)] .= 0
	
	erased_patient_mat[missing_mask] .= -1

	return erased_patient_mat
end

function create_constant_columns_matrix(num_rows::Int, num_cols::Int)
    # Create a matrix where each column is filled with its column index
    matrix = [col for row in 1:num_rows, col in 1:num_cols]
    return matrix
end


function gamma_2_beta_transition(
	gamma::Array{<:Real};
)
	# convert the gamma to the beta

	n_states_minus_1, k_trans, poly_power_plus_1 = size(gamma)
	n_states = n_states_minus_1 + 1
	beta_transition = zeros(eltype(gamma), n_states, k_trans, n_states)

	state_var_mat = create_constant_columns_matrix(poly_power_plus_1, n_states)

	for i in 1:n_states_minus_1
		beta_transition[i+1, :, :] .= gamma[i, :, :] * state_var_mat
	end
	
	return beta_transition
end

# collapsing covars
function gen_rho_weight_param(
	seed::Int64,
	k::Int64
)
	# generate rho
	return rand(MersenneTwister(seed), Uniform(-2.0, 2.0), k)
end

function get_aggr_covars(rho_vec, x_vec)
	return rho_vec' * x_vec # dot product
end

function get_rho_beta_transition_mat(
	beta_transition::Array{<:Real}, rho_vec::Vector{<:Real}, x_vec::Vector{<:Union{Missing, Float64}}
)
	# inherently uses intercept

	if length(rho_vec) == 0
		trans_mat = get_transition_mat(beta_transition, [1.0])
	else
		aggr_val = get_aggr_covars(rho_vec, x_vec)
		sig = aggr_val
	
		trans_mat = get_transition_mat(beta_transition, [1.0, sig])
	end

	return trans_mat
end

function get_rho_beta_initial_states(
	beta_initial::Array{<:Real}, rho_vec::Vector{<:Real}, x_vec::Vector{<:Union{Missing, Float64}}
)

	if length(rho_vec) == 0
		initial_states = multinom_reg(beta_initial, [1.0])
	else
		aggr_val = get_aggr_covars(rho_vec, x_vec)
		sig = aggr_val
	
		initial_states = multinom_reg(beta_initial, [1.0, sig])
	end

	return initial_states
end

function sigmoid(x)
	return 1 / (1 + exp(-x))
end

function gen_params_from_hyper_params(
	sim_hyper::SimHyperParams
)
	# take in the hyper params and generate a random collectioon paramters for simulation
	covariate_df = gen_covariate_data(
		sim_hyper.N,
		20, # shouldn't have more than 20 covars anyways...
		constant = false,
		seed = sim_hyper.param_seed)
	covariate_mat = Matrix{Float64}(covariate_df)

	# k_tup_idx = convert_k_tup_2_df_indices(patient_df, sim_hyper.k_tup)
	
	model_params = gen_model_params(;
		seed = sim_hyper.param_seed, 
		n_states = sim_hyper.n_states, 
		covariate_tup = sim_hyper.covariate_tup, 
		n_obs_tup = sim_hyper.n_obs_tup, 
		ref_state = sim_hyper.ref_state,
	)

	sim_params = SimParams(
		sim_hyper = sim_hyper,

		# obtained from simulation params
		covariate_mat_headers = names(covariate_df),
		covariate_mat = covariate_mat, 
		
		model_params = model_params
	)

	return sim_params
end


function run_simulation(
	sim_params::SimParams;
	simulation_seed = 1,
	T = 5,
	percentage_observations_missing = 0.0,
)

	# take in a sim_param struct and run the simulation
	Random.seed!(simulation_seed)	
	
	# simulate!
	covariate_df = DataFrame(sim_params.covariate_mat, sim_params.covariate_mat_headers)
	states, observations = simulate(;
		model_params = sim_params.model_params,
		covariate_df = covariate_df,
		covariate_tup = sim_params.sim_hyper.covariate_tup,
		T = T
	)

	# remove observations
	removed_observations = (;  
		(dist => remove_observations(observations[dist];
									percentage_missing = percentage_observations_missing)
		for dist in keys(observations))...  
	)

	# remove covars
	# patient_mat = remove_patient_covars(
	# 	sim_params.patient_mat;
	# 	percentage_missing = kwargs[:covar_missing_percentage],
	# 	col_idx = kwargs[:covar_missing_cols]
	# )
	# new_sim_params = @set sim_params.patient_mat = patient_mat

	# assign all vars to struct now
	sim_output = SimulationOutput(
		sim_params = sim_params,
		simulation_seed = simulation_seed, 
		
		states = states,
		observations = removed_observations
	)

	return sim_output
end

# saving functions
function save_simulation(
	sim_output::SimulationOutput,
)
	sim_dr = "data/simulations/" * sim_output.sim_params.sim_hyper.sim_no
	save(sim_dr * "/simulation/sim_output.jld", "sim_output", sim_output)

	print("Simulation Saved!")
end
