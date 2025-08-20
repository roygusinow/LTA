@kwdef struct SimHyperParams
	param_seed::Int64

	N::Int64
    n_obs_tup::NamedTuple
	n_states::Int64
    covariate_tup::NamedTuple

	ref_state::Int64

	sim_no::String
	comments::String
end

@kwdef struct SimParams
	sim_hyper::SimHyperParams

	# obtained from simulation params
	covariate_mat_headers::Vector{String}
	covariate_mat::Matrix{Float64}

	model_params::NamedTuple
end

@kwdef struct SimulationOutput
	sim_params::SimParams
	simulation_seed::Int64

	# simulation results
	states::Matrix{Float64}
	observations::NamedTuple
end

@kwdef struct EstimationOutput
	
    sim_output::SimulationOutput
	est_seed::Int64
    lambda::Float64

	true_log_lkl::Float64 

	starting_model_params::Vector{NamedTuple}
	fitted_model_params::Vector{NamedTuple}

	fitted_log_lkl_list::Vector{Float64}

	iterations::Vector{Int64}
	iteration_limit_reached::Vector{Bool}
	converged::Vector{Bool}
    
    se_fitted_model_params::Vector{NamedTuple}

end