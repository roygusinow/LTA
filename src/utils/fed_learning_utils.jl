# Optimisation
using ForwardDiff

function grad_one_step(current_parameters, aggr_grad, sim_output::SimulationOutput; lambda = 0.0)

    function stable_grad_vec!(G, x)
        G .= aggr_grad
    end

	# define the likelihood function
	obj_func = params_vec -> objective_func(
		params_vec,

		sim_output = sim_output,
		lambda = lambda
	)
    next_grad = ForwardDiff.gradient(obj_func, current_parameters)

    return next_grad
end

function get_lkl_val(current_parameters, sim_output::SimulationOutput; lambda = 0.0)

	# define the likelihood function
	obj_func = params_vec -> objective_func(
		params_vec,

		sim_output = sim_output,
		lambda = lambda
	)
    likelihood = obj_func(current_parameters)

    return likelihood
end

# processing data
function create_LTS_sim_hyper(;
	kwargs...
)
    sim_hyper = SimHyperParams(
		param_seed = 1, 
		N = kwargs[:N],
        n_obs_tup = kwargs[:n_obs_tup],
		n_states = kwargs[:n_states],
        ref_state = 1, 
		covariate_tup = kwargs[:covariate_tup],
		sim_no = kwargs[:sim_no],
		comments = kwargs[:comments]
	)
	return sim_hyper
end

# ——— Build the 3D tensor according to visit_group designations ———
function build_tensor_designation(df::DataFrame,
                                  symptom_names::Vector{String},
                                  visit_prefixes::Vector{String},
                                  time_labels::Vector{String};
                                  Ttype::Type{T}=Int) where {T}
    N = nrow(df)
    Tn = length(time_labels)
    S = length(symptom_names)
    X = Array{T}(undef, N, Tn, S)
    X .= -1

    for (j, sym) in enumerate(symptom_names)
        for (k, label) in enumerate(time_labels)
            for i in 1:N
                if label == "acute"
                    col = Symbol("acute." * sym)
                    if df[i, col] !== missing
                        X[i, k, j] = convert(T, df[i, col])
                    else
                        X[i, k, j] = T(-1)
                    end
                else
                    # build one column at a time, per patient
                    for prefix in visit_prefixes
                        grp_col = Symbol(prefix * ".visit_group")
                        
                        # if the entry matches the time label, use it
                        # println(df[i, grp_col] !== missing)
                        if df[i, grp_col] !== missing
                            # println("hit")
                            if df[i, grp_col] == label
                                sym_col = Symbol(prefix * "." * sym)
                                if df[i, sym_col] !== missing
                                    X[i, k, j] = convert(T, df[i, sym_col])
                                else
                                    X[i, k, j] = T(-1)
                                end
                            end
                        end
                    end
                end

            end

        end
    end

    return X
end

# ——— Build the covariate matrix ———
function build_covariates(df::DataFrame,
                          covariate_names::Vector{String})

    patient_df = df[:, covariate_names]
    # insertcols!(patient_df, 1, :Intercept => ones(nrow(patient_df)))
    # insertcols!(patient_df, 1, :x1 => ones(nrow(patient_df))) # need to change for all sims..

    # convert all columns to Float64
    # for col in names(patient_df)
    #     patient_df[!, col] = Float64.(patient_df[!, col])
    # end

    return patient_df
end

# ——— Top‐level preprocess function ———
function preprocess(df::DataFrame;
                    covariates::Vector{String},
                    binary_syms::Vector{String},
                    cont_syms::Vector{String},
                    visit_prefixes::Vector{String},
                    time_labels::Vector{String})

    # Binary symptoms → Int tensor (with -1 for missing)
    bernoulli_observations  = build_tensor_designation(df, binary_syms,
                                        visit_prefixes, time_labels;
                                        Ttype = Int)

    # Continuous symptoms → Float64 tensor (with -1.0 for missing)
    gaussian_observations = build_tensor_designation(df, cont_syms,
                                        visit_prefixes, time_labels;
                                        Ttype = Float64)

    # Covariates → matrix
    pat_cov  = build_covariates(df, covariates)

    return (
      bernoulli_observations = bernoulli_observations,
      gaussian_observations   = gaussian_observations
    ), pat_cov
end


function is_all_minus_one(matrix)
    # Check if all elements in the matrix are equal to -1
    return all(x -> x == -1, matrix)
end

function mask_all_minus_one_matrices(arrs::NamedTuple)
    A1 = first(values(arrs))
    @assert ndims(A1) == 3 "All inputs must be 3D arrays"
    n = size(A1, 1)

    # Basic shape checks
    @assert all(ndims(A) == 3 for A in values(arrs)) "All inputs must be 3D arrays"
    @assert all(size(A, 1) == n for A in values(arrs)) "All arrays must agree on size(A,1)"

    mask = falses(n)
    @inbounds @views for i in 1:n
        # true iff every array's slice at i is all -1
        mask[i] = all(A -> is_all_minus_one(A[i, :, :]), values(arrs))
    end
    return mask
end

function create_data_sim_params(
	observations, covariate_df;
	kwargs...
)

    mask = mask_all_minus_one_matrices(observations)

    n_obs_tup = ()
    masked_observations = ()
    N = 0
    T = 0
    # risk of overwriting N and T if multiple observation types are present
    if haskey(observations, :bernoulli_observations)
        bernoulli_observations = observations.bernoulli_observations[.!mask, :, :]
        N, T, symp = size(bernoulli_observations)
        n_obs_tup = (; n_obs_tup..., bernoulli = symp)
        masked_observations = (; masked_observations..., bernoulli_observations = bernoulli_observations)
        N = size(bernoulli_observations, 1)
    end
    if haskey(observations, :gaussian_observations)
        gaussian_observations = observations.gaussian_observations[.!mask, :, :]

        N, T, symp = size(gaussian_observations)
        n_obs_tup = (; n_obs_tup..., gaussian = symp)
        masked_observations = (; masked_observations..., gaussian_observations = gaussian_observations)
    end
	covariate_df = covariate_df[.!mask, :]
    covariate_mat = Matrix{Float64}(covariate_df)

	sim_hyper = create_LTS_sim_hyper(;
		N = N,
        n_obs_tup = n_obs_tup,
        n_states = kwargs[:n_states],
        ref_state = 1,
        covariate_tup = kwargs[:covariate_tup],
		kwargs...
	)

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

	sim_output = SimulationOutput(
		sim_params = sim_params,
		simulation_seed = 1, # doesnt matter
		
		states = zeros(N, T),
		observations = masked_observations
	)

	return sim_output
end

function create_sim_mod_data(
    tab;
    kwargs...
)

    df = DataFrame(tab)

    observations, covariate_df = preprocess(df;
        covariates      = kwargs[:covs],
        binary_syms     = kwargs[:bins],
        cont_syms       = kwargs[:conts],
        visit_prefixes  = kwargs[:visits],
        time_labels     = kwargs[:labels],
    )

    # return observations, covariate_df

    sim_output = create_data_sim_params(
        observations, covariate_df;
        kwargs...
    )

    return sim_output
end

# data simulation
function create_ds_simulation_from_sim_params(
    sim_params::SimParams;
    kwargs...
)

    sim_output = run_simulation(
        sim_params;
        T = kwargs[:T],
        simulation_seed = kwargs[:simulation_seed],
        percentage_observations_missing = kwargs[:percentage_observations_missing],
    )

    bernoulli_observations = sim_output.observations.bernoulli_observations
    gaussian_observations = sim_output.observations.gaussian_observations

    v_vec = "v" .* string.(collect(1:4))
    v_vec = vcat("acute", v_vec)

    bin_obs_names = kwargs[:bin_obs_names]
    cont_obs_names = kwargs[:cont_obs_names]

    output_df = DataFrame(sim_params.covariate_mat, sim_params.covariate_mat_headers)
    
    for t in 1:size(bernoulli_observations, 2)
        # binary obs
        for symp_i in 1:size(bernoulli_observations, 3)
            output_df[!, Symbol(v_vec[t] * "." * bin_obs_names[symp_i])] = bernoulli_observations[:, t, symp_i]
        end

        # cont obs
        for symp_i in 1:size(gaussian_observations, 3)
            output_df[!, Symbol(v_vec[t] * "." * cont_obs_names[symp_i])] = gaussian_observations[:, t, symp_i]
        end
    end

    # add visit group
    visit_group_labels = kwargs[:visit_group_labels]
    for i in 1:length(visit_group_labels)
        visit_group_col = Symbol("v" * string(i) * ".visit_group")
        output_df[!, visit_group_col] .= visit_group_labels[i]
    end

    # add id col for datashield upload
    insertcols!(output_df, 1, :ID => collect(1:nrow(output_df)))

    return sim_output, output_df
end