using PlotlyJS, ColorSchemes

cscale = [(0, "white"), (1, "red")]

function get_model_params(
    est_output::EstimationOutput;
    type = "simulation"
)

    sim_output, sim_params, sim_hyper = unpack(est_output)

    if type == "simulation"
        return est_output.sim_output.sim_params.model_params
    elseif type == "best_fitted"
        
        argmin_idx = argmin(est_output.fitted_log_lkl_list)
		model_params = est_output.fitted_model_params[argmin_idx]
        return model_params
    else
        model_params = est_output.fitted_model_params[type]
        return model_params
    end

end

# initial probs
function plot_single_initial(
	vec::Vector{Float64};
	# symptom_list = symptom_dict
)

	n_states = length(vec)
	mat = reshape(vec, (n_states, 1))
	plt =
		PlotlyJS.heatmap(
			x = "",
			y = reverse("State " .* string.(collect(1:n_states))),
			z = mat[end:-1:1, :],
			zmin = 0.0, zmax = 1.0,
			colorscale = cscale,
		)

	return plt
end

function compare_initial_heatmaps(
	est_output::EstimationOutput;
    patient_idx = 1,
    save_dir = nothing
)
	sim_output, sim_params, sim_hyper = unpack(est_output)

    covariate_idx = convert_covariate_2_df_indices(
        sim_params.covariate_mat_headers,
        sim_hyper.covariate_tup
    )
	x_vec = sim_params.covariate_mat[patient_idx, :]
    lkl_idx = sortperm(est_output.fitted_log_lkl_list)

	subplot_titles = ["Estimated $idx<br>-Log-lkl: $(round(Int, est_output.fitted_log_lkl_list[idx]))" for (val, idx) in enumerate(lkl_idx)]
	subplot_titles = vcat("Simulation<br>-Log-lkl:$(round(Int, est_output.true_log_lkl))", subplot_titles)
	s = length(est_output.fitted_log_lkl_list) + 1
	fig = PlotlyJS.make_subplots(
		rows = 1, cols = s,
		subplot_titles = reshape(subplot_titles, 1, s)
	)

	# ------------------------------------------------------------------
	# true model
	model_params = get_model_params(
        est_output;
        type = "simulation"
    )
    initial_states = get_rho_beta_initial_states(
        model_params.beta_initial, 
        model_params.rho_initial, 
        x_vec[covariate_idx[:initial]])
    true_heat_trace = plot_single_initial(initial_states)
	add_trace!(fig, true_heat_trace, row = 1, col = 1)

	# loop through all multistarts
	for (val, idx) in enumerate(lkl_idx)
		model_params = get_model_params(
            est_output;
            type = idx
        )
        initial_states = get_rho_beta_initial_states(
            model_params.beta_initial, 
            model_params.rho_initial, 
            x_vec[covariate_idx[:initial]])
        heat_trace = plot_single_initial(initial_states)
		add_trace!(fig, heat_trace, row = 1, col = val + 1)
	end

	relayout!(fig, width = 700 * s, height = 500)

    if !isnothing(save_dir)
        open(save_dir, "w") do io
            PlotlyBase.to_html(io, fig.plot)
        end
    end

	return fig
end

# transitiions
function plot_single_transition_mat(
	mat::Matrix{Float64},
)

	n_states = size(mat, 1)

	text_data = [string(round(mat[i, j], digits=2)) for i in 1:size(mat, 1), j in 1:size(mat, 2)]

	plt =
		PlotlyJS.heatmap(
			x = "To State " .* string.(collect(1:n_states)),
			y = reverse("From State " .* string.(collect(1:n_states))),
			z = mat[end:-1:1, :],
			zmin = 0.0, zmax = 1.0,

			colorscale = cscale,
		)

	return plt
end

function compare_trans_heatmaps(
	est_output::EstimationOutput;
    patient_idx = 1,
    save_dir = nothing
)
	sim_output, sim_params, sim_hyper = unpack(est_output)

    covariate_idx = convert_covariate_2_df_indices(
        sim_params.covariate_mat_headers,
        sim_hyper.covariate_tup
    )
	x_vec = sim_params.covariate_mat[patient_idx, :]
    lkl_idx = sortperm(est_output.fitted_log_lkl_list)

	subplot_titles = ["Estimated $idx<br>-Log-lkl: $(round(Int, est_output.fitted_log_lkl_list[idx]))" for (val, idx) in enumerate(lkl_idx)]
	subplot_titles = vcat("Simulation<br>-Log-lkl:$(round(Int, est_output.true_log_lkl))", subplot_titles)
	s = length(est_output.fitted_log_lkl_list) + 1
	fig = PlotlyJS.make_subplots(
		rows = 1, cols = s,
		subplot_titles = reshape(subplot_titles, 1, s)
	)

	# ------------------------------------------------------------------
	# true model
	model_params = get_model_params(
        est_output;
        type = "simulation"
    )
	transition_probs = get_rho_beta_transition_mat(
		model_params.beta_transition, model_params.rho_trans, x_vec[covariate_idx[:trans]])
    true_heat_trace = plot_single_transition_mat(transition_probs)
	add_trace!(fig, true_heat_trace, row = 1, col = 1)

	# loop through all multistarts
	for (val, idx) in enumerate(lkl_idx)
		model_params = get_model_params(
            est_output;
            type = idx
        )
		transition_probs = get_rho_beta_transition_mat(
			model_params.beta_transition, model_params.rho_trans, x_vec[covariate_idx[:trans]])
        heat_trace = plot_single_transition_mat(transition_probs)
		add_trace!(fig, heat_trace, row = 1, col = val + 1)
	end

	relayout!(fig, width = 700 * s, height = 500)

    if !isnothing(save_dir)
        open(save_dir, "w") do io
            PlotlyBase.to_html(io, fig.plot)
        end
    end

	return fig
end

# emissions
function plot_single_bernoulli_probs(
	mat::Matrix{Float64};
	symptom_labels = nothing
)

	if isnothing(symptom_labels)
		symptom_labels = "To Symptom " .* string.(collect(1:size(mat, 2)))
	end

	plt =
		PlotlyJS.heatmap(
			x = symptom_labels,
			y = reverse("From State " .* string.(collect(1:size(mat, 1)))),
			z = mat[end:-1:1, :],
			zmin = 0.0, zmax = 1.0,
			colorscale = cscale,
            )

	return plt
end

function compare_bernoulli_heatmaps(
	est_output::EstimationOutput;
    patient_idx = 1,
    save_dir = nothing
)
	sim_output, sim_params, sim_hyper = unpack(est_output)

    covariate_idx = convert_covariate_2_df_indices(
        sim_params.covariate_mat_headers,
        sim_hyper.covariate_tup
    )
	x_vec = sim_params.covariate_mat[patient_idx, :]
    lkl_idx = sortperm(est_output.fitted_log_lkl_list)

	subplot_titles = ["Estimated $idx<br>-Log-lkl: $(round(Int, est_output.fitted_log_lkl_list[idx]))" for (val, idx) in enumerate(lkl_idx)]
	subplot_titles = vcat("Simulation<br>-Log-lkl:$(round(Int, est_output.true_log_lkl))", subplot_titles)
	s = length(est_output.fitted_log_lkl_list) + 1
	fig = PlotlyJS.make_subplots(
		rows = 1, cols = s,
		subplot_titles = reshape(subplot_titles, 1, s)
	)

	# ------------------------------------------------------------------
	# true model
	model_params = get_model_params(
        est_output;
        type = "simulation"
    )
	em_states = get_bernoulli_probs(model_params.emissions.beta_bernoulli, x_vec[covariate_idx[:em]])
	true_heat_trace = plot_single_bernoulli_probs(em_states)
	add_trace!(fig, true_heat_trace, row = 1, col = 1)

	# loop through all multistarts
	for (val, idx) in enumerate(lkl_idx)
		model_params = get_model_params(
            est_output;
            type = idx
        )
		em_states = get_bernoulli_probs(model_params.emissions.beta_bernoulli, x_vec[covariate_idx[:em]])
		heat_trace = plot_single_bernoulli_probs(em_states)
		add_trace!(fig, heat_trace, row = 1, col = val + 1)
	end

	relayout!(fig, width = 700 * s, height = 500)

    if !isnothing(save_dir)
        open(save_dir, "w") do io
            PlotlyBase.to_html(io, fig.plot)
        end
    end

	return fig
end

# gaussian emissions
function plot_single_gaussian_pdf(;
	μ::Float64, σ::Float64, x_range::Vector{Float64}, name::String, line_color = "red"
)

	# Define the normal distribution
    dist = Normal(μ, σ)
    # Compute the PDF values over the x_range
    y_values = pdf.(dist, x_range)

	trace = PlotlyJS.scatter(
		x = x_range, y = y_values, 
		mode = "lines", name = name, 
		line_color = line_color, fill="tozeroy", showlegend = false)

	return trace
end

function compare_gaussian_pdf(
	est_output::EstimationOutput;
	xlims::Vector{Float64} = [0.0, 1.0],
	ylims::Vector{Float64} = [0.0, 10.0],
    save_dir = nothing
)
	sim_output, sim_params, sim_hyper = unpack(est_output)

    lkl_idx = sortperm(est_output.fitted_log_lkl_list)

	subplot_titles = [repeat(["Estimated $idx<br>-Log-lkl: $(round(Int, est_output.fitted_log_lkl_list[idx]))"], sim_hyper.n_obs_tup.gaussian) for (val, idx) in enumerate(lkl_idx)]
	subplot_titles = vcat(repeat(["Simulation<br>-Log-lkl:$(round(Int, est_output.true_log_lkl))"], sim_hyper.n_obs_tup.gaussian), subplot_titles...)
	s = (length(est_output.fitted_log_lkl_list) + 1) * sim_hyper.n_obs_tup.gaussian
	fig = PlotlyJS.make_subplots(
		rows = sim_hyper.n_states, cols = s,
		subplot_titles = reshape(subplot_titles, 1, s),
		shared_yaxes = "rows",
		shared_xaxes = "columns"
	)
	

	# ------------------------------------------------------------------
	# true model
	model_params = get_model_params(
        est_output;
        type = "simulation"
    )
	true_gaussian_emission = convert_gauss_2_true(model_params.emissions.beta_gaussian)
	x_range = collect(range(xlims[1], xlims[2]; length = 500))
	for i in 1:sim_hyper.n_states
		for j in 1:sim_hyper.n_obs_tup.gaussian
			# plot the true gaussian pdf
			true_heat_trace = plot_single_gaussian_pdf(;
				μ = true_gaussian_emission.means[i, j], 
				σ = true_gaussian_emission.stds[i, j], 
				x_range = x_range, 
				name = "State " * string(i)
			)
			add_trace!(fig, true_heat_trace, row = i, col = j)
		end
	end

	# loop through all multistarts
	for (val, idx) in enumerate(lkl_idx)
		model_params = get_model_params(
            est_output;
            type = idx
        )
		true_gaussian_emission = convert_gauss_2_true(model_params.emissions.beta_gaussian)
		x_range = collect(range(xlims[1], xlims[2]; length = 500))
		for i in 1:sim_hyper.n_states
			for j in 1:sim_hyper.n_obs_tup.gaussian
				# plot the true gaussian pdf
				true_heat_trace = plot_single_gaussian_pdf(;
					μ = true_gaussian_emission.means[i, j], 
					σ = true_gaussian_emission.stds[i, j], 
					x_range = x_range, 
					name = "State " * string(i)
				)
				add_trace!(fig, true_heat_trace, row = i, col = val*sim_hyper.n_obs_tup.gaussian + j)
			end
		end
	end

	# relayout!(fig; Symbol("yaxis"*"_range") => ylims)
    # for axis_idx in 1:(sim_hyper.n_states * sim_hyper.n_obs_tup.gaussian)
    #     relayout!(fig; Symbol("yaxis$(axis_idx)"*"_range") => ylims)
    # end

	relayout!(fig; Symbol("yaxis"*"_range")=> ylims)
	for axis_idx in 1:(sim_hyper.n_states * (length(lkl_idx) + 1) * sim_hyper.n_obs_tup.gaussian * 2)
        relayout!(fig; Symbol("yaxis$axis_idx"*"_range")=> ylims)
    end


	relayout!(fig, width = 600 * s, height = 500)

    if !isnothing(save_dir)
        open(save_dir, "w") do io
            PlotlyBase.to_html(io, fig.plot)
        end
    end

	return fig
end

function dynamic_stacked_subplots(x, all_series, all_labels)
    n = length(all_series)
    fig = make_subplots(rows = n, cols = 1, shared_xaxes = true, vertical_spacing = 0.03)
    palette = ColorScheme(get(ColorSchemes.darkrainbow, range(0.0, 1.0, length=length(all_labels[1]))))
	# palette = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    for i in 1:n
        ys = all_series[i]
        labels = all_labels[i]
		idx = 1
        for (y, lbl) in zip(ys, labels)
            add_trace!(fig, bar(x = x, y = y, name = lbl, marker_color = palette[idx]), row = i, col = 1)
			idx += 1
        end
    end
    relayout!(fig, barmode = "stack", title_text = "")

	relayout!(fig; Symbol("yaxis_title_text") => "State 1")
	for i in 1:n
		relayout!(fig; Symbol("yaxis$(i)_title_text") => "State $(i)")
	end
    fig
end

function plot_ordinal_probs(
    est_output::EstimationOutput;
    type = "best_fitted",
)
	sim_output, sim_params, sim_hyper = unpack(est_output)
    model_params = get_model_params(est_output; type = type)

    ordinal_probs = LTA.convert_ordinal_2_true(model_params.emissions.beta_ordinal)

    x = string.(keys(ordinal_probs))
    states_vec_mat = zeros(sim_hyper.n_states, length(sim_hyper.n_obs_tup.ordinal), maximum(sim_hyper.n_obs_tup.ordinal))
    for sympt_mat_i in 1:length(ordinal_probs)
        for state_i in 1:size(ordinal_probs[sympt_mat_i], 1)
            row = ordinal_probs[sympt_mat_i][state_i, :]
            states_vec_mat[state_i, sympt_mat_i, 1:length(row)] = row
        end
    end

    all_series = [[states_vec_mat[i, :, level_i] for level_i in 1:maximum(sim_hyper.n_obs_tup.ordinal)] for i in 1:sim_hyper.n_states]
    all_labels = [string.(collect(1:maximum(sim_hyper.n_obs_tup.ordinal))) for _ in 1:sim_hyper.n_states]

    fig = dynamic_stacked_subplots(x, all_series, all_labels)

    return fig
end

# rho
function rho_trace(
	variables::Vector{String}, 
	estimates::Vector{Float64};
	se_vals::Vector{Float64} = zeros(length(estimates))
)

	# handle p_ values
	pv = p_values(estimates, se_vals ./ 1.96)

	# Define color intensities based on p-values
	function get_color(value, p_value)
		# If not significant or the value is too close to zero, return gray.
		if p_value > 0.05 || abs(value) ≤ 0.1
			return "gray"
		end
	
		# Compute an intensity factor based on the p-value.
		# For p_value in [0, 0.05], we map it to an intensity in [1, 0]:
		#   p_value == 0   --> intensity = 1 (most significant → darkest)
		#   p_value == 0.05 --> intensity = 0 (least significant → lightest)
		intensity = 1 - (p_value / 0.05)
	
		# Define thresholds for the intensity factor.
		# You can adjust these thresholds to fine-tune the color breakpoints.
		if value > 0.1  # Positive values: use red shades.
			if intensity ≥ 0.8
				return "darkred"
			elseif intensity ≥ 0.5
				return "red"
			else
				return "lightcoral"
			end
		elseif value < -0.1  # Negative values: use green shades.
			if intensity ≥ 0.8
				return "darkgreen"
			elseif intensity ≥ 0.5
				return "green"
			else
				return "lightgreen"
			end
		else
			# For values between -0.1 and 0.1, return gray.
			return "gray"
		end
	end

	colors = [get_color(estimates[i], pv[i]) for i in 1:length(estimates)]

	trace = PlotlyJS.scatter(
		x = estimates,
		y = variables,
		mode = "markers+text",
		marker = attr(
			size = 14,
			color = colors # Conditional coloring
			),
		error_x = attr(
			type = "data",
			symmetric = true,  # Use asymmetric error bars if needed
			array = se_vals,  # Upper error (distance from estimate to upper bound)
			color = colors
			# arrayminus = estimates .- lower_bound  # Lower error (distance from estimate to lower bound)
		)
	)

	return trace
	
end

function set_all_xranges!(fig, lo, hi)
    n_axes = count(k -> startswith(String(k), "xaxis"), keys(fig.plot.layout))
    
    for i in 1:n_axes
		if i == 1
			PlotlyJS.relayout!(
				fig;
				Symbol("xaxis") => attr(range = (lo, hi)),
			)
		else
			PlotlyJS.relayout!(
				fig;
				Symbol("xaxis$i") => attr(range = (lo, hi)),
			)
		end
    end
    
    return fig
end

function compare_rhos(
	est_output::EstimationOutput;
    rho_type = "initial",
    save_dir = nothing
)
	sim_output, sim_params, sim_hyper = unpack(est_output)

    lkl_idx = sortperm(est_output.fitted_log_lkl_list)

	subplot_titles = ["Estimated $idx<br>-Log-lkl: $(round(Int, est_output.fitted_log_lkl_list[idx]))" for (val, idx) in enumerate(lkl_idx)]
	subplot_titles = vcat("Simulation<br>-Log-lkl:$(round(Int, est_output.true_log_lkl))", subplot_titles)
	s = length(est_output.fitted_log_lkl_list) + 1
	fig = PlotlyJS.make_subplots(
		rows = 1, cols = s,
		subplot_titles = reshape(subplot_titles, 1, s),
		shared_yaxes = "rows",
		shared_xaxes = "columns"
	)

	# ------------------------------------------------------------------
	# true model
	model_params = get_model_params(
        est_output;
        type = "simulation"
    )
	true_heat_trace = rho_trace(
		sim_hyper.covariate_tup[rho_type == "initial" ? :initial : :trans],
		model_params[rho_type == "initial" ? :rho_initial : :rho_trans]
	)
	add_trace!(fig, true_heat_trace, row = 1, col = 1)

	# loop through all multistarts
	for (val, idx) in enumerate(lkl_idx)
		model_params = get_model_params(
            est_output;
            type = idx
        )
		heat_trace = rho_trace(
			sim_hyper.covariate_tup[rho_type == "initial" ? :initial : :trans],
			model_params[rho_type == "initial" ? :rho_initial : :rho_trans]
		)
		add_trace!(fig, heat_trace, row = 1, col = val + 1)
	end
	set_all_xranges!(fig, -2, 2)

	relayout!(fig, width = 700 * s, height = 500)

    if !isnothing(save_dir)
        open(save_dir, "w") do io
            PlotlyBase.to_html(io, fig.plot)
        end
    end

	return fig
end

# rho 
function p_values(estimates, std_errs)
    # Compute the test statistics (z-values) element-wise
    t_values = estimates ./ std_errs
    # Compute two-tailed p-values for each z-value
    return 2 .* (1 .- cdf.(Normal(0, 1), abs.(t_values)))
end

# model evaluations
# function run_simulation_from_estimation(
# 	est_output::EstimationOutput;
# 	simulation_seed = 1,
# 	T = 5,
# 	type = "best_fitted",

# )

# 	# unpack
# 	sim_output, sim_params, sim_hyper = unpack(est_output)

# 	# take in a sim_param struct and run the simulation
# 	Random.seed!(simulation_seed)	
	
# 	# simulate!
# 	covariate_df = DataFrame(sim_params.covariate_mat, sim_params.covariate_mat_headers)

# 	model_params = get_model_params(
#         est_output;
#         type = type
#     )
# 	states, observations = simulate(;
# 		model_params = model_params,
# 		covariate_df = covariate_df,
# 		covariate_tup = sim_params.sim_hyper.covariate_tup,
# 		T = T
# 	)

# 		# assign all vars to struct now
# 	sim_output = SimulationOutput(
# 		sim_params = sim_params,
# 		simulation_seed = simulation_seed, 
		
# 		states = states,
# 		observations = observations
# 	)

# 	return sim_output
	
# end

# # new
# _symptom_time(mat) = size(mat, 1) ≤ size(mat, 2) ? mat : permutedims(mat)

# function gaussian_means_by_symptom_time(obs)
#     # Supports either Vector{Matrix} (each: individuals × time) per symptom,
#     # or a 3D Array (individuals × time × symptoms)
#     if obs isa AbstractVector{<:AbstractMatrix}
#         S = length(obs)
#         T = size(obs[1], 2)
#         M = Array{Float64}(undef, S, T)
#         @inbounds for s in 1:S, t in 1:T
#             M[s, t] = mean(obs[s][:, t])
#         end
#         return M
#     elseif ndims(obs) == 3
#         N, T, S = size(obs)
#         M = Array{Float64}(undef, S, T)
#         @inbounds for s in 1:S, t in 1:T
#             M[s, t] = mean(@view obs[:, t, s])
#         end
#         return M
#     else
#         error("Unsupported gaussian_observations container: $(typeof(obs))")
#     end
# end

# function _subplot_titles(S, base)
#     [string(base, " (symptom ", s, ")") for s in 1:S]
# end

# # --- plotting --------------------------------------------------------------

# function create_bernoulli_proportion_plot2(estimated_props, true_props; symptom_names::Union{Nothing,Vector}=nothing)
#     est = _symptom_time(estimated_props)
#     tru = _symptom_time(true_props)
#     S, T = size(est)
#     titles = isnothing(symptom_names) ? _subplot_titles(S, "Bernoulli proportion") :
#              [string("Bernoulli proportion — ", n) for n in symptom_names]

#     plt = make_subplots(rows=S, cols=1; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=titles)

#     xs = collect(1:T)
#     for s in 1:S
#         add_trace!(plt, scatter(x=xs, y=vec(tru[s, :]), mode="lines+markers", name=(s==1 ? "True" : "True (s$s)")); row=s, col=1)
#         add_trace!(plt, scatter(x=xs, y=vec(est[s, :]), mode="lines+markers", name=(s==1 ? "Estimated" : "Estimated (s$s)")); row=s, col=1)
#         relayout!(plt, Dict("yaxis$(s)_title_text" => "prop", "xaxis$(s)_title_text" => (s==S ? "time" : "")))
#     end
#     relayout!(plt, title="Observed vs Estimated Bernoulli Proportions", legend_title_text="Series")
#     return plt
# end

# function create_gaussian_mean_plot(estimated_gauss, true_gauss; symptom_names::Union{Nothing,Vector}=nothing)
#     estM = _gaussian_means_by_symptom_time(estimated_gauss)  # S × T means
#     truM = _gaussian_means_by_symptom_time(true_gauss)       # S × T means
#     S, T = size(estM)
#     titles = isnothing(symptom_names) ? _subplot_titles(S, "Gaussian mean") :
#              [string("Gaussian mean — ", n) for n in symptom_names]

#     plt = make_subplots(rows=S, cols=1; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=titles)

#     xs = collect(1:T)
#     for s in 1:S
#         add_trace!(plt, scatter(x=xs, y=vec(truM[s, :]), mode="lines+markers", name=(s==1 ? "True mean" : "True mean (s$s)")); row=s, col=1)
#         add_trace!(plt, scatter(x=xs, y=vec(estM[s, :]), mode="lines+markers", name=(s==1 ? "Estimated mean" : "Estimated mean (s$s)")); row=s, col=1)
#         relayout!(plt, Dict("yaxis$(s)_title_text" => "mean", "xaxis$(s)_title_text" => (s==S ? "time" : "")))
#     end
#     relayout!(plt, title="Observed vs Estimated Gaussian Means", legend_title_text="Series")
#     return plt
# end

# # --- main ------------------------------------------------------------------

# function compare_estimation_2_data(
#     est_output::EstimationOutput;
#     simulation_seed = 1,
#     T = 5,
#     type = "best_fitted",
#     observation_type = "bernoulli",
#     symptom_names::Union{Nothing,Vector}=nothing
# )
#     estimated_sim_output = run_simulation_from_estimation(
#         est_output;
#         simulation_seed = simulation_seed,
#         T = T,
#         type = type
#     )

#     if observation_type == "bernoulli"
#         # compare Bernoulli observations via proportions
#         estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
#         true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)

#         fig = create_bernoulli_proportion_plot2(estimated_props, true_props; symptom_names = symptom_names)

#     elseif observation_type == "gaussian"
#         # compare Gaussian observations via per-time means (one subplot per symptom)
#         est_gauss  = estimated_sim_output.observations.gaussian_observations
#         true_gauss = est_output.sim_output.observations.gaussian_observations

#         fig = create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names=symptom_names)

#     else
#         error("Unknown observation type: $observation_type")
#     end

#     display(fig)
#     return estimated_sim_output, fig
# end

function run_simulation_from_estimation(
    est_output::EstimationOutput;
    simulation_seed::Integer = 1,
    T::Integer = 5,
    type::AbstractString = "best_fitted",
)
    # unpack (keep names explicit for clarity)
    sim_output, sim_params, sim_hyper = unpack(est_output)

    # seed for reproducibility
    Random.seed!(simulation_seed)

    # covariates to DataFrame (assumes headers align with columns)
    covariate_df = DataFrame(sim_params.covariate_mat, sim_params.covariate_mat_headers)

    # pull model params from estimation
    model_params = get_model_params(est_output; type = type)

    # simulate!
    states, observations = simulate(
        ; model_params = model_params,
          covariate_df = covariate_df,
          covariate_tup = sim_params.sim_hyper.covariate_tup, # uses sim_params' hyper
          T = T
    )

    # package results
    return SimulationOutput(
        sim_params = sim_params,
        simulation_seed = simulation_seed,
        states = states,
        observations = observations,
    )
end

# --- helpers -----------------------------------------------------------------

# Ensure matrices are S × T (symptom × time)
_symptom_time(mat::AbstractMatrix) = size(mat, 1) ≤ size(mat, 2) ? mat : permutedims(mat)

# Compute per-time means for Gaussian observations across individuals.
# Accepts either Vector{Matrix} with (N × T) per symptom, or a 3D Array (N × T × S).
function gaussian_means_by_symptom_time(obs)
    if obs isa AbstractVector{<:AbstractMatrix}
        S = length(obs)
        @assert S > 0 "Empty gaussian observation vector"
        N, T = size(obs[1])
        @assert all(size(M) == (N, T) for M in obs) "All symptom matrices must be N×T"
        M = Array{Float64}(undef, S, T)
        @inbounds for s in 1:S, t in 1:T
            M[s, t] = mean(obs[s][:, t])
        end
        return M
    elseif ndims(obs) == 3
        N, T, S = size(obs)
        M = Array{Float64}(undef, S, T)
        @inbounds for s in 1:S, t in 1:T
            M[s, t] = mean(@view obs[:, t, s])
        end
        return M
    else
        error("Unsupported gaussian_observations container: $(typeof(obs))")
    end
end

_subplot_titles(S::Integer, base::AbstractString) =
    [string(base, " (symptom ", s, ")") for s in 1:S]

# --- plotting ---------------------------------------------------------------
function get_binary_proportions(obs; skip_missing::Bool = true)
    proportion(v) = skip_missing ? mean(skipmissing(v)) : mean(v)

    if obs isa AbstractVector{<:AbstractMatrix}
        S = length(obs)
        @assert S > 0 "Empty bernoulli observation vector"
        N, T = size(obs[1])
        @assert all(size(M) == (N, T) for M in obs) "All symptom matrices must be N×T"

        P = Array{Float64}(undef, S, T)
        @inbounds for s in 1:S, t in 1:T
            P[s, t] = proportion(@view obs[s][:, t])
        end
        return P

    elseif ndims(obs) == 3
        N, T, S = size(obs)
        P = Array{Float64}(undef, S, T)
        @inbounds for s in 1:S, t in 1:T
            P[s, t] = proportion(@view obs[:, t, s])
        end
        return P

    else
        error("Unsupported bernoulli_observations container: $(typeof(obs))")
    end
end

function create_bernoulli_proportion_plot2(
    estimated_props::AbstractMatrix,
    true_props::AbstractMatrix;
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    est = _symptom_time(estimated_props)
    tru = _symptom_time(true_props)
    @assert size(est) == size(tru) "Estimated and true proportion arrays must have same size"
    S, T = size(est)

    titles = isnothing(symptom_names) ? _subplot_titles(S, "Bernoulli proportion") :
             [string("Bernoulli proportion — ", n) for n in symptom_names]
	titles = reshape([string(t) for t in titles], S, 1)

    plt = make_subplots(rows=S, cols=1; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=titles)
    xs = collect(1:T)

    for s in 1:S
        add_trace!(plt, scatter(x=xs, y=vec(tru[s, :]), mode="lines+markers", name=(s==1 ? "True" : "True ($(titles[s]))")); row=s, col=1)
        add_trace!(plt, scatter(x=xs, y=vec(est[s, :]), mode="lines+markers", name=(s==1 ? "Estimated" : "Estimated ($(titles[s]))")); row=s, col=1)
		relayout!(plt;
			Symbol("yaxis$(s)") => attr(title = "Percentage Frequency"),
			Symbol("xaxis$(s)") => attr(title = (s == S ? "time" : ""))  # only bottom row gets x-title
		)
    end

    relayout!(plt, title="Observed vs Estimated Bernoulli Proportions",
                   legend_title_text="Series")
    return plt
end

function create_gaussian_mean_plot(
    estimated_gauss,
    true_gauss;
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    estM = gaussian_means_by_symptom_time(estimated_gauss)  # S × T
    truM = gaussian_means_by_symptom_time(true_gauss)       # S × T
    @assert size(estM) == size(truM) "Estimated and true Gaussian mean arrays must have same size"
    S, T = size(estM)

    titles = isnothing(symptom_names) ? _subplot_titles(S, "Gaussian mean") :
             [string("Gaussian mean — ", n) for n in symptom_names]
	titles = reshape([string(t) for t in titles], S, 1)

    plt = make_subplots(rows=S, cols=1; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=titles)
    xs = collect(1:T)

    for s in 1:S
        add_trace!(plt, scatter(x=xs, y=vec(truM[s, :]), mode="lines+markers", name=(s==1 ? "True mean" : "True mean (s$s)")); row=s, col=1)
        add_trace!(plt, scatter(x=xs, y=vec(estM[s, :]), mode="lines+markers", name=(s==1 ? "Estimated mean" : "Estimated mean (s$s)")); row=s, col=1)
		relayout!(plt;
			Symbol("yaxis$(s)") => attr(title = "Mean"),
			Symbol("xaxis$(s)") => attr(title = (s == S ? "time" : ""))  # only bottom row gets x-title
		)
    end

    relayout!(plt, title="Observed vs Estimated Gaussian Means",
                   legend_title_text="Series")
    return plt
end

# --- main -------------------------------------------------------------------

function compare_estimation_2_data(
    est_output::EstimationOutput;
    simulation_seed::Integer = 1,
    T::Integer = 5,
    type::AbstractString = "best_fitted",
    observation_type::AbstractString = "bernoulli",
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    estimated_sim_output = run_simulation_from_estimation(
        est_output;
        simulation_seed = simulation_seed,
        T = T,
        type = type
    )

    fig = if observation_type == "bernoulli"
        # proportions over time (S × T expected after _symptom_time)
        estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
        true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)
        create_bernoulli_proportion_plot2(estimated_props, true_props; symptom_names = symptom_names)

    elseif observation_type == "gaussian"
        # per-time means (one subplot per symptom)
        est_gauss  = estimated_sim_output.observations.gaussian_observations
        true_gauss = est_output.sim_output.observations.gaussian_observations
        create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names = symptom_names)

    else
        error("Unknown observation type: $observation_type (use \"bernoulli\" or \"gaussian\")")
    end

    return fig
end