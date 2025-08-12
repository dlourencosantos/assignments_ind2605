using FacilityLocationProblems
using JuMP
using HiGHS
using Printf
using LinearAlgebra

function solve_benders(data::FacilityLocationProblem, max_iter::Int=100, ϵ::Float64=1e-4)
    # --- 1. Inicialização dos Dados ---
    C = 1:length(data.demands)
    F = 1:length(data.fixed_costs)
    Q = data.capacities
    d = data.demands
    f = data.fixed_costs
    c = data.costs'
    total_demand = sum(d)

    # --- 2. Problema Mestre ---
    master_model = Model(HiGHS.Optimizer)
    set_silent(master_model)
    @variable(master_model, y[j in F], Bin)
    @variable(master_model, η >= 0)
    @objective(master_model, Min, sum(f[j] * y[j] for j in F) + η)

    # Força que a capacidade total das instalações abertas seja suficiente para a demanda total.
    # Isso evita a solução trivial y=[0,0,...] e a infactibilidade imediata.
    @constraint(master_model, initial_capacity_cut, sum(Q[j] * y[j] for j in F) >= total_demand)


    # --- 3. Loop da Decomposição de Benders ---
    lower_bound = -Inf
    upper_bound = +Inf
    
    for iter in 1:max_iter
        # --- 3.1. Resolver o Problema Mestre Relaxado ---
        optimize!(master_model)
        
        if termination_status(master_model) != MOI.OPTIMAL
            @error "Problema mestre não resolvido otimamente. Status: $(termination_status(master_model))"
            break
        end

        y_star = value.(y)
        lower_bound = objective_value(master_model)
        current_fixed_cost = lower_bound - value(η)
        
        @printf("\nIteração %d:\n", iter)
        @printf("  Limite Inferior (Mestre): %.3f\n", lower_bound)

        # --- 3.2. Resolução do Subproblema Dual ---
        dual_subproblem = Model(HiGHS.Optimizer)
        set_silent(dual_subproblem)
        @variable(dual_subproblem, u[i in C])
        @variable(dual_subproblem, w[j in F] >= 0)
        
        @objective(dual_subproblem, Max, 
            sum(u[i] for i in C) - sum(Q[j] * y_star[j] * w[j] for j in F)
        )
        
        @constraint(dual_subproblem, dual_cons[i in C, j in F], u[i] - d[i] * w[j] <= c[i, j])

        optimize!(dual_subproblem)
        status = termination_status(dual_subproblem)

        # --- 3.3. Gerar Cortes e Atualizar Limites ---
        if status == MOI.OPTIMAL
            subproblem_obj = objective_value(dual_subproblem)
            current_upper_bound = current_fixed_cost + subproblem_obj
            upper_bound = min(upper_bound, current_upper_bound)
            
            @printf("  Subproblema factível. Custo: %.3f\n", subproblem_obj)
            @printf("  Melhor Limite Superior: %.3f\n", upper_bound)

            if upper_bound - lower_bound < ϵ
                println("\n Convergência alcançada!")
                break
            end

            @info "Adicionando corte de otimalidade."
            u_star = value.(u)
            w_star = value.(w)
            
            @constraint(master_model, η >= sum(u_star) - sum(Q[j] * w_star[j] * y[j] for j in F))

        elseif status == MOI.DUAL_INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
            @info "Subproblema infactível. Adicionando corte de viabilidade."
            
            open_facilities = [j for j in F if y_star[j] > 0.5]
            # Agora, este corte não será adicionado para um array vazio, pois o corte inicial já impede isso.
            if !isempty(open_facilities)
                @constraint(master_model, sum(y[j] for j in open_facilities) <= length(open_facilities) - 1)
            else
                # Este caso não deve mais acontecer, mas é bom ter um aviso.
                @warn "O subproblema foi infactível mesmo com o corte de capacidade inicial. O problema pode ser infactível."
            end
        else
             @error "Erro inesperado ao resolver o subproblema dual: $(status)"
             break
        end
        
        if iter == max_iter
            println("\n Máximo de iterações atingido.")
        end
    end

    @printf("\n--- Resultados Finais ---\n")
    @printf("Obj = %.3f\n", upper_bound)
    @printf("Bound = %.3f\n", lower_bound)
end

function solveModel(data::FacilityLocationProblem)
    C = 1:length(data.demands)
    F = 1:length(data.fixed_costs)
    Q = data.capacities
    d = data.demands
    f = data.fixed_costs
    c = data.costs'

    model = Model(HiGHS.Optimizer)

    @variable(model, y[j in F], Bin)
    @variable(model, 0 <= x[i in C, j in F] <= 1)

    @objective(model, Min, sum(c[i, j]x[i, j] for i in C, j in F) + sum(f[j]y[j] for j in F))

    @constraint(model, service[i in C], sum(x[i, j] for j in F) == 1)
    @constraint(model, bind[i in C, j in F], x[i, j] <= y[j])
    @constraint(model, knap[j in F], sum(d[i]x[i, j] for i in C) <= Q[j])
    # @constraint(model, knap[j in F], sum(d[i]x[i, j] for i in C) <= Q[j]y[j])

    # println(model)
    set_silent(model)
    set_time_limit_sec(model, 60)
    optimize!(model)
    println(termination_status(model))

    @printf("Obj = %.3f\n", objective_value(model))
    @printf("Bound = %.3f\n", objective_bound(model))
end

function main()
instances_to_run = [
    :cap41,
    :cap42,
    :cap43,
    :cap44,
    :cap51,
    :cap61,
    :cap62,
    :cap63,
    :cap64,
    :cap71,
    :cap72,
    :cap73,
    :cap74,
    :cap81,
    :cap82,
    :cap83,
    :cap84,
    :cap91,
    :cap92,
    :cap93,
    :cap94,
    :cap101,
    :cap102,
    :cap103,
    :cap104,
    :cap111,
    :cap112,
    :cap113,
    :cap114,
    :cap121,
    :cap122,
    :cap123,
    :cap124,
    :cap131,
    :cap132,
    :cap133,
    :cap134
]
    for i in instances_to_run
        data = loadFacilityLocationProblem(i)
        println(data)
        println("\n--- Resolvendo com Decomposição de Benders ---")
        solve_benders(data, 20)
        println("\n ---- Resolvendo modelo visto em aula")
        solveModel(data)
    end
end

@time main()