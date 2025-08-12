using FacilityLocationProblems
using JuMP
using HiGHS
using Printf

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
    # set_silent(model)
    set_time_limit_sec(model, 30)
    optimize!(model)
    println(termination_status(model))

    @printf("Obj = %.3f\n", objective_value(model))
    @printf("Bound = %.3f\n", objective_bound(model))
end

function main()
    data = loadFacilityLocationProblem(:capa, 8000)
    println(data)
    solveModel(data)
end

@time main()