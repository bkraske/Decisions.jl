using Decisions

testrs = RockSampleDecisionsPOMDP()

#Test Sampling/Rand
testrs[:sp](;s=rand(support(testrs[:sp])),a=rand(support(testrs[:a])))
otest = testrs[:o](;s=rand(support(testrs[:sp])),a=rand(support(testrs[:a])),sp=rand(support(testrs[:sp])))
testrs[:r](;s=rand(support(testrs[:sp])),a=rand(support(testrs[:a])),sp=rand(support(testrs[:sp])))
testrs[:r](;s=rand(support(testrs[:sp])),a=rand(support(testrs[:a])),sp=rand(support(testrs[:sp])))
btest = testrs.initial.rand()
bp = testrs[:mp](;a=2,m=btest,o=3) #Check belief update correct

#support/pdf Test
sp = support(testrs[:sp];s=DecisionDomains.RockSample.RSState{3}([1, 1], Bool[0, 0, 0]),a=2)[1]
pdf(testrs[:sp],sp;s=DecisionDomains.RockSample.RSState{3}([1, 1], Bool[0, 0, 0]),a=2)
o = support(testrs[:o];s=DecisionDomains.RockSample.RSState{3}([1, 1], Bool[0, 0, 0]),a=1,sp=DecisionDomains.RockSample.RSState{3}([1, 1], Bool[0, 0, 0]))
for obs in o
    @show pdf(testrs[:o],obs;s=DecisionDomains.RockSample.RSState{3}([1, 1], Bool[1, 0, 0]),a=6,sp=DecisionDomains.RockSample.RSState{3}([1, 1], Bool[1, 0, 0]))
end
testrs[:r](;s=DecisionDomains.RockSample.RSState{3}([1, 1], Bool[1, 0, 0]),a=1,sp=DecisionDomains.RockSample.RSState{3}([1, 1], Bool[1, 0, 0]))
support(testrs[:mp];a=2,m=btest,o=3)
pdf(testrs[:mp],bp;a=2,m=btest,o=3)

## MOMDP Transformation Work
const MOMDP_DN = DecisionGraph(
  [ # Six regular nodes (:xp, :yp, :r, :o, :mp, :a)
    (Dense(:x), Dense(:y), Dense(:a)) => Joint(:xp),
    (Dense(:x), Dense(:y), Dense(:a), Dense(:xp)) => Joint(:yp),
    (Dense(:x), Dense(:y), Dense(:a), Dense(:xp), Dense(:yp)) => Joint(:r),
    (Dense(:x), Dense(:y), Dense(:a), Dense(:xp), Dense(:yp)) => Joint(:o),
    (Dense(:x), Dense(:o), Dense(:a), Dense(:m)) => Joint(:mp),
    (Dense(:m),) => Joint(:a),
  ],
  (; # Three dynamic nodes (:x, :y, :m)
    :x => :xp,
    :y => :yp,
    :m => :mp
  ),
  (;) # No ranges
)


struct ToMixedObservability{X,Y} <: DNTransformation
    fully_observable_state_components::X
    partially_observable_state_components::Y
    ToMixedObservability(x, y) = isa(x,Tuple) && isa(y,Tuple) ? new{typeof(x),typeof(y)}(x,y) : error("Use Tuples")
end

# function marginalize_old(model::POMDP_DN,s,a,idx)
#     sps = model[:s].support(;s=s,a=a)
#     for sp in model[:s].support(;s=s,a=a)

#     end
# end

function get_components(s,idx)
    if all(x -> x isa Symbol, idx)
        xp = Tuple(getfield(s,i) for i in idx)
    elseif all(x -> x isa Int, idx)
        xp = Tuple(s[i] for i in idx)
    else
        error("Use either all Symbols or all Ints for specifying indicies.")
    end
    return xp
end

# function marginalize(model,s,a,x_idx)
#     supp = support(model[:sp];s=s,a=a)
#     x_list = []
#     prob_list = Float64[]
#     for s in supp
#         prob = pdf(model[:sp],s;s=s,a=a)
#         x = get_components(s,x_idx)
#         inner_idx = findfirst(v->v==x,x_list)
#         if isnothing(inner_idx)
#             push!(x_list,x)
#             push!(prob_list,prob)
#         else
#             prob_list[inner_idx] += prob
#         end
#     end
#     return SparseCat()
# end

function marginal_pdf(model,x,s,a,x_idx)
    supp = support(model[:sp];s=s,a=a)
    prob = 0.0
    for s in supp
        sprob = pdf(model[:sp],s;s=s,a=a)
        if x == get_components(s,x_idx)
            prob += sprob
        end
    end
    return prob
end

function marginal_support(model,s,a,x_idx)
    supp = support(model[:sp];s=s,a=a)
    x = get_components(supp[1],x_idx)
    x_list = [x]
    for s in supp[2:end]
        x = get_components(s,x_idx)
        inner_idx = findfirst(v->v==x,x_list)
        if isnothing(inner_idx)
            push!(x_list,x)
        end
    end
    return x_list
end

function marginal_pdfy(model,yp,xp,s,a,y_idx,x_idx)
    supp = support(model[:sp];s=s,a=a)
    prob = 0.0
    for s in supp
        sprob = pdf(model[:sp],s;s=s,a=a)
        if yp == get_components(s,y_idx) && xp == get_components(s,x_idx)
            prob += sprob
        end
    end
    return prob
end

function marginal_supporty(model,xp,s,a,y_idx,x_idx)
    supp = support(model[:sp];s=s,a=a)
    yp = get_components(supp[1],y_idx)
    y_list = typeof(yp)[]
    for s in supp
        yp = get_components(s,y_idx)
        inner_idx = findfirst(v->v==yp,y_list)
        if isnothing(inner_idx) && get_components(s,x_idx) == xp
            push!(y_list,yp)
        end
    end
    return y_list
end

function s_from_xy(model,x,y,x_idx,y_idx)
    supp = support(model[:sp]) #Make this less terrible
    for s in supp
        if get_components(s,x_idx) == x && get_components(s,y_idx) == y
            return s
        end
    end
end

function Decisions.transform(components::ToMixedObservability, model::POMDP_DN)
    sup = support(model[:s])
    sup_size = length(sup)
    x_type = typeof(get_components(sup[1],components.fully_observable_state_components))
    y_type = typeof(get_components(sup[1],components.partially_observable_state_components))
    xs = x_type[] #(undef, sup_size)
    ys = y_type[] #(undef, sup_size)
    for s in sup
        push!(xs,get_components(s,components.fully_observable_state_components))
        push!(ys,get_components(s,components.partially_observable_state_components))
    end
    obs_type = typeof(support(model[:o])[1])
    # mem_type = typeof(support(model[]))

    xtransition = @ConditionalDist x_type begin
        function support(;x,y,a)
            if isnothing(x) || isnothing(a) || isnothing(y)
                xs
            else
                s = s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
                marginal_support(model,s,a,components.fully_observable_state_components)
            end
        end

        function rand(rng;x,y,a) #rand! (?)
            if POMDPs.isterminal(pomdp,s)
                Terminal()
            else
                s = s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
                get_components(model[:sp](;s=s,a=a),components.fully_observable_state_components)
            end
        end

        function pdf(xp;x,y,a)
            s = s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
            marginal_pdf(model,xp,s,a,x_idx)
        end
    end

    ytransition = @ConditionalDist y_type begin
        function support(;x,y,a,xp)
            if isnothing(x) || isnothing(a) || isnothing(y) || isnothing(xp)
                xs
            else
                s = s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
                marginal_supporty(model,xp,s,a,components.partially_observable_state_components,components.fully_observable_state_components)
            end
        end

        function rand(rng;x,y,a,xp) #rand! (?)
            if POMDPs.isterminal(pomdp,s)
                Terminal()
            else
                s = s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
                yp_support = marginal_supporty(model,xp,s,a,components.partially_observable_state_components,components.fully_observable_state_components)
                yp_probs = [marginal_pdfy(model,yp,xp,s,a,components.partially_observable_state_components,components.fully_observable_state_components) for yp in support]
                yp_support[rand(rng,DecisionDomains.Distributions.Categorical(yp_probs))]
            end
        end

        function pdf(yp;x,y,a,xp)
            s = s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
            marginal_pdfy(model,yp,xp,s,a,components.partially_observable_state_components,components.fully_observable_state_components)
        end
    end

    observation = @ConditionalDist obs_type begin
        function support(;x,y,a,xp,yp)
            if isnothing(x) || isnothing(y) || isnothing(a) ||  isnothing(xp) || isnothing(yp)
                support(model[:o])
            else
                s = s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
                sp = s_from_xy(model,xp,yp,components.fully_observable_state_components,components.partially_observable_state_components)
                Distributions.support(model[:o],s=s,a=a,sp=sp)
            end
        end

        function rand(rng;x,y,a,xp,yp)
            s = s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
            sp = s_from_xy(model,xp,yp,components.fully_observable_state_components,components.partially_observable_state_components)
            model[:o](rng;s=s,a=a,sp=sp)
        end

        function pdf(o;x,y,a,xp,yp)
            s = s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
            sp = s_from_xy(model,xp,yp,components.fully_observable_state_components,components.partially_observable_state_components)
            pdf(model[:o],o;s=s,a=a,sp=sp)
        end
    end

    reward = @ConditionalDist Float64 begin
        function rand(rng;x,y,a,xp,yp)
            s = s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
            sp = s_from_xy(model,xp,yp,components.fully_observable_state_components,components.partially_observable_state_components)
            model[:r](rng;a=a,s=s,sp=sp)
        end
    end

    belief = @ConditionalDist typeof(model[:mp]).parameters[2] begin #Return pomdp memory for now #FIX Typing here
        function support(;a,m,o,x)
            support(model[:mp];a=a,m=m,o=o)
        end

        function rand(rng;a,m,o,x)
            model[:mp](rng;a=a,m=m,o=o)
        end

        function pdf(mp;a,m,o,x)
            pdf(model[:mp],mp;a=a,m=m,o=o)
        end
    end

    return MOMDP_DN(;xp = xtransition,
    yp = ytransition,
    r = reward,
    o = observation,
    mp = belief)
end
