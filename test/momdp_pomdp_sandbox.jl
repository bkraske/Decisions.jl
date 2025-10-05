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
    for sp in supp
        sprob = pdf(model[:sp],sp;s=s,a=a)
        if x == get_components(sp,x_idx)
            prob += sprob
        end
    end
    return prob
end

function marginal_support(model,s,a,x_idx)
    supp = support(model[:sp];s=s,a=a)
    x = get_components(supp[1],x_idx)
    x_list = [x]
    for sp in supp[2:end]
        x = get_components(sp,x_idx)
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
    for sp in supp
        sprob = pdf(model[:sp],sp;s=s,a=a)
        if yp == get_components(sp,y_idx) && xp == get_components(sp,x_idx)
            prob += sprob
        end
    end
    return prob
end

function marginal_supporty(model,xp,s,a,y_idx,x_idx)
    supp = support(model[:sp];s=s,a=a)
    yp = get_components(supp[1],y_idx)
    y_list = typeof(yp)[]
    for sp in supp
        yp = get_components(sp,y_idx)
        inner_idx = findfirst(v->v==yp,y_list)
        if isnothing(inner_idx) && get_components(sp,x_idx) == xp
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

    xs = Vector{x_type}(undef, sup_size)
    ys = Vector{y_type}(undef, sup_size)
    xys_dict = Dict{Tuple{x_type,y_type},typeof(sup[1])}()

    for (i,s_i) in enumerate(sup)
        x_i = get_components(s_i,components.fully_observable_state_components)
        y_i = get_components(s_i,components.partially_observable_state_components)
        xs[i] = x_i
        ys[i] = y_i
        push!(xys_dict,(x_i,y_i)=>s_i)
        # x_i in xs ? nothing : push!(xs,x_i)
        # y_i in ys ? nothing : push!(ys,y_i)
    end
    xs = unique(xs)
    ys = unique(ys)

    obs_type = typeof(support(model[:o])[1])

    xtransition = @ConditionalDist x_type begin
        function support(;x,y,a)
            if isnothing(x) || isnothing(a) || isnothing(y)
                xs
            else
                s = xys_dict[(x,y)]
                # s_from_xy(model,x,y,components.fully_observable_state_components,components.partially_observable_state_components)
                marginal_support(model,s,a,components.fully_observable_state_components)
            end
        end

        function rand(rng;x,y,a) #rand! (?)
            s = xys_dict[(x,y)]
            if isterminal(s)
                Terminal()
            else
                get_components(model[:sp].rand(rng;s=s,a=a),components.fully_observable_state_components)
            end
        end

        function pdf(xp;x,y,a)
            s = xys_dict[(x,y)]
            marginal_pdf(model,xp,s,a,components.fully_observable_state_components)
        end
    end

    ytransition = @ConditionalDist y_type begin
        function support(;x,y,a,xp)
            if isnothing(x) || isnothing(a) || isnothing(y) || isnothing(xp)
                ys
            else
                s = xys_dict[(x,y)]
                marginal_supporty(model,xp,s,a,components.partially_observable_state_components,components.fully_observable_state_components)
            end
        end

        function rand(rng;x,y,a,xp) #rand! (?)
            s = xys_dict[(x,y)]
            if isterminal(s)
                Terminal()
            else
                yp_support = marginal_supporty(model,xp,s,a,components.partially_observable_state_components,components.fully_observable_state_components)
                yp_probs = [marginal_pdfy(model,yp,xp,s,a,components.partially_observable_state_components,components.fully_observable_state_components) for yp in yp_support]
                yp_support[rand(rng,DecisionDomains.Distributions.Categorical(yp_probs))] #Make Cleaner
            end
        end

        function pdf(yp;x,y,a,xp)
            s = xys_dict[(x,y)]
            marginal_pdfy(model,yp,xp,s,a,components.partially_observable_state_components,components.fully_observable_state_components)
        end
    end

    mom_observation = @ConditionalDist obs_type begin
        function support(;x,y,a,xp,yp)
            if isnothing(x) || isnothing(y) || isnothing(a) ||  isnothing(xp) || isnothing(yp)
                support(model[:o])
            else
                s = xys_dict[(x,y)]
                sp = xys_dict[(xp,yp)]
                support(model[:o];s=s,a=a,sp=sp)
            end
        end

        function rand(rng;x,y,a,xp,yp)
            s = xys_dict[(x,y)]
            sp = xys_dict[(xp,yp)]
            model[:o].rand(rng;s=s,a=a,sp=sp)
        end

        function pdf(o;x,y,a,xp,yp)
            s = xys_dict[(x,y)]
            sp = xys_dict[(xp,yp)]
            pdf(model[:o],o;s=s,a=a,sp=sp)
        end
    end

    mom_reward = @ConditionalDist Float64 begin
        function rand(rng;x,y,a,xp,yp)
            s = xys_dict[(x,y)]
            sp = xys_dict[(xp,yp)]
            model[:r].rand(rng;a=a,s=s,sp=sp)
        end
    end

    mom_belief = @ConditionalDist typeof(model[:mp]).parameters[2] begin #Return pomdp memory for now #FIX Typing here
        function support(;a,m,o,x)
            support(model[:mp];a=a,m=m,o=o)
        end

        function rand(rng;a,m,o,x)
            model[:mp].rand(rng;a=a,m=m,o=o)
        end

        function pdf(mp;a,m,o,x)
            pdf(model[:mp],mp;a=a,m=m,o=o)
        end
    end

    return MOMDP_DN(;xp = xtransition,
    yp = ytransition,
    r = mom_reward,
    o = mom_observation,
    mp = mom_belief,
    a = model[:a])
end

#Test Sampling/Rand
momdp[:xp](;x=rand(support(momdp[:xp])),a=rand(support(momdp[:a])),y=rand(support(momdp[:yp])))
momdp[:yp](;x=rand(support(momdp[:xp])),a=rand(support(momdp[:a])),y=rand(support(momdp[:yp])))
otest = momdp[:o](;x=rand(support(momdp[:xp])),y=rand(support(momdp[:yp])),a=rand(support(momdp[:a])),xp=rand(support(momdp[:xp])),yp=rand(support(momdp[:yp])))
momdp[:r](;x=rand(support(momdp[:xp])),y=rand(support(momdp[:yp])),a=rand(support(momdp[:a])),xp=rand(support(momdp[:xp])),yp=rand(support(momdp[:yp])))
momdp[:r](;x=([1,1],),y=([0,1,1],),a=1,xp=([1,1],),yp=([1,1,1],))
btest = testrs.initial.rand()
bp = momdp[:mp](;a=2,m=btest,o=3,x=rand(support(momdp[:xp]))) #Check belief update correct

#support/pdf Test
sp = support(momdp[:xp];x=([1, 1],), y= (Bool[0, 0, 0],),a=2)[1]
pdf(momdp[:xp],sp;x=([1, 1],), y= (Bool[0, 0, 0],),a=2)
sp = support(momdp[:yp];x=([1, 1],), y= (Bool[0, 0, 0],),a=2)[1]
pdf(momdp[:yp],sp;x=([1, 1],), y= (Bool[0, 0, 0],),a=2)
pdf(momdp[:yp],([0,0,0],);x=([1, 1],), y= (Bool[0, 0, 0],),a=2)
o = support(momdp[:o];x=([1, 1],), y=(Bool[0, 0, 0],),a=1,xp=([1, 1],), yp=(Bool[0, 0, 0],))
for obs in o
    @show pdf(momdp[:o],obs;x=([1, 1],), y=(Bool[0, 0, 0],),a=1,xp=([1, 1],), yp=(Bool[0, 0, 0],))
end
support(momdp[:mp];a=2,m=btest,o=3,x=rand(support(momdp[:xp])))
pdf(momdp[:mp],bp;a=2,m=btest,o=3,x=rand(support(momdp[:xp])))
