mutable struct FixedMarginalConstraint <: ReactiveMP.AbstractFormConstraint
    fixed_value :: Any
end

function ReactiveMP.constrain_form(constraint::FixedMarginalConstraint, something)
    if constraint.fixed_value !== nothing
        return Message(constraint.fixed_value, false, false)
    else 
        return something
    end
end 

ReactiveMP.default_form_check_strategy(::FixedMarginalConstraint) = FormConstraintCheckLast()

ReactiveMP.is_point_mass_form_constraint(::FixedMarginalConstraint) = false