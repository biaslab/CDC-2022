mutable struct GalacticCallback
    niter :: Int
    p
    GalacticCallback(maxiter) = new(0, ProgressMeter.Progress(maxiter))
end

function (callback::GalacticCallback)(args...)
    ProgressMeter.next!(callback.p)
    return false
end