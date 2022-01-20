mutable struct GalacticCallback
    niter :: Int
    p
    GalacticCallback(maxiter) = new(0, ProgressMeter.Progress(maxiter))
end

function (callback::GalacticCallback)(args...)
    Core.println(args[2])
#     ProgressMeter.next!(callback.p)
    return false
end