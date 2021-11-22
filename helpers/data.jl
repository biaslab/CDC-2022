using Parameters

sigmoid(x) = 1 / (1 + exp(x))

function narx(y, u, h)
    y_ = y .^ 3
    u_ = 3 .* sin.(u)
    h_ = h
    z_ = mapreduce(x -> sum(x), +, [y_, u_, h_])
    z = mod(z_, 1)
    tanh(sign(z_)*z)
end

"""
y_k = f(Y_prev, U, E_prev) + e_k
"""
function generate_data(n_samples, f, orders, hyperparams)
    @unpack order_y, order_u, order_e = orders
    @unpack er_var, u = hyperparams

    h     = randn(order_e)
    y_obs = randn(order_y)
    y_lat = []

    for i in 1:n_samples
        err = sqrt(er_var)*randn()

        y_prev = y_obs[end:-1:end-order_y+1]
        e_prev = h[end:-1:end-order_e+1]

        push!(y_lat, f(y_prev, u[order_u+i-1:-1:i], e_prev))
        push!(y_obs, last(y_lat) + err)

        push!(h, err)

    end

    return y_lat, y_obs, h
end