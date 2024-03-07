using Random
using Flux
Random.seed!(123)

x = Float32.([0.1 10; 0.13 13; 0.17 17; 0.2 20; 1 1; 1.3 1.3; 1.7 1.7; 2 2; 10 0.1; 13 0.13; 17 0.17; 20 0.2]')
y = [1,1,1,1,2,2,2,2,3,3,3,3] |>
        i -> Flux.onehotbatch(i, 1:3) .|> 
        Float32
nndata = Flux.Data.DataLoader((x, y), batchsize=3,shuffle=true)

Flux_nn = Chain(
    Dense(2, 10, relu),
    Dense(10, 3),    # no relu here
    softmax
)
ps = Flux.params(Flux_nn)

loss(x, y) = Flux.crossentropy(Flux_nn(x), y)

opt = ADAM()

Flux.@epochs  50 Flux.train!(loss, ps, nndata, opt)     

acc = sum(Flux.onecold(Flux_nn(x), 1:3) .== Flux.onecold(y, 1:3)) / size(y, 2) # 1.0