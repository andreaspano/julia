###########################################

println("Hello Julia") 

using Pkg 
Pkg.activate(".")

using CSV
using DataFrames
using Random
using Downloads
using ARFFFiles
using ScientificTypes

#using DecisionTree
using DataFramesMeta
using DynamicPipe
#using Statistics
#using StatsModels
using MLJ
using MLJDecisionTreeInterface


println("Pkg loaded")


# mf = ModelFrame(@formula(y ~ 0 + x), df, contrasts = Dict(:x => DummyCoding())) |> modelmatrix

titanic_reader  = CSV.File("/home/andrea/dev/julia/titanic.csv"; header = 1);
titanic = DataFrame(titanic_reader);

select!(titanic, Not([:survied, :age]))


# remove missing values
titanic =  dropmissing(titanic);

schema(titanic)

titanic = @transform(titanic, 
    :class=categorical(:class), 
    :sex=categorical(:sex),  
    :survived=categorical(:survived)
    )

# sceham definisce typeof per ogni variabile
#schema(titanic)


# target and features
y, X = unpack(titanic, ==(:survived), rng = 123)

schema(X)

# partitiont training & test 
(X_trn, X_tst), (y_trn, y_tst)  = partition((X, y), 0.75, multi=true,  rng=123)

#MyDecisionTree = @load DecisionTreeClassifier pkg = "DecisionTree" 
mod = @load DecisionTreeClassifier pkg = "DecisionTree" 


#dc
fm = mod() 


fm_mach = machine(fm, X_trn, y_trn)
fit!(fm_mach)



yhat = predict(dc_mach, Xtest);

yhat_classes = predict_mode(dc_mach, Xtest) ;

correct_classes = count(ytest .== yhat_classes)
accuracy = correct_classes / length(ytest) 




######################################


# to be removed
titanic = @chain titanic begin
    @select  :survived :sex :class
    #@transform :survived = categorical(:survived)
    #@transform :sex = categorical(:sex)
end;

first(titanic, 3)






n = nrow(titanic);
k = floor(Int, 0.8 * n);

# trn
trn_idx = randperm(n)[1:k];
trn = titanic[trn_idx, :];
trn_feature = ModelFrame(@formula(survived ~ 0 + sex + class), trn, contrasts = Dict(:sex => DummyCoding(), :class => DummyCoding())) |> modelmatrix
#trn_feature = Matrix(trn[:, Not(:survived)]);
trn_target = trn[:, :survived];


# tst
tst_idx = setdiff(1:n, trn_idx);
tst = titanic[tst_idx, :];
tst_feature = ModelFrame(@formula(survived ~ 0 + sex), tst, contrasts = Dict(:sex => DummyCoding())) |> modelmatrix;
#trn_feature = Matrix(trn[:, Not(:survived)]);
tst_target = tst[:, :survived];




#rawdata = DataFrame(make=["honda", "ford", "subaru"], mpg=[33,29,31], price=[30000,25000,27500])
@load DecisionTreeRegressor pkg=DecisionTree
pp = @pipeline OneHotDTR(
                X -> coerce(X, :make=>Multiclass),
                hot = OneHotEncoder(),
                dtc = DecisionTreeRegressor())
X = select(rawdata, Not(:price))
y = rawdata.price
dtm = machine(pp, X, y)
fit!(dtm)