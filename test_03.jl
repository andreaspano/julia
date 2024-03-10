
println("Hello Julia") 

using Pkg 
Pkg.activate(".")

using CSV
using DataFrames
using Random
using DecisionTree
using DataFramesMeta
using DynamicPipe
#using Statistics
using StatsModels
#using MLJ
#using MLJDecisionTreeInterface

println("Pkg loaded")



# mf = ModelFrame(@formula(y ~ 0 + x), df, contrasts = Dict(:x => DummyCoding())) |> modelmatrix

titanic_reader  = CSV.File("/home/andrea/dev/julia/titanic.csv"; header = 1);
#println(typeof(titanic_reader))

titanic = DataFrame(titanic_reader);


# remove missing values
titanic =  dropmissing(titanic);





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



# Train the decision tree model
fm = DecisionTreeClassifier( min_purity_increase=0.000001);
DecisionTree.fit!(fm, trn_feature, trn_target);
print_tree(fm);

#prd = apply_tree(fm, tst_feature)
#prd = apply_tree(fm, [0,1])
#DecisionTree.confusion_matrix(ytest, pred)



