
1+8
using CSV
using DataFrames
using Random
using DecisionTree

titanic_reader  = CSV.File("/home/andrea/dev/julia/titanic.csv"; header = 1)
#println(typeof(titanic_reader))

titanic = DataFrame(titanic_reader);


# remove missing values
titanic =  dropmissing(titanic);

n = nrow(titanic);
k = floor(Int, 0.8 * n);

# trn
trn_idx = randperm(n)[1:k];
trn = titanic[trn_idx, :];

# tst
tst_idx = setdiff(1:n, trn_idx);
tst = titanic[tst_idx, :];

# titanic_reader

trn_feature = Matrix(trn[:, Not(:survived)]);
trn_target = trn[:, :survived];

# Train the decision tree model
fm0 = DecisionTreeClassifier(max_depth=1, min_samples_split=2)
DecisionTree.fit!(fm0, trn_feature, trn_target)



fm = DecisionTreeClassifier( min_samples_split=2, min_samples_leaf=1, min_purity_increase=0.001)

DecisionTree.fit!(fm, trn_feature, trn_target)

print_tree(fm)

