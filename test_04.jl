1+1
using CSV 
using DataFramesMeta 
using MLJ
using DecisionTree
using UrlDownload
using VegaLite;
using Random;

url = "https://raw.githubusercontent.com/NicJC/Customers/main/Customers.csv"
data = urldownload(url) |> DataFrame;

#We can get the column names currently in the data frame, and then rename if needed:
getfield(data, :colindex)

# Now we can rename the columns, and print the resulting data frame with the pretty format (which requires the MLJ package)

rename!(data,
    :"Annual Income (\$)" => :Income ,
    :"Spending Score (1-100)" => :SpendingScore,
    :"Work Experience" => :WorkExperience,
    :"Family Size" => :FamilySize);
    
    

# Nice way to print data    
first(data,4) |> pretty


#Visualization
#Let’s look at a matrix plot to get an idea of what the data looks like:

data |>
@vlplot(
    repeat={
        row=[:Age, :Income, :WorkExperience],
        column=[:SpendingScore, :CustomerID, :FamilySize]
    },title="Matrix plot of Customers that purchase"
) +
@vlplot(
    :point,
    selection={
        brush={
            type=:interval,
            resolve=:union,
            on="[mousedown[event.shiftKey], window:mouseup] > window:mousemove!",
            translate="[mousedown[event.shiftKey], window:mouseup] > window:mousemove!",
            zoom="wheel![event.shiftKey]"
        },
        grid={
            type=:interval,
            resolve=:global,
            bind=:scales,
            translate="[mousedown[!event.shiftKey], window:mouseup] > window:mousemove!",
            zoom="wheel![!event.shiftKey]"
        }
    },
    x={field={repeat=:column}, type=:quantitative},
    y={field={repeat=:row}, type=:quantitative},
                    
    color={
        condition={
            selection=:brush,
            field=:Profession,
            type=:nominal
        },
        value=:grey
    }
)


#Let’s create a data frame to select a group of variables:


df = select(data, :CustomerID, :Age, :Income, :SpendingScore, :Profession,  :WorkExperience,  :FamilySize );
first(df,4)



# describe data
describe(df)


# split trn & tst
n = nrow(df);
k = floor(Int, 0.8 * n);

# trn
trn_idx = randperm(n)[1:k];
train = df[trn_idx, :];

# tst
tst_idx = setdiff(1:n, trn_idx);
test = df[tst_idx, :];

first(tst, 3) |> pretty



#Now we can create training and test sets:
## Figo unpack
ytrain, Xtrain = unpack(train, ==(:Profession), col->true);
ytest, Xtest = unpack(test, ==(:Profession), col->true);

#Then we need to ensure that the types are correct, typically the features will be floats and the targets should either be strings, floats or ordered factors …

ytrain = string.(ytrain);
ytest = string.(ytest);
Xtrain = float.(Xtrain);
Xtest = float.(Xtest);

#We need to then turn the Xtrain and Xtest features into a matrix:

Xtrain= Matrix(Xtrain[:,1:6])
Xtest= Matrix(Xtest[:,1:6])


# Model 
model = build_tree(ytrain, Xtrain)


# Now we can do some pruning:
model = prune_tree(model, 0.9)

# print tree
print_tree(model, 5)


#Prediction & accuracy:

pred = apply_tree(model, Xtest)

DecisionTree.confusion_matrix(ytest, pred)

# CV
n_folds = 3
r2 = nfoldCV_tree(ytrain, Xtrain, n_folds)
