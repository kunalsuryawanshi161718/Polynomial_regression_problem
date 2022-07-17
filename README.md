# Polynomial_regression_problem
Handling polynomial dataset
Pumpkin Price Polynomial Regression
Note: this notebook is a continuation on Pumpkin Price Linear Regression. You should start there.

Suppose that we have a target variable  y  and, for a single records of interest, a set of predictor variables  xn . A linear regression model solves for a sequence of weights,  w , which when multiplied against the data values, produces an estimated  y^ :

y^=w0+w1x1+w2x2+...
 
In OLS, this equation is solved to optimize a squared distance fitness metric. An optimal solution to this equation will minimize the square of the distance between  y  and  y^ , for some large number of records.

There are two ways of changing things around. The first way is to change the metric that we optimize for. For example, what if instead of solving for least squares, we solved for least absolute values? It's just a question of how mathematically difficult this is to do. Mathematically, least squares turn out to be the easiest metric possible to solve for.

The other way to change things is to change the model equation. This equation is linear because all of the weights  w  are first-order. This makes it easy to solve this equation using linear algebra matrices (see the previous notebook for this solution). However, this also assumes that our features are related in a linear way. Oftentimes, this is not true!

An easy way to extend regression to more complex feature relationships is to use a polynomial model. A second-order polynomial model (for two variables in these examples) looks like:

y^=w0+w1x1+w2x2+w3x1x2+w4x21+w5x22
 
I said earlier that equations are easiest to solve when they're linear, and this equation is no longer linear. What now?

We can use a cute trick to make it linear. Just define the following variables:

zn=[x1,x2,x1x2,x21,x22]
 
Then, relabeling the points:

y^=w0+w1z1+w2z2+w3z3+w4z4+w5z5
 
Tada! The equation is linear again. We can solve this equation using ordinary least squares, same as before, then "downcast" the  zn  variables into  xn  ones.

That's how polynomial regression works.

Now let's look at the scikit-learn implementation.

We'll use polynomial regression to estimate the size of pumpkins sold in New York City, given their average price. The next code cell transforms the data into the shape we need it in:
