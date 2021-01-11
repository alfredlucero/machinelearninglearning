# Linear Regression

Fast, only train one time, then use for any predition, uses methods that will be important in other ML algorithms; lot harder to understand intuitively

Trying to figure out an equation to relate an independent variable to a dependent variable

i.e. given sqft lot size, trying to predict house price with a math equation - price = m \* lotsize + b

Why use this versus spreadsheets applications like Google Sheets with trendlines capability? We can use an arbitrary number of independent variables for a single output

Methods of solving linear regression

- Ordinary Least Squares, Generalized Least Squares, Gradient Descent

- Gradient Descent is the main one used in ML

- There is a mean squared error equation to tell you how wrong we were = 1/n (sum i to n)(guessi - actuali)^2 -> calculating distances from your guess line and actual value; can use this value to compare with the mean squared error of another equation to see if it's better or not; smaller the mean squared error, the better it is (unlikely to be exactly 0)

- For an equation like price = m \* lotsize + b, the better we guess m and b, the smaller the mean squared error; 'm' and 'b' will be as correct as can be when MSE is as low as possible

- Optimizing/finding the best coefficients 'm' and 'b'

  - Bruteforcing is not the move: The minimum value of the MSE for the guess of 'b' will be the ideal 'b'; not great to loop and guess 'b' since we don't know the possible range of 'b' (can be positive/negative), the step size for incrementing 'b' (can be small/large) - huge computational demands when adding in more features

  - knowing the the slope or rate of change of MSE is very valuable cause you can tell if you overshot or undershot the optimal b with lowest MSE (should be close to flat line for minimum MSE)

  - taking derivative of equation gives a new equation that tells us the slope at any location i.e. y = x^2 + 5 -> dy/dx = 2x; negative value means sloping down (positive sloping up); when slope is 0 can find the minimum

  - take derivative of MSE => 2/n (sum i to n) (b - actual i)

  - Gradient Descent

    - pick a value for 'b'

    - calculate slope of MSE with 'b'

    - is the slope very small? if yes, we are done

    - multiply the slope by an arbitrary small value called a 'learning rate' (adjusts how quickly we adjust 'b' to get to the optimal value) - to correct us better and control how much we change 'b' when we over/undershot the minimum, can potentially miss the optimal value of 'b' -> learning rate shouldn't be too high, need to optimize it depending on the data set

    - subtract that from 'b'

House price = m \* (sqft lot) + b

- We want to find an equation that relates an independent variable and a dependent variable

- We can guess at values of 'b', then use Mean Squared Error to figure out how wrong we were

- The slope, or rate of change, of MSE can be used to figure out whether or 'b' value was too high or low

- Take the slope of MSE at 'b' using the derivative of MSE equation -> why use the derivative rather than MSE directly in comparing 2 MSEs? calculating slope is the same as calculating MSE twice

- Subtract the slope from 'b'to update our guess

- We want slope of 0 so why not set the derivative equal to 0 and solve for b? - for finding a vertex point (minimum) because we've been ignoring 'm' in our solution so far

Mean Squared Error = 1/n (summation i to n)\*((mxi + b) - actuali)^2

- Take slope(derivative) of MSE with respect to b and take slope of MSE with respect to m

- Pick a value for 'b' and 'm'

- Calculate the slope of MSE with respect to 'm' and 'b'

- Are both slopes very small? If so, we are done

- Multiply both slopes by learning rate

- Subtract results from both 'b' and 'm'
