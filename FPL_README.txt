PREDICTIVE MACHINE LEARNING FOR SPORTS VALUATION

An end to end Python data pipeline and machine learning script that detects market inefficiencies in Fantasy Premier League (FPL). By leveraging a feature Linear Regression model, this tool bypasses raw points leaderboard bias to uncover under rewarded "hidden gem" assets based on underlying physical match performance vectors.

In this code, it provides data and visualisation of a FPL player real season points vs their predicted FPL points
These predicted points are the points the player could have predictively gotten,
the higher the predicted points, the more underperforming the player is. This means that then player
could potentially outperform other fpl assets provided they take the chances they receive

The TOOLS/LIBRARIES USED IN THE CODE:

Python 3x is the programming language used
Requests utalizing the official live FPL unauthenticated data trees
Pandas for data processing and engineering
Numpy for data processing and engineering
Scikit-learn for machine learning
matplotlib for data visualization

REASON WHY WE USE THE > 180 CODE TO FILTER:

This code df["minutes"] > 180. is used in the code to specifically factor out players who have nor played less than 180 minutes total this season.
For example a player who has not played at all this season will not be tracked by the system as he has too small data from a significance perspective
Instead we look at and work with players who have played at least a minimum of 180 minutes this season

LEARNING METRICS FOR THE MACHINE LEARNING:

This model looks at both what makes a player gain points and what they loose points. For example a yellow cards deduct 1 point, a goal adds 4 points for a forward.
It then uses this to calculate the predicted points the player should have had at the given time in the season

RUN THIS CODE IF YOU DO NOT HAVE ALL LIBRARIES NEEDED:

pip install requests pandas scikit-learn matplotlib
