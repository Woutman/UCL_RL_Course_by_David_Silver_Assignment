# Easy21 Reinforcement Learning 
Implementation of several reinforcement learning algorithms according to the assignment of the UCL RL Course by David-Silver. The assignment can be found <a href="https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf">here</a>.

# Monte-Carlo Control in Easy21
After 10,000,000 episodes, the following value function was obtained:

<img src="https://github.com/Woutman/UCL_RL_Course_by_David_Silver_Assignment/blob/master/graphs/mc_vstar.png" width="500" height="600">

# TD Learning in Easy21
For every lambda between 0 and 1 with a 0.1 step interval, the mean square error was calculated against the true value function obtained through monte-carlo control. For each lambda, 10,000 episodes were run.

<img src=https://github.com/Woutman/UCL_RL_Course_by_David_Silver_Assignment/blob/master/graphs/td_learnratelambda0.0.png width="1000" height="500">

<img src=https://github.com/Woutman/UCL_RL_Course_by_David_Silver_Assignment/blob/master/graphs/td_learnratelambda1.0.png width="500" height="600">

<img src=https://github.com/Woutman/UCL_RL_Course_by_David_Silver_Assignment/blob/master/graphs/td_mselambda.png width="500" height="600">

# Linear Function Approximation in Easy21
A linear function approximation was implemented to replace the look up table of the previous assignment. For every lambda between 0 and 1 with a 0.1 step interval, the mean square error was calculated against the true value function obtained through monte-carlo control. For each lambda, 10,000 episodes were run.

<img src=https://github.com/Woutman/UCL_RL_Course_by_David_Silver_Assignment/blob/master/graphs/lfa_learnratelambda0.0.png width="500" height="600">

<img src=https://github.com/Woutman/UCL_RL_Course_by_David_Silver_Assignment/blob/master/graphs/lfa_learnratelambda1.0.png width="500" height="600">

<img src=https://github.com/Woutman/UCL_RL_Course_by_David_Silver_Assignment/blob/master/graphs/lfa_mselambda.png width="500" height="600">
