# ADCS with DRL

This repository includes the files for the attitude control 
simulation using Deep Reinforcement Learning techniques.

The project aims at providing a robust solution for the 
attitude control of a small 6U satellite. A self programmed version
of the PPO agent is employed to select the actions to take
to tackle and solve the problem. The action space is continuous, which means that the user defines the boundary of the control effort according to the available actuators specifications and the agent selects arbitrarily at each time step any value inside that boundary. 

After a short training of nearly 2 hours, the agent already achieved a 3-axis precision in the order of 1°.

<img src="RLADCS.gif" width="600" height="450">


