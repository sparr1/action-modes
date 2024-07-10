We would like to develop a relatively complete and mature system of transferrable action-abstractions from the ground up where as much is learned as possible. 

To do so, we introduce the action mode framework.

An action mode is a triple of 
1. mode support: a support of states on which the mode is defined
2. mode actions: a bounded, n-dimensional action space
3. projection: a function taking the mode actions to a base action space

To solve a problem with a collection of M completely defined action modes, you follow a four step process: 

1. take the state s you are in, and restrict your attention to the k supported modes for which s is in their mode support.
2. select a mode (let's call it mode i) from the k supported options.
3. select a latent action-within-mode z_i from the ith mode actions.
4. project z_i to a base action a, using the ith projection function. 
5. take action a in the base space, and repeat! 

Given that the mode supports and projection functions are completely packaged together with the modes, the only part left is to solve 2-3, which can essentially be solved via masking (using the supports) your favorite PAMDP solver. 

So now that we know how to use modes, that's great! But how did we get these projection functions and support sets to begin with? I'm glad you asked! 

There may be many way to get all kinds of different modes, but not all of the modes you end up with are going to be particularly useful. Right now, in order to learn modes, our current thinking is that it suffices to learn parameterized skills which solve average reward tasks on a restriction of the state space and a low-dimensional compression of the action space. These skills will become our projection functions. There will be some special auxillary losses which help these skills become useful.

The mode actions then become interpretable as saying "if I execute this mode, and this particular mode action, I will eventually settle in the high-dimensional base space, into a pattern of behavior which is high reward under this task. For instance, walking forwards, or rotating, etc. 