# INFO-F409-EXAM-PROJECT
Project inspired from paper **"Learning to cooperate: Emergent communication in multi-agent navigation".**

One can run the training experiments by running the `runner.py` file. Currently, the layouts used for the first experiment are the "Pong", "Two room" and "Empty room"; and the layouts used for the second experiment are the "Four room" and "Flower". One can change the layout he/she wants the agents to train on by changing the values in the layouts list in the line 445 and 458 in the `runner.py` file.

If one wants to test a combination of trained sender(s)/receiver by playing games and printing the grid-world at each step to visually evaluate their performance, the `tester.py` shall be used.

If one wants to generate plots based on the results of the experiences realised with `runner.py`, the `plotter.py` file can be run. It can also export to a `csv` file the predictions from the trained models under given circumstances.
