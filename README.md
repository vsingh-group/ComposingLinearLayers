To setup
1. Clone this repo
2. Install torch_ga_fix in a different directory with ```pip install .```
3. Clone and install fast-hadamard-transofrm in a different directory (https://github.com/Dao-AILab/fast-hadamard-transform.git) with ```pip install .```
4. Install requirements.txt with ```pip install -r requirements.txt```
5. Set HUGGINGTOKEN environment variable

To run, be in the rotor directory and run the corresponding sh file in the run directory depending on which LLM you want to replace layers of.
If you are getting weird errors regarding hugging face, make sure you are logged into hugging face in the terminal with "huggingface-cli login"
For the projection experiment in appendix D, run "python -m run.test_projection_convergence" to get the data