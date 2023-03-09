
# Prototype Wrapper Network

This is the repository for the paper *"Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes"*[^1].

The paper was published as a spotlight at ICLR 2023, Rwanda, see paper [here](https://openreview.net/forum?id=hWwY_Jq0xsN)

--------------

To reproduce the results for the Car Racing Domain Simply copy the whole repo and run these commands in the terminal whilst in the directory:

```
python3 -m venv pwnet
source pwnet/bin/activate
pip3 install --upgrade pip
pip install toml
pip install numpy
pip3 install torch torchvision torchaudio
pip install gym'[box2d]'
pip install tqdm
pip install gym==0.24.0
pip install gym-notices==0.0.7
pip install scikit-learn
```

Then run

```
Python collect_data.py
```

When you have the collected data you can now train the various wrappers. Simply run...

```
Python run_kmeans.py
Python run_pwnet*.py
Python run_pwnet.py
```

And the terminal will print off the results, reproducing the results for Car Racing from the paper.

------------------------

## To-do List
1. Add other domains from paper.
2. ... 



[^1]: Kenny, E.M., Tucker, M. and Shah, J., Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes. In *The Eleventh International Conference on Learning Representations.* Kigali, Rwanda, 2023.


