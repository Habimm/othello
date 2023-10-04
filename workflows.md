```sh
chmod u+x RUN
chmod u+x INSTALL
```

```sh
RAY_memory_usage_threshold=0.9 ipython3 oracle.py
RAY_memory_usage_threshold=1 ipython3 oracle.py
```

```sh
redis-cli keys "*"
redis-cli get "[[[0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0], [0, 1]], [[0, 0], [0, 0], [1, 0], [0, 0], [0, 1], [0, 0], [0, 1], [0, 0]], [[0, 0], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]], [[0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]"
```

Style JavaScript code.
```sh
bun install eslint
find -iname "*eslint*"
npm init @eslint/config
./node_modules/.bin/eslint --fix j.js
```

```sh
!git pull
```

Run the game AI in Google Colab:
```sh
!git clone -b mcts_without_ray --single-branch https://github.com/Habimm/othello.git
%cd othello/

# If you use `python -m venv` instead of `virtualenv` here,
# then you might get an error message related to `ensurepip`.
!python -m pip install virtualenv
!virtualenv .virtualenv_environment
!cat conda.yaml | awk '/pip:/ {flag=1; next} flag' | grep -E '^[[:space:]]+- ' | sed 's/^[[:space:]]*- //' > requirements.txt
!. .virtualenv_environment/bin/activate && pip install -r requirements.txt
!. .virtualenv_environment/bin/activate && python decompose_games.py
!. .virtualenv_environment/bin/activate && python train.py
!. .virtualenv_environment/bin/activate && python oracle.py
!. .virtualenv_environment/bin/activate && python play_against_random.py
!. .virtualenv_environment/bin/activate && python self_generation.py

!echo $VIRTUAL_ENV
!. .virtualenv_environment/bin/activate && echo $VIRTUAL_ENV
!. .virtualenv_environment/bin/activate && pip list
```
