



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

Run the game AI in Google Colab:
```sh
!git clone https://github.com/Habimm/othello.git
%cd othello/
!git checkout main
!git reset --hard origin/main
!git pull

env_var_commands = !python config.py | awk '{print "env", $3 "=" $4}'
for line in env_var_commands:
  get_ipython().run_line_magic(*line.strip().split(' ', 1))

!cat conda.yaml | awk '/pip:/ {flag=1; next} flag' | grep -E '^[[:space:]]+- ' | sed 's/^[[:space:]]*- //' > requirements.txt

!python -m venv .venv_environment
!. .venv_environment/bin/activate && pip install -r requirements.txt
!. .venv_environment/bin/activate && python decompose_games.py
!. .venv_environment/bin/activate && python train.py
!. .venv_environment/bin/activate && python oracle.py
!. .venv_environment/bin/activate && python play_against_random.py

!echo $VIRTUAL_ENV
!. .venv_environment/bin/activate && echo $VIRTUAL_ENV
!. .venv_environment/bin/activate && pip list
```
