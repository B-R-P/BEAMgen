# Usage
Compile an example using the Compiler:
```console
python simplelang.py
```
Load the example into Erlang environment:
```console
$ erl
> code:add_path(".").
> code:load_file(bada).
> bada:add(1,2).
```
