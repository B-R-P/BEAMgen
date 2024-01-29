from compiler import *

custom_module = Module()
x_param = Param(name="x",type=Type.Int,index=0)
y_param = Param(name="y",type=Type.Int,index=1)
func1 = Func(
    name="add",
    params={x_param.name: x_param,y_param.name: y_param},
    body=Binop(BinopKind.Sum,Var(x_param.name),Var(y_param.name))
)
custom_module.funcs[func1.name] = func1
with open("bada.beam","wb") as output_file:
	output_file.write(generate_output_bytes(custom_module))
