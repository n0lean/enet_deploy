#qlua demo.lua -i videos/0006R0.MXF -d ./ -m model --net 1
#qlua demo_sim.lua -d ./ -m model --net best-v2
qlua demo_sim.lua -d ./model/ -m new --net best-v4

