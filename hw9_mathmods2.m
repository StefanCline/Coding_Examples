%% HW9 Math Models

a = sym('a')
f = @t x(t)*exp(a*t)

integral(f,0,10)